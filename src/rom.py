"""Local POD bases, reduced quadratic operators, and the cluster-routed ensemble.

The chemistry is quadratic, ``dx/dt = A x + B (x kron x)``, with ``A`` and ``B`` rebuilt per hydro interval. Each cluster carries a local basis

    encode:  z = U^T D^-1 (x - x_bar)
    decode:  x = x_bar + D U z

for POD basis ``U``, mean state ``x_bar`` and diagonal species scaling ``D``.

A tracer is solved piecewise: route, integrate one ``dt_hydro`` in that cluster's coordinates, hand the end state to the next step.
"""
from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .cluster import block_starts

#: Below this many blocks a process pool costs more than it saves.
_PAR_MIN = 1000


def apply_B_xy(x, y, B):
    """``out[i] = sum_jk B[i,j,k] x[j] y[k]``, with ``B`` stored as ``(N, N*N)``."""
    x = np.asarray(x, dtype=np.float64)
    if B is None or not B.nnz:
        return np.zeros_like(x)
    N = x.size
    Bcoo = B.tocoo(copy=False)
    out = np.zeros(N, dtype=np.float64)
    np.add.at(out, Bcoo.row,
              Bcoo.data * x[Bcoo.col // N] * np.asarray(y)[Bcoo.col % N])
    return out


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

class Basis:
    """One cluster's local coordinate system."""

    def __init__(self):
        self.scaler = None
        self.pca = None
        self.rank = 0
        self.var_threshold = None
        # Precomputed coordinate-space matrices (see _precompute).
        self._scale = None
        self._x_bar = None
        self._Q_full = None    # every PCA row, (n_pca, n_species)
        self._R_full = None    # D[:,None] * Q_full.T
        self._L_full = None    # Q_full / D[None,:]
        self._R = self._L = None   # rank-dependent slices

    def _precompute(self):
        scale = self.scaler.scale_
        self._scale = scale
        self._x_bar = self.scaler.mean_ + scale * self.pca.mean_
        self._Q_full = np.asarray(self.pca.components_, dtype=np.float64)
        self._R_full = scale[:, None] * self._Q_full.T
        self._L_full = self._Q_full / scale[None, :]
        self._slice()

    def _slice(self):
        self._R = self._R_full[:, :self.rank]
        self._L = self._L_full[:self.rank, :]

    def set_rank(self, rank):
        """Change the active rank; no refit, the PCA is already full-rank."""
        rank = int(rank)
        if not 0 <= rank <= self.pca.components_.shape[0]:
            raise ValueError(f"rank must be in [0, "
                             f"{self.pca.components_.shape[0]}]; got {rank}")
        self.rank = rank
        self._slice()
        return self

    @property
    def max_rank(self):
        return int(self.pca.components_.shape[0])

    @property
    def n_species(self):
        return self.scaler.mean_.size

    def encode(self, x):
        """``z = U^T D^-1 (x - x_bar)``, for a vector or ``(n_species, n)`` block.

        Centres before projecting: the same result as ``L@x - L@x_bar``, without cancelling two large numbers to reach a small coordinate.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return self._L @ (x - self._x_bar)
        return self._L @ (x - self._x_bar[:, None])

    def decode(self, z):
        """``x = x_bar + D U z``.

        A diverged solve can return an empty or truncated reduced state; hand back an empty block so the caller can report it via ``sol.status``.
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            if z.size < self.rank:
                return np.empty((self.n_species, 0))
            return self._x_bar + self._R @ z
        if z.shape[1] == 0 or z.shape[0] < self.rank:
            return np.empty((self.n_species, 0))
        return self._x_bar[:, None] + self._R @ z


def fit_basis(X, var_threshold, scale_method="pareto"):
    """Fit one local basis from a snapshot block ``(n_samples, n_species)``.

    A full-rank PCA on the scaled block; the active rank is the smallest reaching ``var_threshold``.

    ``"pareto"`` divides by ``sqrt(std)``: full z-scoring would weight a near-constant species like one spanning ten decades, and the square root keeps trace species visible without letting them dominate. ``"none"`` leaves the basis orthonormal in physical space, which is what a homogeneous state vector wants.
    """
    scaler = StandardScaler().fit(X)
    scaler.scale_ = (np.sqrt(scaler.scale_) if scale_method == "pareto"
                     else np.ones_like(scaler.scale_))
    scaler.var_ = scaler.scale_ ** 2

    X_scaled = (X - scaler.mean_) / scaler.scale_
    n_comp = min(X.shape[1], X_scaled.shape[0])
    try:
        pca = PCA(n_components=n_comp, svd_solver="full").fit(X_scaled)
    except np.linalg.LinAlgError:
        # A near-rank-deficient block can make the data-matrix SVD fail to converge; the covariance eigendecomposition is robust and equivalent for n_samples >> n_features.
        pca = PCA(n_components=n_comp, svd_solver="covariance_eigh").fit(X_scaled)

    # Pin the POD sign gauge: U and -U decode identically, but sklearn picks the sign from memory layout, so stored operators would otherwise differ run to run.
    comp = pca.components_
    if comp.size:
        lead = np.argmax(np.abs(comp), axis=1)
        sign = np.sign(comp[np.arange(comp.shape[0]), lead])
        sign[sign == 0] = 1.0
        pca.components_ = comp * sign[:, None]

    cumvar = np.cumsum(np.nan_to_num(pca.explained_variance_ratio_))
    rank = (n_comp if var_threshold >= 1
            else min(int(np.searchsorted(cumvar, var_threshold)) + 1, n_comp))

    basis = Basis()
    basis.scaler, basis.pca = scaler, pca
    basis.rank, basis.var_threshold = int(rank), float(var_threshold)
    basis._precompute()
    return basis


# ---------------------------------------------------------------------------
# Reduced model
# ---------------------------------------------------------------------------

@dataclass
class ReducedOperators:
    """Reduced quadratic operators for one ROM in one fixed environment."""
    f0: np.ndarray
    Ar: np.ndarray
    Bred: np.ndarray
    Bsym: np.ndarray = None       # Bred[i,l,b] + Bred[i,b,l], for the Jacobian

    def __post_init__(self):
        if self.Bsym is None:
            self.Bsym = self.Bred + self.Bred.transpose(0, 2, 1)


class ROM:
    """Single-cluster reduced model: one :class:`Basis` plus its dynamics."""

    def __init__(self, basis):
        self.basis = basis
        self._bi = self._bj = self._bk = None
        self._L_bi = self._G = None
        self._xbar_bj = self._xbar_bk = None
        self._cached_rank = None

    def set_B_structure(self, bi, bj, bk):
        """Cache the B-tensor sparsity slices for the fast operator build."""
        self._bi, self._bj, self._bk = bi, bj, bk
        self._slice_B()

    def _slice_B(self):
        """Re-cut the rank-dependent slices against the basis's current rank."""
        b = self.basis
        self._L_bi = b._L[:, self._bi]
        R_bj, R_bk = b._R[self._bj], b._R[self._bk]
        # G[n, j*m+k] = R_bj[n,j] * R_bk[n,k] is independent of the rates, so folding those into L_bi and `@ G` makes Bred a single BLAS gemm.
        self._G = (R_bj[:, :, None] * R_bk[:, None, :]).reshape(R_bj.shape[0], -1)
        self._R_bj, self._R_bk = R_bj, R_bk
        self._xbar_bj, self._xbar_bk = b._x_bar[self._bj], b._x_bar[self._bk]
        self._cached_rank = b.rank

    def build_operators(self, A, B):
        """Project the full quadratic operators into this ROM's coordinates."""
        b = self.basis
        x_bar, R, L, m = b._x_bar, b._R, b._L, b.rank

        f0 = L @ (np.asarray(A @ x_bar).ravel() + apply_B_xy(x_bar, x_bar, B))
        A_R = np.asarray(A @ R)

        if self._bi is not None and B is not None and B.nnz and m:
            if self._cached_rank != m:
                self._slice_B()
            Bcoo = B.tocoo(copy=False)
            # The cached indices are the superset pattern; if this environment's B differs, the rates misalign and every reduced operator is silently wrong.
            if Bcoo.data.size != self._bi.size:
                raise RuntimeError(
                    f"B nnz={Bcoo.data.size} != cached structure nnz="
                    f"{self._bi.size}: this network's sparsity pattern is "
                    "environment-dependent, so precompute_B_structure() is "
                    "invalid for it.")
            bv = Bcoo.data.astype(np.float64)

            mixed = bv[:, None] * (self._xbar_bj[:, None] * self._R_bk
                                   + self._R_bj * self._xbar_bk[:, None])
            B_Ar = np.zeros_like(A_R)
            np.add.at(B_Ar, self._bi, mixed)
            Ar = L @ (A_R + B_Ar)
            Bred = ((self._L_bi * bv[None, :]) @ self._G).reshape(m, m, m)
        else:
            Ar_full = np.empty_like(A_R)
            for j in range(m):
                Rj = R[:, j]
                Ar_full[:, j] = (A_R[:, j] + apply_B_xy(x_bar, Rj, B)
                                 + apply_B_xy(Rj, x_bar, B))
            Ar = L @ Ar_full
            Bred = np.zeros((m, m, m), dtype=np.float64)
            if B is not None and B.nnz and m:
                Bcoo = B.tocoo(copy=False)
                N = b.n_species
                bi = Bcoo.row.astype(np.int64)
                bj, bk = Bcoo.col // N, Bcoo.col % N
                Bred = np.einsum("pn,n,nj,nk->pjk", L[:, bi],
                                 np.asarray(Bcoo.data, dtype=np.float64),
                                 R[bj], R[bk], optimize=True)

        return ReducedOperators(f0=f0, Ar=Ar, Bred=Bred)

    @staticmethod
    def _rhs(ops, z):
        return ops.f0 + ops.Ar @ z + (ops.Bred @ z) @ z

    @staticmethod
    def _jac(ops, z):
        return ops.Ar + ops.Bsym @ z

    def solve(self, A, B, x0, t_span, atol, rtol, t_eval=None,
              method="BDF"):
        """Integrate one fixed-environment interval.

        Returns ``(t, y, sol)``. A failed solve returns what it completed rather than raising, so check ``sol.status``.
        """
        ops = self.build_operators(A, B)
        z0 = np.asarray(self.basis.encode(x0), dtype=np.float64)
        if t_eval is not None:
            # A stored knot can land a few ULPs outside t_span, which solve_ivp rejects outright; clipping is a no-op otherwise.
            t_eval = np.clip(np.asarray(t_eval, dtype=np.float64),
                             min(t_span), max(t_span))
        # Only the implicit solvers use a Jacobian; handing one to an explicit method is ignored and warned about on every step.
        jac = ({"jac": lambda _t, z: self._jac(ops, z)}
               if method in ("BDF", "Radau", "LSODA") else {})
        sol = solve_ivp(lambda _t, z: self._rhs(ops, z), t_span, z0,
                        method=method, atol=atol, rtol=rtol, t_eval=t_eval,
                        **jac)
        return sol.t, self.basis.decode(sol.y), sol


# ---------------------------------------------------------------------------
# Error criteria
# ---------------------------------------------------------------------------

def qoi_error(y_rom, y_fom, qoi_rows, norm="relative"):
    """Worst-QoI L2-in-time error between two ``(n_species, n)`` blocks.

    Scoring by the worst QoI means a cluster clears the tolerance only when every QoI does.

    ``"relative"`` divides by ``||y_fom||``, so the tolerance reads as a fraction. ``"absolute"`` is the RMS-in-time error in state units, for a signed state whose denominator can collapse on a block where a QoI sits near zero throughout -- the max reduction would then let that block drive splitting toward a tolerance nothing can reach.
    """
    d = np.asarray(y_rom)[qoi_rows] - np.asarray(y_fom)[qoi_rows]
    if norm == "absolute":
        return float(np.max(np.sqrt(np.mean(d ** 2, axis=1))))
    num = np.sqrt(np.sum(d ** 2, axis=1))
    den = np.sqrt(np.sum(np.asarray(y_fom)[qoi_rows] ** 2, axis=1)) + 1e-300
    return float(np.max(num / den))


def proj_error_by_rank(basis, y_fom, qoi_rows, norm="relative"):
    """The projection error of one block at every rank, in one pass.

    ``err[r]`` is what :func:`qoi_error` would give for ``decode(encode(y_fom))`` at rank ``r``. Accumulated incrementally, so the whole sweep costs about one reconstruction, which is what makes rank escalation affordable.

    A heuristic, not a bound: the projection is optimal in the scaled L2 norm while this measures relative error in physical units, so a rank that clears it still needs confirming by a ROM solve.
    """
    Y = np.asarray(y_fom, dtype=np.float64)
    scaler, pca = basis.scaler, basis.pca
    C = (Y.T - scaler.mean_) / scaler.scale_ - pca.mean_
    coef = C @ pca.components_.T
    n_snap = Y.shape[1]
    den = (np.full(qoi_rows.size, np.sqrt(n_snap)) if norm == "absolute"
           else np.sqrt(np.sum(Y[qoi_rows] ** 2, axis=1)) + 1e-300)
    res = C[:, qoi_rows].copy()
    w = scaler.scale_[qoi_rows]
    M = pca.components_[:, qoi_rows]

    err = np.empty(pca.components_.shape[0] + 1)
    for r in range(err.size):
        if r:
            res -= np.outer(coef[:, r - 1], M[r - 1])
        err[r] = float(np.max(np.sqrt(np.sum((w * res) ** 2, axis=0)) / den))
    return err


# ---------------------------------------------------------------------------
# Worker pools
# ---------------------------------------------------------------------------

_score_state = None


def _score_init(network, roms, feat_path, sp, pslice, dt_hydro, atol, rtol,
                qoi_rows, method, norm):
    """Pool initializer: hold the bases, reopen the memmap read-only per worker."""
    global _score_state
    _score_state = dict(network=network, roms=roms,
                        mm=np.load(feat_path, mmap_mode="r"), sp=sp,
                        pslice=pslice, dt_hydro=float(dt_hydro), atol=atol,
                        rtol=rtol, qoi_rows=np.asarray(qoi_rows, dtype=int),
                        method=method, norm=norm)


def _score_block(args):
    """Score one ``(tracer, step)`` block: one-step local ROM error vs the FOM."""
    b, rows, teval, label = args
    s = _score_state
    rom = s["roms"].get(int(label))
    if rom is None:
        return b, float("inf")
    y_fom = np.asarray(s["mm"][rows][:, s["sp"]], dtype=np.float64).T
    net = s["network"]
    A, B = net.operators(net.env(
        np.asarray(s["mm"][rows[0]][s["pslice"]], dtype=np.float64)))
    try:
        _, y_rom, sol = rom.solve(A, B, y_fom[:, 0], (0.0, s["dt_hydro"]),
                                  s["atol"], s["rtol"], t_eval=teval,
                                  method=s["method"])
    except Exception:
        return b, float("inf")
    if sol.status != 0 or y_rom.shape[1] != y_fom.shape[1]:
        return b, float("inf")
    return b, qoi_error(y_rom, y_fom, s["qoi_rows"], s["norm"])


_fit_state = None


def _fit_init(feat_path, species_slice):
    global _fit_state
    _fit_state = (np.load(feat_path, mmap_mode="r"), species_slice)


def _fit_one(args):
    c, idx, var_threshold, scale_method = args
    mm, sp = _fit_state
    return c, fit_basis(np.asarray(mm[idx][:, sp], dtype=np.float64),
                        var_threshold, scale_method), idx.size


_solve_ensemble = None


def _solve_init(ensemble):
    global _solve_ensemble
    _solve_ensemble = ensemble


def _solve_one(args):
    i, pt, x0, dt_hydro, atol, rtol, t_eval, positivity, method = args
    try:
        return i, _solve_ensemble.solve_tracer(pt, x0, dt_hydro, atol, rtol,
                                               t_eval=t_eval,
                                               positivity=positivity,
                                               method=method)
    except Exception as exc:
        return i, exc


# ---------------------------------------------------------------------------
# Positivity projection
# ---------------------------------------------------------------------------

def project_nonnegative(z0, x_bar, R, scale):
    """Nearest reduced state to ``z0`` that decodes to a nonnegative abundance.

    ``min 0.5||z - z0||^2  s.t.  x_bar + R z >= 0``, via SLSQP.

    Positivity is imposed in the basis's scaled units. SLSQP's ``ftol`` is absolute and governs constraint satisfaction, so on a state spanning many decades an unscaled constraint is inert for trace species: a violation far above their own magnitude still reads as feasible, and ``z0`` comes back untouched.
    """
    scale = np.asarray(scale, dtype=np.float64)
    pos_A, pos_b = R / scale[:, None], x_bar / scale
    res = minimize(lambda z: 0.5 * np.sum((z - z0) ** 2), z0,
                   jac=lambda z: z - z0, method="SLSQP",
                   constraints=[{"type": "ineq",
                                 "fun": lambda z: pos_b + pos_A @ z,
                                 "jac": lambda z: pos_A}],
                   options={"maxiter": 200, "ftol": 1e-14})
    return res.x, bool(res.success)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleROM:
    """Cluster-routed ensemble of local ROMs."""

    def __init__(self, cluster, network, roms):
        self.cluster = cluster
        self.network = network
        self.roms = dict(roms)
        self.summary = {}

    # ---- build ----------------------------------------------------------

    @classmethod
    def build_adaptive(cls, cluster, network, fm, n_steps, dt_hydro, cfg,
                       split_tols, qoi_indices, verbose=True):
        """Fit one ensemble per split tolerance, sharing one base pass.

        From ``cluster``'s flat partition, repeatedly: take the worst cluster still over tolerance, split it in two, refit both children and re-score only the steps routed to the parent. Stops when every cluster is under tolerance, none can be split, or ``max_clusters`` is reached.

        A cluster's error is the max over its steps of the one-step ROM error: integrate one ``dt_hydro`` from the FOM state at the step start and score against the FOM. Restarting from the true state isolates the cluster's own dynamics from upstream drift; a failed step counts as ``+inf``, so its cluster is split first.

        A cluster over tolerance that cannot be split has its rank raised instead when ``cfg.increase_rank`` is set, and is abandoned otherwise.

        The base fit and first scoring pass are tolerance-independent, so they run once and each tolerance restores from that snapshot. Returns one ``(ensemble, cluster)`` pair per tolerance; ``cluster`` is not mutated.
        """
        mm, sp, pslice = fm.memmap, fm.species_slice, fm.param_slice
        qoi_rows = np.unique(np.asarray(qoi_indices, dtype=int))
        n_w = int(cfg.n_workers)
        var_threshold = cfg.var_threshold
        scale_method, norm = cfg.scale_method, cfg.error_norm
        rank_proxy = cfg.rank_proxy

        # ---- group valid rows into (tracer, step) blocks ----
        # A block shares one hydro step, hence one cluster label, so it is the unit the one-step solve operates on.
        valid = np.flatnonzero(fm.valid_rows)
        # Keyed on (tracer, step) rather than step alone, matching the boundary Cluster fitted its labels against.
        cuts = np.flatnonzero(block_starts(np.asarray(fm.sample_tracer)[valid],
                                           np.asarray(fm.sample_step)[valid]))[1:]
        block_rows, block_teval = [], []
        for rows in np.split(valid, cuts):
            times = np.asarray(mm[rows][:, 1], dtype=np.float64)
            srt = np.argsort(times)
            teval = np.unique(times[srt] - times[srt][0])
            if teval.size < 2:
                continue                       # need >= 2 points to compare
            block_rows.append(rows[srt])
            block_teval.append(teval)
        n_blocks = len(block_rows)

        sample_labels = cluster.labels_for_rows(fm).copy()
        sample_labels[~fm.valid_rows] = -1

        def score(block_ids, roms_subset, out):
            ids = np.asarray(list(block_ids), dtype=np.int64)
            if ids.size == 0:
                return
            args = [(int(b), block_rows[b], block_teval[b], int(block_label[b]))
                    for b in ids]
            init = (network, roms_subset, fm.path, sp, pslice, dt_hydro,
                    cfg.atol, cfg.rtol, qoi_rows, cfg.ode_method, norm)
            if n_w == 1 or ids.size < _PAR_MIN:
                _score_init(*init)
                for a in args:
                    b, e = _score_block(a)
                    out[b] = e
                return
            with ProcessPoolExecutor(max_workers=n_w, initializer=_score_init,
                                     initargs=init) as pool:
                for b, e in pool.map(_score_block, args,
                                     chunksize=max(1, len(args) // (n_w * 8))):
                    out[b] = e

        # ---- 1. one basis per starting cluster ----
        if verbose:
            print(f"  [adaptive] fitting {cluster.n_clusters} base bases ...")
        roms = _fit_bases(fm, sample_labels, cluster.n_clusters,
                          var_threshold, scale_method, n_w, verbose)
        block_label = np.array([sample_labels[rows[0]] for rows in block_rows],
                               dtype=np.int64)
        blocks_of = {}
        for b, lab in enumerate(block_label):
            blocks_of.setdefault(int(lab), []).append(b)

        # ---- 2. one full ROM-scoring pass ----
        block_err = np.full(n_blocks, np.inf)
        if verbose:
            print(f"  [adaptive] ROM-scoring {n_blocks:,} tracer-steps across "
                  f"{cluster.n_clusters} clusters ...")
        score(range(n_blocks), roms, block_err)

        def cluster_error(c):
            bs = blocks_of.get(c, [])
            return float(np.max(block_err[bs])) if bs and c in roms else None

        base = SimpleNamespace(
            roms=roms, blocks_of=blocks_of, block_label=block_label,
            block_err=block_err, sample_labels=sample_labels,
            errors={c: cluster_error(c) for c in blocks_of},
            sizes={c: int(np.count_nonzero(sample_labels == c))
                   for c in blocks_of},
            template=cluster.clone())

        # Independent, order-invariant split streams per tolerance.
        seeds = (np.random.SeedSequence(cfg.rng_seed).spawn(len(split_tols))
                 if len(split_tols) > 1 else [cfg.rng_seed])

        outputs = []
        for ti, tau in enumerate(split_tols):
            cluster = base.template.clone()
            roms = copy.deepcopy(base.roms)
            blocks_of = {c: list(v) for c, v in base.blocks_of.items()}
            block_label = base.block_label.copy()
            block_err = base.block_err.copy()
            sample_labels = base.sample_labels.copy()
            errors, sizes = dict(base.errors), dict(base.sizes)
            rng = np.random.default_rng(seeds[ti])

            blocked, rank_increases = set(), {}

            def raise_rank(c):
                """Raise cluster ``c``'s rank until its ROM error meets ``tau``.

                The projection proxy costs no ODE solves, so it picks the lowest rank that could plausibly clear the tolerance and the walk starts there; each candidate is still confirmed by real solves on the same blocks. The best rank seen is kept even if none reaches ``tau``. With ``cfg.rank_proxy`` off the walk starts at ``r0 + 1`` and re-scores every rank.
                """
                rom, bs = roms.get(c), blocks_of.get(c, [])
                if rom is None or not bs or rom.basis.rank >= rom.basis.max_rank:
                    return
                r0, basis = rom.basis.rank, rom.basis
                if rank_proxy:
                    proj = np.max([proj_error_by_rank(
                        basis, np.asarray(mm[block_rows[b]][:, sp],
                                          dtype=np.float64).T, qoi_rows, norm)
                        for b in bs], axis=0)
                    clears = np.flatnonzero(proj[r0 + 1:] <= tau)
                    start = (r0 + 1 + int(clears[0]) if clears.size
                             else basis.max_rank)
                else:
                    start = r0 + 1

                best_rank, best_err = r0, errors.get(c)
                best_scores = {b: block_err[b] for b in bs}
                for r in range(start, basis.max_rank + 1):
                    basis.set_rank(r)
                    score(bs, {c: rom}, block_err)
                    err = cluster_error(c)
                    if err is not None and (best_err is None or err < best_err):
                        best_rank, best_err = r, err
                        best_scores = {b: block_err[b] for b in bs}
                    if err is not None and err <= tau:
                        break
                basis.set_rank(best_rank)
                block_err[list(best_scores)] = list(best_scores.values())
                errors[c] = best_err
                if best_rank != r0:
                    rank_increases[c] = (r0, best_rank)
                    if verbose:
                        print(f"    cluster {c}: rank {r0} -> {best_rank}, "
                              f"err={best_err:.4g}")

            n_splits, stop_reason = 0, "all clusters under tol"
            while cluster.n_clusters < cfg.max_clusters:
                cands = [(e, c) for c, e in errors.items()
                         if e is not None and e > tau and c not in blocked]
                if not cands:
                    if any(e is not None and e > tau for e in errors.values()):
                        stop_reason = ("clusters over tol but unsplittable "
                                       "(cannot form two children >= "
                                       f"min_cluster_size={cfg.min_cluster_size})")
                    break
                parent_err, c = max(cands)

                new_id = cluster.split_cluster(c, rng, cfg.min_cluster_size)
                if new_id is None:
                    blocked.add(c)
                    if cfg.increase_rank:
                        raise_rank(c)
                    continue

                affected = blocks_of.pop(c, [])
                sample_labels = cluster.labels_for_rows(fm)
                sample_labels[~fm.valid_rows] = -1
                for cid in (c, new_id):
                    idx = np.flatnonzero(sample_labels == cid)
                    sizes[cid] = int(idx.size)
                    if idx.size >= 2:
                        roms[cid] = ROM(fit_basis(
                            np.asarray(mm[idx][:, sp], dtype=np.float64),
                            var_threshold, scale_method))
                    else:
                        roms.pop(cid, None)
                    blocks_of[cid] = []
                for b in affected:
                    lab = int(sample_labels[block_rows[b][0]])
                    block_label[b] = lab
                    blocks_of.setdefault(lab, []).append(b)

                children = {cid: roms[cid] for cid in (c, new_id) if cid in roms}
                score([b for cid in (c, new_id) for b in blocks_of.get(cid, [])],
                      children, block_err)
                for cid in (c, new_id):
                    errors[cid] = cluster_error(cid)
                n_splits += 1
                if verbose:
                    over = sum(1 for e in errors.values()
                               if e is not None and e > tau)
                    print(f"  [tau={tau:g}] split {n_splits}: {c} "
                          f"(err={parent_err:.4g}) -> {c},{new_id}; "
                          f"{cluster.n_clusters} clusters, {over} over tol")
            else:
                stop_reason = f"reached max_clusters={cfg.max_clusters}"
                # Raising rank adds no clusters, so anything still over tol can improve even though no further split is allowed.
                if cfg.increase_rank:
                    for cid in sorted(c for c, e in errors.items()
                                      if e is not None and e > tau
                                      and c not in blocked):
                        raise_rank(cid)

            finite = [e for e in errors.values() if e is not None]
            over_ids = sorted(c for c, e in errors.items()
                              if e is not None and e > tau)
            ens = cls(cluster=cluster, network=network, roms=roms)
            ens.summary = {
                "split_tol": float(tau),
                "n_clusters": int(cluster.n_clusters),
                "n_splits": int(n_splits),
                "n_over_tol": len(over_ids),
                "n_unsplittable": len(blocked),
                "worst_err": float(max(finite)) if finite else float("nan"),
                "stop_reason": stop_reason,
                "rank_increases": {int(c): [int(a), int(b)]
                                   for c, (a, b) in rank_increases.items()},
                "cluster_ranks": {int(c): int(r.basis.rank)
                                  for c, r in roms.items()},
            }
            if verbose:
                print(f"  [tau={tau:g}] done: {n_splits} splits, "
                      f"{cluster.n_clusters} clusters, {len(over_ids)} over tol, "
                      f"worst err={ens.summary['worst_err']:.4g} "
                      f"({stop_reason})")
            outputs.append((ens, cluster))
        return outputs

    # ---- persistence ----------------------------------------------------

    def save(self, path):
        """Save each cluster's fitted basis.

        The clustering and network are not stored; pass the same ones to :meth:`load`.
        """
        state = {"cluster_ids": np.array(sorted(self.roms), dtype=int),
                 "summary_json": np.array(json.dumps(self.summary))}
        for c, rom in self.roms.items():
            b = rom.basis
            state[f"scaler_mean_{c}"] = b.scaler.mean_
            state[f"scaler_scale_{c}"] = b.scaler.scale_
            state[f"components_{c}"] = b.pca.components_
            state[f"explained_variance_ratio_{c}"] = b.pca.explained_variance_ratio_
            state[f"pca_mean_{c}"] = b.pca.mean_
            state[f"rank_{c}"] = np.array(b.rank)
        np.savez(path, **state)
        print(f"Saved to {path}.npz")

    @classmethod
    def load(cls, path, cluster, network):
        d = np.load(path, allow_pickle=True)
        roms = {}
        for c in d["cluster_ids"].astype(int):
            b = Basis()
            b.scaler = StandardScaler()
            b.scaler.mean_ = d[f"scaler_mean_{c}"]
            b.scaler.scale_ = d[f"scaler_scale_{c}"]
            b.scaler.var_ = b.scaler.scale_ ** 2
            b.pca = PCA()
            b.pca.components_ = d[f"components_{c}"]
            b.pca.explained_variance_ratio_ = d[f"explained_variance_ratio_{c}"]
            b.pca.mean_ = d[f"pca_mean_{c}"]
            b.pca.n_components_ = b.pca.components_.shape[0]
            b.rank = int(d[f"rank_{c}"])
            b._precompute()
            roms[int(c)] = ROM(b)
        ens = cls(cluster=cluster, network=network, roms=roms)
        ens.summary = json.loads(str(d["summary_json"]))
        return ens

    # ---- inference ------------------------------------------------------

    def precompute_B_structure(self):
        """Cache the B sparsity slices on every ROM. Call once after loading, before solving."""
        bi, bj, bk = self.network.get_B_structure()
        for rom in self.roms.values():
            rom.set_B_structure(bi, bj, bk)
        return self

    def mean_rank(self):
        return float(np.mean([r.basis.rank for r in self.roms.values()]))

    def summary_columns(self):
        """The build diagnostics worth carrying into the metrics CSV."""
        return {k: self.summary.get(k) for k in
                ("n_splits", "n_over_tol", "n_unsplittable", "worst_err",
                 "stop_reason")}

    def _positivity(self, c, x):
        """Project ``x`` back to nonnegativity in cluster ``c``'s basis.

        The SLSQP solve runs only when ``x`` is negative somewhere.
        """
        if not np.any(x < 0.0):
            return x
        basis = self.roms[c].basis
        z, ok = project_nonnegative(basis.encode(x), basis._x_bar, basis._R,
                                    basis._scale)
        return basis._x_bar + basis._R @ z if ok else x

    def solve_tracer(self, pt, x0, dt_hydro, atol, rtol, t_eval=None,
                     positivity=True, method="BDF"):
        """Piecewise ROM solve over one tracer's hydro history.

        At each step, route on ``(pt[k], x)``, integrate one interval, hand the end state to the next. With ``positivity=False`` that is the whole solve, and a state drifting negative is carried into the next interval, which is where the stiff solver collapses.

        With it on, positivity is enforced at two points, since feasibility is basis-specific:

        * on entry, when the routed cluster changes and at ``k = 0``. A state projected in the previous basis says nothing about ``decode_c(encode_c(x))`` in this one, which is what this step integrates from; at ``k = 0`` that is the projected initial condition, which can be negative even though ``x0`` is not. Skipped when the cluster is unchanged, where it would be a no-op.
        * on exit, on the endpoint -- the state that propagates.

        Interior samples are left as the solver produced them.

        Returns ``(t, y, cluster_path, info)``. ``info`` is ``None`` on success, otherwise the failure that ended the solve, with ``t`` and ``y`` covering only the completed steps.
        """
        pt = np.asarray(pt, dtype=np.float64)
        x = np.asarray(x0, dtype=np.float64).copy()
        n_steps = pt.shape[0] - 1

        all_t, all_y = [], []
        path = np.full(n_steps, -1, dtype=int)

        def done(info):
            if not all_t:
                return np.array([]), np.empty((x.size, 0)), path, info
            # Consecutive segments share their boundary knot.
            t_out, y_out = [all_t[0]], [all_y[0]]
            for j in range(1, len(all_t)):
                if all_t[j].size and all_t[j][0] == t_out[-1][-1]:
                    t_out.append(all_t[j][1:])
                    y_out.append(all_y[j][:, 1:])
                else:
                    t_out.append(all_t[j])
                    y_out.append(all_y[j])
            return np.concatenate(t_out), np.hstack(y_out), path, info

        for k in range(n_steps):
            c = self.cluster.predict(pt[k], x)
            path[k] = c
            if c not in self.roms:
                return done(KeyError(f"no ROM fitted for cluster {c}"))
            if positivity and (k == 0 or c != path[k - 1]):
                basis = self.roms[c].basis
                x = self._positivity(c, basis.decode(basis.encode(x)))

            t_eval_step = None
            if t_eval is not None:
                lo, hi = k * dt_hydro, (k + 1) * dt_hydro
                pts = np.unique(t_eval[(t_eval >= lo) & (t_eval <= hi)] - lo)
                t_eval_step = pts if pts.size >= 2 else None
            try:
                A, B = self.network.operators(self.network.env(pt[k]))
                t_seg, y_seg, sol = self.roms[c].solve(
                    A, B, x, (0.0, dt_hydro), atol, rtol, t_eval=t_eval_step,
                    method=method)
            except Exception as exc:
                return done(exc)
            if sol.status != 0:
                return done(sol)

            if positivity:
                y_seg = y_seg.copy()
                y_seg[:, -1] = self._positivity(c, y_seg[:, -1])
            all_t.append(t_seg + k * dt_hydro)
            all_y.append(y_seg)
            x = y_seg[:, -1]

        return done(None)

    def solve_tracers(self, tracers, x0s, dt_hydro, atol, rtol, t_evals,
                      n_workers, positivity=True, method="BDF", verbose=True):
        """Solve many tracers in parallel, one result tuple each.

        Nothing bounds a single solve: a tracer stuck inside a compiled scipy call hangs the run, so enforce wall-clock outside the process.
        """
        tracers = list(tracers)
        n = len(tracers)
        x0s = np.asarray(x0s, dtype=np.float64)
        x0_list = [x0s[i].copy() for i in range(n)]

        def failed(i, info):
            # Same shape as a real result so zip(*results) stays valid.
            return (np.array([]), np.empty((x0_list[i].size, 0)),
                    np.full(np.asarray(tracers[i]).shape[0] - 1, -1, dtype=int),
                    info)

        args = [(i, tracers[i], x0_list[i], dt_hydro, atol, rtol, t_evals[i],
                 positivity, method) for i in range(n)]
        results = [None] * n

        if n_workers <= 1:
            _solve_init(self)
            for a in args:
                i, r = _solve_one(a)
                results[i] = failed(i, r) if isinstance(r, Exception) else r
        else:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_solve_init,
                                     initargs=(self,)) as pool:
                futures = {pool.submit(_solve_one, a): a[0] for a in args}
                for done_n, fut in enumerate(as_completed(futures), 1):
                    i = futures[fut]
                    try:
                        idx, r = fut.result()
                        results[idx] = (failed(idx, r) if isinstance(r, Exception)
                                        else r)
                    except Exception as exc:   # worker died, e.g. OOM-killed
                        results[i] = failed(i, exc)
                    if verbose and done_n % 50 == 0:
                        print(f"  completed {done_n}/{n}", end="\r")
        if verbose:
            print(f"  completed {n}/{n}")
        return results


def fit_global_rom(fm, network, cfg, qoi_indices, param_names, rank,
                   n_workers=1):
    """A single basis over all the training snapshots, truncated to ``rank``.

    The baseline the abundance figure compares against: the same machinery with one cluster, so the only difference is one set of coordinates for every environment. ``rank`` above the fitted rank is clamped, not an error.
    """
    from .cluster import Cluster

    cl = Cluster(param_names)
    cl.fit(fm, qoi_indices, 1, cfg.log_cols, cfg.drop_cols,
           qoi_log=cfg.qoi_log)
    labels = np.where(fm.valid_rows, 0, -1)
    # Through the worker pool like every other fit: the k=1 block is the whole training set, and this frees it with the pool.
    roms = _fit_bases(fm, labels, 1, 1.0, cfg.scale_method, n_workers,
                      verbose=False)
    basis = roms[0].basis
    basis.set_rank(min(int(rank), basis.max_rank))
    print(f"global ROM: one basis over {int(fm.valid_rows.sum()):,} snapshots, "
          f"rank {basis.rank} of {basis.max_rank}")
    return EnsembleROM(cluster=cl, network=network,
                       roms=roms).precompute_B_structure()


def _fit_bases(fm, sample_labels, n_clusters, var_threshold, scale_method,
               n_workers, verbose):
    """Fit one basis per cluster label, in parallel across clusters.

    Each worker reads its block straight from the memmap. Full-rank PCA already uses multithreaded BLAS, so keep ``n_workers`` below the core count.
    """
    args = []
    for c in range(int(n_clusters)):
        idx = np.flatnonzero(sample_labels == c)
        if idx.size >= 2:
            args.append((c, idx, var_threshold, scale_method))
        elif verbose:
            print(f"  [cluster {c}] skipped ({idx.size} snapshot(s))")

    roms = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_fit_init,
                             initargs=(fm.path, fm.species_slice)) as pool:
        for fut in as_completed([pool.submit(_fit_one, a) for a in args]):
            c, basis, n_snap = fut.result()
            roms[c] = ROM(basis)
            if verbose:
                print(f"  [cluster {c}] fit on {n_snap:,} snapshots "
                      f"-> rank {basis.rank}")
    return roms
