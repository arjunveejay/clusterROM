"""Hybrid ROM/FOM tracer solve with rewind.

Run the clustered ROM interval by interval, tracking the projection residual of the state entering each new cluster. A large residual means the incoming state is not represented by the basis about to integrate it, so rewind ``lookback`` intervals, redo that window as one full-order solve, and resume the ROM from the window's end.

    res = solve_hybrid(ens, cfg, ctx.test_tracers[i], x_eqs[i], ctx.dt_hydro,
                       t_eval=t_foms[i], hybrid=HybridConfig(thresh=1e-4))

Times are in the solver's own units throughout, matching ``ctx.dt_hydro``.

The only correction applied is the positivity projection; ``conservation_matrix`` is here for the invariant-defect diagnostic and is not imposed as a constraint.
"""
from __future__ import annotations

import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import null_space

from .evaluate import STATS, _stats
from .lorenz96 import Lorenz96Network
from .rom import apply_B_xy

#: Cluster id for an interval integrated by the FOM, which routes to no cluster.
FOM = -1

_FLOOR = 1e-30


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridConfig:
    """Knobs of the hybrid solve. Validated on construction.

    ``thresh`` is the rewind trigger, on the relative projection residual in the entered cluster's basis. ``inf`` disables the rewind, reducing the solve to the pure ROM; with ``positivity`` that gives three comparable configurations:

        thresh=inf + positivity=False   -- plain ROM
        thresh<inf + positivity=False   -- isolates the rewind
        thresh=inf + positivity=True    -- positivity only

    ``force_fom`` integrates every interval at full order, the reference path for validating the window solver.

    ``fom_atol``, ``fom_rtol`` and ``fom_min_scale`` govern the full-order windows and default to the values with which KIDApy generated the chemistry datasets. ``fom_atol`` applies to the scaled variables ``x / max(x_entry, fom_min_scale)``, not to the abundances.
    """

    lookback: int = 10
    n_fom: int = 11
    thresh: float = 1e-4
    positivity: bool = True
    force_fom: bool = False
    fom_atol: float = 1e-6
    fom_rtol: float = 1e-3
    fom_min_scale: float = 1e-22

    def __post_init__(self):
        if self.lookback < 0:
            raise ValueError(f"lookback must be >= 0; got {self.lookback!r}")
        if self.n_fom < 1:
            raise ValueError(f"n_fom must be >= 1; got {self.n_fom!r}")
        # A window ending before its own trigger resumes at k1+1 <= k and makes no forward progress.
        if self.n_fom <= self.lookback:
            raise ValueError(
                f"n_fom ({self.n_fom}) must exceed lookback ({self.lookback}) "
                "or the FOM window ends before the interval that triggered it")
        if not self.thresh > 0.0:
            raise ValueError(
                f"thresh must be > 0 (use math.inf to disable the rewind); "
                f"got {self.thresh!r}")
        for name in ("fom_atol", "fom_rtol", "fom_min_scale"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0; got {getattr(self, name)!r}")

    @property
    def rewind_enabled(self):
        return math.isfinite(self.thresh)

    def summary(self):
        return (f"HybridConfig(lookback={self.lookback}, n_fom={self.n_fom}, "
                f"thresh={self.thresh:g}, positivity={self.positivity}, "
                f"force_fom={self.force_fom}, fom_atol={self.fom_atol:g}, "
                f"fom_rtol={self.fom_rtol:g}, "
                f"fom_min_scale={self.fom_min_scale:g})")


# ---------------------------------------------------------------------------
# Conservation reference
# ---------------------------------------------------------------------------

def stoichiometry(network):
    """Dense net stoichiometric matrix ``S``, shape (n_species, n_reactions).

    ``S[i, r]`` counts species ``i`` as a product minus as a reactant of reaction ``r``. Mass-action kinetics gives ``dx/dt = S @ rate(x)``, so the rate constants are not needed to derive the invariants.
    """
    species_map = network.species_map
    reactions = getattr(network, "reactions", [])
    S = np.zeros((len(network.species), len(reactions)), dtype=np.float64)
    dropped = set()
    for r, rxn in enumerate(reactions):
        for s in rxn["reactants"]:
            if s in species_map:
                S[species_map[s], r] -= 1
            else:
                dropped.add(s)
        for s in rxn["products"]:
            if s in species_map:
                S[species_map[s], r] += 1
            else:
                dropped.add(s)
    if dropped:
        warnings.warn(
            f"stoichiometry: {sorted(dropped)} appear in reactions but not in "
            "species_map (dropped as passive?); conservation over the "
            "remaining species is not exact if they carry atoms or charge out "
            "of the tracked subsystem.", RuntimeWarning, stacklevel=2)
    return S


def conservation_matrix(network):
    """Rows spanning the left null space of ``S``, so ``C @ x`` is conserved.

    A network with no reaction list (Lorenz-96) has no stoichiometric invariant; the null space is then the whole species space, returned as the identity rather than through an SVD of a zero-column matrix.
    """
    S = stoichiometry(network)
    if S.shape[1] == 0:
        return np.eye(S.shape[0], dtype=np.float64)
    return null_space(S.T).T


@dataclass(frozen=True)
class ConservationCheck:
    """The network's linear invariants against a fixed target.

    ``c0`` is pinned to the tracer's initial condition, which mass-action kinetics conserves exactly. ``keep`` drops invariants negligible against the largest, whose relative defect is a ratio of two near-zero numbers.
    """

    W: np.ndarray
    c0: np.ndarray
    keep: np.ndarray
    rel_floor: float = 1e-8

    @classmethod
    def from_ensemble(cls, ens, x0, rel_floor=1e-8):
        W = conservation_matrix(ens.network)
        c0 = W @ np.asarray(x0, dtype=np.float64)
        a = np.abs(c0)
        return cls(W=W, c0=c0, keep=a > rel_floor * a.max(), rel_floor=rel_floor)

    def defect(self, y):
        """Largest relative invariant defect over the kept invariants and all columns."""
        y = np.asarray(y, dtype=np.float64)
        if y.size == 0 or not self.keep.any():
            return float("nan")
        err = np.abs((self.W @ y) - self.c0[:, None])[self.keep]
        return float(np.max(err / np.abs(self.c0[self.keep])[:, None]))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """One hybrid trajectory plus its per-interval bookkeeping.

    ``t``/``y`` are concatenated with shared block boundaries de-duplicated, so the columns stay registered against the reference trajectory's.

    ``path``, ``fom_step``, ``resid``, ``pos_entry`` and ``pos_endpoint`` are indexed by hydro interval; ``x_entry`` by interval boundary, so it is one longer. ``resid`` is ``nan`` where none was measured -- no cluster change, or entry from a FOM window.

    ``trigger_step`` and ``trigger_resid`` are separate because a rewind replaces its own trigger interval with a FOM one, so the residuals that fired are exactly the ones missing from ``resid``.
    """

    t: np.ndarray
    y: np.ndarray
    path: np.ndarray
    fom_step: np.ndarray
    resid: np.ndarray
    x_entry: np.ndarray
    pos_entry: np.ndarray
    pos_endpoint: np.ndarray
    n_rewind: int
    trigger_step: np.ndarray
    trigger_resid: np.ndarray
    n_steps: int
    dt_hydro: float
    config: HybridConfig
    failure: Optional[str] = None
    # An interval count is not a cost: per-interval expense differs between the two solvers, and n_rom_attempted exceeds n_rom_solves by the reduced solves a rewind discarded.
    n_fom_solves: int = 0
    n_rom_solves: int = 0
    n_rom_attempted: int = 0
    fom_seconds: float = 0.0
    rom_seconds: float = 0.0

    @property
    def solve_seconds(self):
        return self.fom_seconds + self.rom_seconds

    @property
    def fom_time_fraction(self):
        """Share of integration time spent at full order."""
        total = self.solve_seconds
        return float(self.fom_seconds / total) if total > 0 else 0.0

    @property
    def n_rom_discarded(self):
        """Reduced solves completed and then discarded by a rewind."""
        return int(self.n_rom_attempted - self.n_rom_solves)

    @property
    def complete(self):
        return self.failure is None and self.n_intervals == self.n_steps

    @property
    def n_intervals(self):
        return int(self.path.size)

    @property
    def n_fom_intervals(self):
        return int(self.fom_step.sum())

    @property
    def fom_fraction(self):
        return float(self.fom_step.mean()) if self.fom_step.size else 0.0

    @property
    def n_switches(self):
        return count_switches(self.path)

    def fom_windows(self):
        """Contiguous runs of FOM intervals as inclusive ``(k0, k1)`` pairs."""
        w = np.flatnonzero(self.fom_step)
        if not w.size:
            return []
        edges = np.r_[0, np.flatnonzero(np.diff(w) != 1) + 1, w.size]
        return [(int(w[a]), int(w[b - 1])) for a, b in zip(edges[:-1], edges[1:])]

    def summary(self, time_scale=1.0, time_label="s"):
        """Report of the rewinds, positivity, cost and FOM windows.

        ``time_scale`` divides the window times; pass ``NSEC * 1e3`` with ``time_label="kyr"`` for the chemistry figures' units.
        """
        c, dt = self.config, self.dt_hydro
        lines = [c.summary(),
                 f"  rewinds {self.n_rewind}, FOM intervals "
                 f"{self.n_fom_intervals}/{self.n_steps} ({self.fom_fraction:.1%})",
                 f"  positivity: entry {int(self.pos_entry.sum())}, endpoint "
                 f"{int(self.pos_endpoint.sum())}",
                 f"  cluster switches {self.n_switches}",
                 f"  cost: {self.n_fom_solves} FOM solve(s) {self.fom_seconds:.2f}s, "
                 f"{self.n_rom_solves} ROM solve(s) {self.rom_seconds:.2f}s "
                 f"({self.n_rom_discarded} discarded) -- "
                 f"{self.fom_time_fraction:.1%} of time at full order"]
        if self.trigger_resid.size:
            lines.append("  triggers: " + ", ".join(
                f"interval {int(k)} at {v:.3g}"
                for k, v in zip(self.trigger_step, self.trigger_resid)))
        for k0, k1 in self.fom_windows():
            lines.append(f"  FOM intervals {k0}-{k1} = "
                         f"{k0 * dt / time_scale:.3g}-"
                         f"{(k1 + 1) * dt / time_scale:.3g} {time_label}")
        if self.failure is not None:
            lines.append(f"  INCOMPLETE after {self.n_intervals}/{self.n_steps} "
                         f"intervals: {self.failure}")
        return "\n".join(lines)


@dataclass
class RomResult:
    """A pure-ROM baseline trajectory, shaped to compare against :class:`HybridResult`."""

    t: np.ndarray
    y: np.ndarray
    path: np.ndarray
    info: object = None

    @property
    def n_switches(self):
        return count_switches(self.path)


@dataclass(frozen=True)
class TrajectoryScore:
    """Error of one trajectory against a reference, on the QoI species."""

    l2_per_qoi: np.ndarray
    l2: float
    defect: float
    coverage: float
    species: tuple = ()

    @property
    def complete(self):
        """True when the trajectory reaches the reference's final column."""
        return self.coverage > 0.999


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------

def count_switches(path):
    """Number of cluster occupancies along ``path``, ignoring :data:`FOM` intervals."""
    p = np.asarray(path)
    p = p[p >= 0]
    return int((np.diff(p) != 0).sum() + 1) if p.size else 0


def to_interval_path(path, t, dt_hydro, n_steps):
    """Per-interval cluster path, reducing a per-snapshot one if needed.

    ``solve_tracer``'s is already per interval. A per-snapshot path is reduced by sampling strictly inside each interval, since at the knot itself ``int(t/dt)`` names the interval that just ended. Intervals with no interior sample get :data:`FOM`.
    """
    path, t = np.asarray(path), np.asarray(t, dtype=np.float64)
    if len(path) == n_steps:
        return path.astype(int, copy=False)
    out = np.full(n_steps, FOM, dtype=int)
    for k in range(n_steps):
        inside = np.flatnonzero((t > k * dt_hydro) & (t < (k + 1) * dt_hydro))
        if inside.size:
            out[k] = int(path[min(inside[inside.size // 2], len(path) - 1)])
    return out


def _knots(n_steps, dt_hydro):
    return np.arange(n_steps + 1, dtype=np.float64) * dt_hydro


def _window_eval(t_eval, lo, hi):
    """Reference time points inside ``[lo, hi]``, rebased to 0, or None if under two."""
    if t_eval is None:
        return None
    p = np.unique(t_eval[(t_eval >= lo) & (t_eval <= hi)] - lo)
    return p if p.size >= 2 else None


@dataclass
class IntervalStep:
    """One stretch of hydro intervals, as returned by either solver.

    ``cluster`` is :data:`FOM` for a full-order window, whose multi-interval ``t``/``y`` are split by :meth:`_Intervals.extend_window`.
    """

    t: np.ndarray
    y: np.ndarray
    cluster: int
    x_end: np.ndarray
    pos_entry: bool = False
    pos_endpoint: bool = False
    failure: Optional[str] = None


# ---------------------------------------------------------------------------
# Interval solvers
# ---------------------------------------------------------------------------

class RomIntervalSolver:
    """One hydro interval of the clustered ROM. Built once per tracer.

    Must agree with ``EnsembleROM.solve_tracer``, which cannot be reused here because it is a closed loop over every interval while the hybrid has to stop and rewind. Keep :meth:`step` in the order solve_tracer uses: route, project on entry, integrate, project the endpoint.
    """

    def __init__(self, ens, cfg, dt_hydro, t_eval=None, positivity=True):
        self.ens = ens
        self.cfg = cfg
        self.dt = float(dt_hydro)
        self.positivity = bool(positivity)
        self.t_eval = None if t_eval is None else np.asarray(t_eval, dtype=np.float64)
        self.solve_seconds = 0.0
        self.n_solves = 0

    def route(self, pt_k, x):
        return self.ens.cluster.predict(pt_k, x)

    def entry_residual(self, cid, x):
        """Relative residual of ``x`` against cluster ``cid``'s basis, scaled.

        Scaled units, or the residual is dominated by the abundant species every basis reconstructs perfectly.
        """
        b = self.ens.roms[cid].basis
        v = np.asarray(x, dtype=np.float64) - b._x_bar
        e = v - b._R @ (b._L @ v)
        return float(np.linalg.norm(e / b._scale)
                     / (np.linalg.norm(v / b._scale) + 1e-300))

    def _positivity_entry(self, cid, x):
        """``solve_tracer``'s entry positivity step, on the state the interval integrates from.

        ``decode(encode(x))`` can be negative where ``x`` is not, and it is what the interval starts from.
        """
        if not self.positivity:
            return x, False
        b = self.ens.roms[cid].basis
        x_in = b.decode(b.encode(x))
        fired = bool(np.any(x_in < 0.0))
        return self.ens._positivity(cid, x_in), fired

    def _positivity_endpoint(self, cid, x):
        if not self.positivity:
            return x, False
        fired = bool(np.any(np.asarray(x) < 0.0))
        return self.ens._positivity(cid, x), fired

    def step(self, pt_k, k, x, cid, entering):
        """Integrate interval ``k`` in cluster ``cid``.

        ``entering`` gates the entry positivity step: on an unchanged cluster, encode/decode is idempotent and it is a guaranteed no-op.
        """
        applied_entry = False
        if entering:
            x, applied_entry = self._positivity_entry(cid, x)

        net = self.ens.network
        A, B = net.operators(net.env(pt_k))
        t0 = time.perf_counter()
        tt, yy, sol = self.ens.roms[cid].solve(
            A, B, x, (0.0, self.dt), self.cfg.atol, self.cfg.rtol,
            t_eval=_window_eval(self.t_eval, k * self.dt, (k + 1) * self.dt),
            method=self.cfg.ode_method)
        self.solve_seconds += time.perf_counter() - t0
        if sol.status != 0:
            return IntervalStep(
                np.array([]), np.empty((np.size(x), 0)), cid, np.asarray(x),
                failure=f"interval {k}: ROM solve failed in cluster {cid} -- "
                        f"{sol.message}")
        self.n_solves += 1

        yy = np.asarray(yy)
        x_end, applied_end = self._positivity_endpoint(cid, yy[:, -1])
        if applied_end:
            # solve_tracer reports the projected endpoint at the knot, not only carries it forward. Interior samples stay as solved.
            yy = yy.copy()
            yy[:, -1] = x_end
        return IntervalStep(np.asarray(tt) + k * self.dt, yy, cid, x_end,
                            pos_entry=applied_entry, pos_endpoint=applied_end)


def _scale_operators(A, B, s):
    """``(A, B)`` in the coordinates ``z = x / s``, as KIDApy's ``QuadraticSolver.solve`` forms them."""
    N = s.size
    s_inv = 1.0 / s
    A_sc = sp.diags(s_inv, format="csr") @ A.tocsr() @ sp.diags(s, format="csr")
    if B.nnz == 0:
        return A_sc, B
    Bcoo = B.tocoo(copy=False)
    bi = Bcoo.row.astype(np.int64, copy=False)
    bj = (Bcoo.col // N).astype(np.int64, copy=False)
    bk = (Bcoo.col % N).astype(np.int64, copy=False)
    B_sc = sp.coo_matrix(
        (Bcoo.data * s[bj] * s[bk] * s_inv[bi], (Bcoo.row, Bcoo.col)),
        shape=B.shape, dtype=np.float64).tocsr()
    return A_sc, B_sc


class FomWindowSolver:
    """A run of hydro intervals integrated at full order. Built once per tracer.

    The only full-order solve in the package: the dataset's trajectories are read, not integrated. Each interval is solved as KIDApy's piecewise-constant tracer solve generated the chemistry datasets: in the coordinates ``z = x / max(x_entry, fom_min_scale)``, rescaled at every knot. No Jacobian is passed: on networks of a few dozen species, BDF's finite-difference Jacobian with dense LU is faster than the analytic sparse one with sparse LU.

    Raises ``NotImplementedError`` for Lorenz-96, whose signed state admits no such scaling.
    """

    def __init__(self, ens, cfg, pt, dt_hydro, t_eval=None, hybrid=None):
        if isinstance(ens.network, Lorenz96Network):
            raise NotImplementedError(
                "FomWindowSolver solves in coordinates scaled by the positive "
                "abundances and is not defined for the signed Lorenz-96 state")
        self.ens = ens
        self.cfg = cfg
        self.hybrid = hybrid or HybridConfig()
        self.pt = np.asarray(pt, dtype=np.float64)
        self.dt = float(dt_hydro)
        self.t_eval = None if t_eval is None else np.asarray(t_eval, dtype=np.float64)
        self.solve_seconds = 0.0
        self.n_solves = 0

    def step_window(self, k0, k1, x):
        """Full-order solve across intervals ``[k0, k1]`` inclusive, one scaled solve per interval.

        Operators are constant within an interval and jump at each knot, so each interval is a separate solve, and its scale is taken from the state entering it.
        """
        net = self.ens.network
        h = self.hybrid
        x = np.asarray(x, dtype=np.float64)
        N = x.size
        ts, ys = [], []
        t0 = time.perf_counter()
        for k in range(k0, k1 + 1):
            A, B = net.operators(net.env(self.pt[k]))
            s = np.maximum(x, h.fom_min_scale)
            A_sc, B_sc = _scale_operators(A, B, s)
            te = _window_eval(self.t_eval, k * self.dt, (k + 1) * self.dt)
            if te is not None:
                # A stored knot can land a few ULPs outside the span, which solve_ivp rejects outright.
                te = np.clip(te, 0.0, self.dt)
            sol = solve_ivp(
                lambda _t, z: np.asarray(A_sc @ z).ravel() + apply_B_xy(z, z, B_sc),
                (0.0, self.dt), x / s, method=self.cfg.ode_method,
                atol=h.fom_atol, rtol=h.fom_rtol, t_eval=te)
            if sol.status != 0:
                self.solve_seconds += time.perf_counter() - t0
                return IntervalStep(
                    np.array([]), np.empty((N, 0)), FOM, x,
                    failure=f"window {k0}-{k1}: FOM solve failed in interval "
                            f"{k} -- {sol.message}")
            tt, yy = sol.t + k * self.dt, s[:, None] * sol.y
            # Consecutive intervals share their knot; keep it once so extend_window splits the window cleanly.
            if ts and tt.size and np.isclose(tt[0], ts[-1][-1]):
                tt, yy = tt[1:], yy[:, 1:]
            ts.append(tt)
            ys.append(yy)
            x = s * sol.y[:, -1]
        self.solve_seconds += time.perf_counter() - t0
        self.n_solves += 1
        return IntervalStep(np.concatenate(ts), np.hstack(ys), FOM, x)


class _Intervals:
    """Per-interval accumulator for a hybrid trajectory.

    Every list is indexed by hydro interval, ``t``/``y`` included -- one block per interval even when a FOM solve spans many, or :meth:`truncate` would cut in the wrong place. ``x_entry[k]`` is the state entering interval ``k``, so it holds one more element than the rest.
    """

    def __init__(self, x0):
        self.t = []
        self.y = []
        self.path = []
        self.fom = []
        self.resid = []
        self.pos_entry = []
        self.pos_endpoint = []
        self.x_entry = [np.asarray(x0, dtype=np.float64).copy()]

    def __len__(self):
        return len(self.path)

    def append(self, step, resid=np.nan):
        self.t.append(np.asarray(step.t, dtype=np.float64))
        self.y.append(np.asarray(step.y, dtype=np.float64))
        self.path.append(int(step.cluster))
        self.fom.append(step.cluster == FOM)
        self.resid.append(float(resid))
        self.pos_entry.append(bool(step.pos_entry))
        self.pos_endpoint.append(bool(step.pos_endpoint))
        self.x_entry.append(np.asarray(step.x_end, dtype=np.float64))

    def extend_window(self, step, k0, k1, dt):
        """Absorb a multi-interval FOM solve as one block per interval."""
        tt, yy = np.asarray(step.t), np.asarray(step.y)
        for kk in range(k0, k1 + 1):
            msk = (tt >= kk * dt) & (tt <= (kk + 1) * dt)
            j = int(np.argmin(np.abs(tt - (kk + 1) * dt)))
            self.append(IntervalStep(tt[msk], yy[:, msk], FOM, yy[:, j]))

    def truncate(self, k0):
        """Drop every interval from ``k0`` on, keeping the state entering ``k0``."""
        for lst in (self.t, self.y, self.path, self.fom, self.resid,
                    self.pos_entry, self.pos_endpoint):
            del lst[k0:]
        del self.x_entry[k0 + 1:]

    def last_cluster(self):
        """The most recent real cluster, skipping FOM intervals.

        Comparing against the :data:`FOM` sentinel instead would make every interval after a window look like a cluster switch.
        """
        return next((c for c in reversed(self.path) if c >= 0), FOM)

    def concatenate(self):
        """One array pair, with the knot shared by consecutive blocks emitted once."""
        if not self.t:
            return np.array([]), np.empty((self.x_entry[0].size, 0))
        t_out, y_out = [self.t[0]], [self.y[0]]
        for j in range(1, len(self.t)):
            if (self.t[j].size and t_out[-1].size
                    and np.isclose(self.t[j][0], t_out[-1][-1])):
                t_out.append(self.t[j][1:])
                y_out.append(self.y[j][:, 1:])
            else:
                t_out.append(self.t[j])
                y_out.append(self.y[j])
        return np.concatenate(t_out), np.hstack(y_out)


# ---------------------------------------------------------------------------
# The hybrid solve
# ---------------------------------------------------------------------------

def solve_hybrid(ens, cfg, pt, x0, dt_hydro, t_eval=None, hybrid=None):
    """Solve one tracer with the clustered ROM, rewinding to the FOM when a cluster hand-off lands off the incoming basis.

    ``pt`` has one row per knot, so ``n_steps + 1`` rows. ``t_eval`` is the whole trajectory's time array, normally the reference's own times, which is what keeps the output columns registered against it.

    A solver failure is recorded in :attr:`HybridResult.failure` and the trajectory so far is returned, following ``solve_tracer``'s ``info`` convention rather than raising out of a sweep.
    """
    hybrid = hybrid or HybridConfig()
    pt = np.asarray(pt, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    dt = float(dt_hydro)
    n_steps = pt.shape[0] - 1
    if n_steps < 1:
        raise ValueError(f"pt needs at least two rows; got shape {pt.shape}")
    t_eval = (_knots(n_steps, dt) if t_eval is None
              else np.asarray(t_eval, dtype=np.float64))

    rom = RomIntervalSolver(ens, cfg, dt, t_eval=t_eval,
                            positivity=hybrid.positivity)
    fom = FomWindowSolver(ens, cfg, pt, dt, t_eval=t_eval, hybrid=hybrid)

    iv = _Intervals(x0)
    triggered = set()
    force_until, n_rewind = -1, 0
    trigger_step, trigger_resid = [], []
    failure = None
    # Includes intervals a later rewind discards, which the final path does not show.
    n_rom_attempted = 0

    k = 0
    while k < n_steps:
        x = iv.x_entry[k]
        cid = rom.route(pt[k], x)
        if cid not in ens.roms:
            failure = f"interval {k}: no ROM fitted for cluster {cid}"
            break
        # A state leaving a FOM window lies in no basis, so its residual is a larger quantity than thresh is calibrated against.
        from_fom = k > 0 and iv.fom[k - 1]
        entering = cid != iv.last_cluster()
        r = (rom.entry_residual(cid, x)
             if (entering and not from_fom) else np.nan)

        if hybrid.force_fom:
            step = fom.step_window(k, k, x)
            if step.failure is not None:
                failure = step.failure
                break
            iv.append(step)
            k += 1
            continue

        # ---- trigger: rewind and redo the window as one FOM solve ----------
        if (entering and r > hybrid.thresh
                and k not in triggered and k > force_until):
            triggered.add(k)
            n_rewind += 1
            trigger_step.append(k)
            trigger_resid.append(r)
            k0 = max(0, k - hybrid.lookback)
            k1 = min(k0 + hybrid.n_fom - 1, n_steps - 1)
            iv.truncate(k0)

            step = fom.step_window(k0, k1, iv.x_entry[k0])
            if step.failure is not None:
                failure = step.failure
                break
            iv.extend_window(step, k0, k1, dt)
            force_until = k1
            k = k1 + 1
            continue

        # ---- ordinary ROM interval -----------------------------------------
        n_rom_attempted += 1
        step = rom.step(pt[k], k, x, cid, entering)
        if step.failure is not None:
            failure = step.failure
            break
        iv.append(step, resid=r)
        k += 1

    t_out, y_out = iv.concatenate()
    return HybridResult(
        t=t_out, y=y_out,
        path=np.asarray(iv.path, dtype=int),
        fom_step=np.asarray(iv.fom, dtype=bool),
        resid=np.asarray(iv.resid, dtype=float),
        x_entry=np.asarray(iv.x_entry, dtype=float),
        pos_entry=np.asarray(iv.pos_entry, dtype=bool),
        pos_endpoint=np.asarray(iv.pos_endpoint, dtype=bool),
        n_rewind=n_rewind,
        trigger_step=np.asarray(trigger_step, dtype=int),
        trigger_resid=np.asarray(trigger_resid, dtype=float),
        n_steps=n_steps, dt_hydro=dt, config=hybrid, failure=failure,
        n_fom_solves=fom.n_solves, n_rom_solves=rom.n_solves,
        n_rom_attempted=n_rom_attempted,
        fom_seconds=fom.solve_seconds, rom_seconds=rom.solve_seconds)


def solve_pure_rom(ens, cfg, pt, x0, dt_hydro, t_eval=None, positivity=True):
    """Pure-ROM baseline for the same tracer, as a :class:`RomResult`.

    Wraps :meth:`EnsembleROM.solve_tracer` so hybrid and baseline score through the same code.
    """
    pt = np.asarray(pt, dtype=np.float64)
    n_steps = pt.shape[0] - 1
    t, y, path, info = ens.solve_tracer(
        pt, x0, dt_hydro, cfg.atol, cfg.rtol, t_eval=t_eval,
        positivity=positivity, method=cfg.ode_method)
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    return RomResult(t=t, y=y,
                     path=to_interval_path(path, t, dt_hydro, n_steps),
                     info=info)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_trajectory(y, y_ref, qoi_indices, species=None, conservation=None):
    """Relative L2 error per QoI against ``y_ref``, plus the invariant defect.

    A trajectory that stopped early is scored over the columns it has, with the fraction reported as ``coverage``. Otherwise an incomplete solve scores better than a complete one by never reaching what it would have got wrong.
    """
    y, y_ref = np.asarray(y, dtype=float), np.asarray(y_ref, dtype=float)
    qi = list(qoi_indices)
    m = min(y.shape[1], y_ref.shape[1]) if y.ndim == 2 and y.size else 0
    names = tuple(str(j) if species is None else str(species[j]) for j in qi)
    if m == 0:
        return TrajectoryScore(np.full(len(qi), np.nan), float("nan"),
                               float("nan"), 0.0, names)
    per = (np.linalg.norm(y[qi, :m] - y_ref[qi, :m], axis=1)
           / np.linalg.norm(y_ref[qi, :m], axis=1))
    return TrajectoryScore(
        l2_per_qoi=per, l2=float(np.max(per)),
        defect=(conservation.defect(y[:, :m]) if conservation is not None
                else float("nan")),
        coverage=float(m / y_ref.shape[1]), species=names)


def _error_series(t, y, t_fom, y_fom):
    """Per-species ``(mean-in-time, max-in-time, l2-in-time)`` relative errors.

    Same reductions, floor and re-interpolation as ``evaluate.evaluate``, so hybrid numbers are comparable with the sweep CSVs.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    t_fom, y_fom = np.asarray(t_fom, dtype=float), np.asarray(y_fom, dtype=float)
    if t.shape[0] != t_fom.shape[0]:
        y = interp1d(t, y, axis=1, bounds_error=False,
                     fill_value="extrapolate")(t_fom)
    resid = y - y_fom
    err = np.abs(resid) / (np.abs(y_fom) + _FLOOR)
    num = np.sqrt(np.sum(resid ** 2, axis=1))
    den = np.sqrt(np.sum(y_fom ** 2, axis=1)) + _FLOOR
    return err.mean(axis=1), err.max(axis=1), num / den


def tracer_diagnostics(result):
    """Scalar per-tracer summary of a hybrid solve, for a per-tracer CSV row."""
    return {
        "rewinds": int(result.n_rewind),
        "n_fom_intervals": int(result.n_fom_intervals),
        "fom_frac": float(result.fom_fraction),
        "pos_entry": int(result.pos_entry.sum()),
        "pos_endpoint": int(result.pos_endpoint.sum()),
        "switches": int(result.n_switches),
        "max_trigger_resid": (float(result.trigger_resid.max())
                              if result.trigger_resid.size else float("nan")),
        "n_intervals_done": int(result.n_intervals),
        # Full-order windows solved: neither rewinds (force_fom makes none) nor n_fom_intervals (one window spans several intervals).
        "fom_solves": int(result.n_fom_solves),
        "rom_solves": int(result.n_rom_solves),
        "rom_discarded": int(result.n_rom_discarded),
        "fom_seconds": float(result.fom_seconds),
        "rom_seconds": float(result.rom_seconds),
        "fom_time_frac": float(result.fom_time_fraction),
        "failure": result.failure,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

_hybrid_state = None


def _hybrid_init(ens, cfg, hybrid):
    global _hybrid_state
    _hybrid_state = (ens, cfg, hybrid)


def _hybrid_solve_one(args):
    """Solve and score one tracer in a worker.

    Returns only the error vectors and scalars; shipping the whole trajectory back would move data the sweep discards.
    """
    i, pt, x0, t_fom, y_fom, dt_hydro = args
    ens, cfg, hybrid = _hybrid_state
    try:
        res = solve_hybrid(ens, cfg, pt, x0, dt_hydro, t_eval=t_fom,
                           hybrid=hybrid)
        n_steps = np.asarray(pt).shape[0] - 1
        valid = (res.failure is None and res.t.size > 0
                 and np.isclose(res.t[-1], n_steps * dt_hydro))
        errs = _error_series(res.t, res.y, t_fom, y_fom) if valid else None
        return i, errs, tracer_diagnostics(res)
    except Exception as exc:
        return i, None, {"failure": f"{type(exc).__name__}: {exc}"}


DIAG_KEYS = ("rewinds", "n_fom_intervals", "fom_frac", "pos_entry",
             "pos_endpoint", "switches", "max_trigger_resid",
             "n_intervals_done", "fom_solves", "rom_solves", "rom_discarded",
             "fom_seconds", "rom_seconds", "fom_time_frac")


def evaluate_hybrid(ens, tracers, x_eqs, t_foms, y_foms, qoi_indices,
                    qoi_names, dt_hydro, cfg, hybrid=None, n_workers=1,
                    verbose=True):
    """Roll the hybrid out over a tracer set, in ``evaluate.evaluate``'s dict shape.

    Adds a ``hybrid`` block of per-tracer diagnostics with no pure-ROM counterpart, over every attempted tracer rather than only the valid ones -- the failures are what say whether a rewind broke it.
    """
    hybrid = hybrid or HybridConfig()
    n = len(t_foms)
    args = [(i, np.asarray(tracers[i]), np.asarray(x_eqs[i]),
             np.asarray(t_foms[i]), np.asarray(y_foms[i]), dt_hydro)
            for i in range(n)]

    errs, diags = [None] * n, [None] * n
    if n_workers <= 1:
        _hybrid_init(ens, cfg, hybrid)
        for a in args:
            i, e, d = _hybrid_solve_one(a)
            errs[i], diags[i] = e, d
    else:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_hybrid_init,
                                 initargs=(ens, cfg, hybrid)) as pool:
            futures = {pool.submit(_hybrid_solve_one, a): a[0] for a in args}
            for done, fut in enumerate(as_completed(futures), 1):
                i = futures[fut]
                try:
                    idx, e, d = fut.result()
                    errs[idx], diags[idx] = e, d
                except Exception as exc:   # one bad tracer must not end the sweep
                    diags[i] = {"failure": f"{type(exc).__name__}: {exc}"}
                if verbose and done % 50 == 0:
                    print(f"  completed {done}/{n}", end="\r")
    if verbose:
        print(f"  completed {n}/{n}")

    valid = [i for i in range(n) if errs[i] is not None]
    failed = [i for i in range(n)
              if diags[i] is None or diags[i].get("failure") is not None]
    out = {"n_tracers": n, "n_solved": n - len(failed), "n_valid": len(valid),
           "failed_idx": failed, "valid_idx": list(valid),
           "hybrid_config": hybrid, "per_tracer": {}}
    if not valid:
        raise RuntimeError(
            f"no tracer produced a complete hybrid trajectory ({n} attempted). "
            "First failures: " + "; ".join(
                str((diags[i] or {}).get("failure")) for i in failed[:3]))

    for j, tag in enumerate(("mean-in-time", "max-in-time", "l2-in-time")):
        agg = np.stack([errs[i][j] for i in valid])
        x = agg[:, list(qoi_indices)]
        st = _stats(x)
        out[tag] = {name: {s: float(st[s][k]) for s in STATS}
                    for k, name in enumerate(qoi_names)}
        out["per_tracer"][tag] = {name: x[:, k].copy()
                                  for k, name in enumerate(qoi_names)}

    out["hybrid"] = {
        k: np.array([(diags[i] or {}).get(k, np.nan) for i in range(n)],
                    dtype=float) for k in DIAG_KEYS}
    return out
