"""Lorenz-96 as a forcing ensemble: the network adapter, the dataset, and the loading.

The ROM assumes ``zdot = A z + B(z, z)`` with no constant term, which Lorenz-96 does not satisfy in ``x``. Shifting to the uniform fixed point ``x_i = F`` gives

    zdot_i = F (z_{i+1} - z_{i-2}) - z_i + (z_{i+1} - z_{i-2}) z_{i-1}

so ``A(F) = -I + F S`` with ``S[i, i+1] = +1``, ``S[i, i-2] = -1``, and ``B`` independent of the forcing.

Everything downstream therefore works in ``z``, not ``x``; recover ``x = z + F``.

Generating the dataset is the CLI at the bottom:

    python -m src.lorenz96 --N 100 --out datasets/lorenz96/N100_F0-2.5
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from .config import SUBSTEPS
from .config_lorenz96 import N_LEADING, N_PARAMS, PARAM_NAMES

SPECIES_COL0 = N_LEADING + N_PARAMS   # first state column of a source row


class Lorenz96Network:
    """Lorenz-96 in the shape the ROM machinery expects of a network.

    ``N >= 4`` so the stencil offsets +1, -1 and -2 stay distinct modulo N.
    """

    def __init__(self, N):
        N = int(N)
        if N < 4:
            raise ValueError(f"Lorenz-96 needs N >= 4; got {N}")
        self.N = N
        self.species = [f"x{i}" for i in range(N)]
        self.species_map = {name: i for i, name in enumerate(self.species)}
        self._coo = None

    @staticmethod
    def env(p):
        """Params row -> environment. The only parameter is the forcing."""
        return {"F": float(p[0])}

    def _B_coo(self):
        """Canonical COO arrays of the B tensor, built once.

        ``build_operators`` pairs ``B.tocoo().data`` positionally against :meth:`get_B_structure`, so both must come from this one ordering.
        """
        if self._coo is None:
            N = self.N
            i = np.arange(N, dtype=np.int64)
            row = np.concatenate([i, i])
            col_j = np.concatenate([(i + 1) % N, (i - 2) % N])
            col_k = np.concatenate([(i - 1) % N, (i - 1) % N])
            data = np.concatenate([np.ones(N), -np.ones(N)])
            # csr canonicalizes to (row, col) lexicographic order; match it.
            order = np.lexsort((col_j * N + col_k, row))
            self._coo = (row[order], col_j[order], col_k[order], data[order])
        return self._coo

    def get_B_structure(self):
        """Fixed COO indices of B. Topological, independent of the forcing."""
        row, col_j, col_k, _ = self._B_coo()
        return row.copy(), col_j.copy(), col_k.copy()

    def operators(self, env):
        """``(A, B)`` for the shifted system at forcing ``env["F"]``."""
        N = self.N
        F = float(env["F"])
        i = np.arange(N, dtype=np.int64)
        rows = np.concatenate([i, i, i])
        cols = np.concatenate([i, (i + 1) % N, (i - 2) % N])
        vals = np.concatenate([-np.ones(N), np.full(N, F), np.full(N, -F)])
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        row, col_j, col_k, data = self._B_coo()
        B = sp.csr_matrix((data, (row, col_j * N + col_k)), shape=(N, N * N))
        return A, B

    def rhs(self, z, F):
        """The shifted right-hand side evaluated directly, for validation."""
        z = np.asarray(z, dtype=float)
        shift = np.roll(z, -1) - np.roll(z, 2)
        return F * shift - z + shift * np.roll(z, 1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _shift_to_source(gen_dir, out_path, rebuild):
    """Convert the generator dataset to the shifted source the ROM reads.

    The generator writes ``[idx | t | F | x]``; this writes ``[idx | t | F | z]`` with ``z = x - F``. Returns the geometry.
    """
    meta = np.load(os.path.join(gen_dir, "trajectories_meta.npz"))
    N = int(meta["N"])
    t_eval = np.asarray(meta["t_eval"], dtype=float)
    rpt = t_eval.size
    dt = float(t_eval[1] - t_eval[0])
    if (rpt - 1) % SUBSTEPS:
        raise ValueError(f"{rpt} rows per block is not SUBSTEPS*n_steps+1 "
                         f"(SUBSTEPS={SUBSTEPS}); regenerate the dataset.")
    n_steps = (rpt - 1) // SUBSTEPS
    geom = SimpleNamespace(N=N, n_steps=n_steps, rpt=rpt,
                           dt_hydro=dt * SUBSTEPS)

    if not rebuild and os.path.exists(out_path):
        print(f"  reuse shifted source: {out_path}")
        return geom

    src = np.load(os.path.join(gen_dir, "trajectories.npy"), mmap_mode="r")
    if src.shape[1] != 3 + N:
        raise ValueError(f"{gen_dir}: expected {3 + N} columns, "
                         f"got {src.shape[1]}")
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float64,
        shape=(src.shape[0], SPECIES_COL0 + N))
    F_col = np.asarray(src[:, 2], dtype=np.float64)
    out[:, 0] = src[:, 0]                                  # idx
    out[:, 1] = src[:, 1]                                  # t
    out[:, N_LEADING] = F_col                                 # the one parameter
    out[:, SPECIES_COL0:] = np.asarray(src[:, 3:]) - F_col[:, None]
    out.flush()
    print(f"  wrote shifted source: {out_path}  {out.shape}")
    return geom


def _materialize(src_mm, starts, sel, rpt, out_path):
    """Restack the selected forcings onto the fixed per-step grid.

    Step ``k`` becomes source rows ``[SUBSTEPS*k .. SUBSTEPS*k + SUBSTEPS]``, sharing its right knot with the next step, as the chemistry path does.
    """
    sel = np.asarray(sel, dtype=np.int64)
    n_steps = (int(rpt) - 1) // SUBSTEPS
    tp = SUBSTEPS + 1
    rows_out = n_steps * tp
    np.save(os.path.splitext(out_path)[0] + "_indices.npy", sel)

    seg = np.arange(n_steps) * SUBSTEPS
    row_off = (seg[:, None] + np.arange(tp)[None, :]).ravel()
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float64,
        shape=(sel.size * rows_out, int(src_mm.shape[1])))
    for pos, g in enumerate(sel):
        lo = pos * rows_out
        out[lo:lo + rows_out] = src_mm[int(starts[g]) + row_off]
        out[lo:lo + rows_out, 0] = pos
    out.flush()
    print(f"  wrote {out_path}  {out.shape}")


def load_context(cfg):
    """One-time setup: shifted source, training/testing partition, feature matrices.

    Returns the namespace shape the chemistry path produces, so everything downstream is shared.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    source = os.path.join(cfg.out_dir, "lorenz96_source.npy")
    geom = _shift_to_source(cfg.gen_dir, source, cfg.build_features)

    net = Lorenz96Network(geom.N)
    qoi_indices = np.unique(np.asarray(cfg.qoi, dtype=int))
    if qoi_indices.min() < 0 or qoi_indices.max() >= geom.N:
        raise ValueError(f"qoi {cfg.qoi} out of range for N={geom.N}")

    mm = np.load(source, mmap_mode="r")
    idx_col = np.asarray(mm[:, 0])
    starts = np.concatenate([[0], np.flatnonzero(np.diff(idx_col)) + 1])
    n_tracers = starts.size
    if mm.shape[0] != n_tracers * geom.rpt:
        raise ValueError("forcing blocks have differing row counts")
    n_times = geom.n_steps + 1
    print(f"  {n_tracers} forcings, {n_times} knots, {geom.n_steps} steps, "
          f"dt_hydro={geom.dt_hydro:g}")

    # Forcing at each knot, one row per (forcing, knot).
    knot_rows = (starts[:, None] + np.arange(0, geom.rpt, SUBSTEPS)[None, :]).ravel()
    params = np.asarray(mm[knot_rows][:, N_LEADING:SPECIES_COL0], dtype=np.float64)

    train_feat = os.path.join(cfg.out_dir, "feature_matrix.npy")
    test_feat = os.path.join(cfg.out_dir, "feature_matrix_test.npy")
    train_side = os.path.splitext(train_feat)[0] + "_indices.npy"
    test_side = os.path.splitext(test_feat)[0] + "_indices.npy"

    if not cfg.build_features and os.path.exists(train_side):
        train_idx, test_idx = np.load(train_side), np.load(test_side)
        print(f"  loaded split from {train_side}, {test_side}")
    else:
        # Random, not pivoted QR: with a single parameter column there is nothing for the QR to pivot on.
        if cfg.n_train + cfg.n_test > n_tracers:
            raise ValueError(f"n_train + n_test > {n_tracers} forcings")
        order = np.random.default_rng(cfg.rng_seed).permutation(n_tracers)
        # Not sorted: this order reaches k-means++ through the feature matrix.
        train_idx = order[:cfg.n_train]
        test_idx = order[cfg.n_train:cfg.n_train + cfg.n_test]
    print(f"train forcings: {len(train_idx)},  test forcings: {len(test_idx)}")

    if cfg.build_features:
        _materialize(mm, starts, train_idx, geom.rpt, train_feat)
        _materialize(mm, starts, test_idx, geom.rpt, test_feat)

    grid = params.reshape(n_tracers, n_times, N_PARAMS)
    return SimpleNamespace(
        net=net, qoi_indices=qoi_indices,
        # Clustering uses the whole state; the error is measured on qoi_indices.
        cluster_indices=np.arange(geom.N), n_species=geom.N,
        param_names=list(PARAM_NAMES),
        dt_hydro=geom.dt_hydro, n_times=n_times, n_steps=geom.n_steps,
        train_indices=train_idx, train_tracers=grid[train_idx],
        test_indices=test_idx, test_tracers=grid[test_idx],
        train_feat_path=train_feat, test_feat_path=test_feat,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def lorenz96_rhs(_t, x, F):
    """``xdot_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F``.

    The unshifted field, carrying the constant the ROM cannot; contrast :meth:`Lorenz96Network.rhs`, which is the same system in ``z``. Evaluated directly rather than through the operator form, which adds per-call overhead the full-order solve gets nothing for.
    """
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def default_initial_condition(N, amplitude=1.0, n_excited=1):
    """Sparse IC shared by every forcing, so the ensemble of initial states is rank one.

    The excited components break the spatial symmetry that would otherwise pin the trajectory to the uniform state.
    """
    x0 = np.zeros(N, dtype=float)
    x0[:n_excited] = amplitude
    return x0


def generate(N, F_values, x0, t_final, dt, method="RK45", rtol=1e-8,
             atol=1e-10, verbose=True):
    """Integrate one trajectory per forcing, all from the same ``x0``.

    Writes ``x``, not the shifted ``z``; :func:`_shift_to_source` does that.

    The whole interval is recorded: the ROM has to reproduce the evolution from 0 to ``t_final``, and a per-forcing burn-in would replace the shared ``x0`` with a different starting state in every block.

    Returns ``(X, t_eval, F_values)`` with ``X`` of shape ``(len(F_values) * len(t_eval), SPECIES_COL0 + N)``.
    """
    F_values = np.asarray(F_values, dtype=float)
    if np.any(F_values == 0.0):
        raise ValueError(
            "F = 0 is not allowed: the params column doubles as the validity "
            "flag in open_feature_matrix, which reads 0 as a failed solve.")

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (N,):
        raise ValueError(f"x0 must have shape ({N},); got {x0.shape}")

    t_eval = np.arange(0.0, t_final + 0.5 * dt, dt)
    # Checked here rather than at load: _shift_to_source restacks the grid into whole hydro steps and can only refuse a grid that does not divide.
    if (t_eval.size - 1) % SUBSTEPS:
        raise ValueError(
            f"{t_eval.size} time points gives {t_eval.size - 1} intervals, "
            f"not a multiple of SUBSTEPS={SUBSTEPS}; load_context would refuse "
            f"this dataset. Adjust --t-final or --dt.")

    n_times, n_F = t_eval.size, F_values.size
    X = np.zeros((n_F * n_times, SPECIES_COL0 + N), dtype=np.float64)

    for k, F in enumerate(F_values):
        F = float(F)
        sol = solve_ivp(lorenz96_rhs, (0.0, float(t_eval[-1])), x0,
                        t_eval=t_eval, args=(F,), method=method, rtol=rtol,
                        atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve failed at F={F}: {sol.message}")

        rows = slice(k * n_times, (k + 1) * n_times)
        X[rows, 0] = k                 # idx: position in the ensemble
        X[rows, 1] = t_eval
        X[rows, N_LEADING] = F         # the one parameter
        X[rows, SPECIES_COL0:] = sol.y.T
        if verbose:
            print(f"  [{k + 1}/{n_F}] F={F:.4g}  nfev={sol.nfev}  "
                  f"|x|_max={np.abs(sol.y).max():.3f}")

    return X, t_eval, F_values


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate the Lorenz-96 forcing-ensemble dataset.")
    p.add_argument("--N", type=int, default=100, help="state dimension")
    p.add_argument("--F-min", type=float, default=0.0125)
    p.add_argument("--F-max", type=float, default=2.5)
    p.add_argument("--n-F", type=int, default=200, help="number of forcings")
    p.add_argument("--F-values", type=float, nargs="+", default=None,
                   help="explicit forcing list, overriding --F-min/--F-max/--n-F")
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.0078125,
                   help="snapshot spacing; t_final/dt must be a multiple of SUBSTEPS")
    p.add_argument("--amplitude", type=float, default=1.0,
                   help="value of the excited components in the shared sparse IC")
    p.add_argument("--n-excited", type=int, default=1,
                   help="number of nonzero components in the shared sparse IC")
    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--out", required=True,
                   help="output directory for trajectories.npy and trajectories_meta.npz")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)

    F_values = (np.asarray(a.F_values, dtype=float) if a.F_values is not None
                else np.linspace(a.F_min, a.F_max, a.n_F))
    x0 = default_initial_condition(a.N, a.amplitude, a.n_excited)

    print(f"Lorenz-96: N={a.N}, {F_values.size} forcings in "
          f"[{F_values.min():.4g}, {F_values.max():.4g}]")
    X, t_eval, F_values = generate(
        N=a.N, F_values=F_values, x0=x0, t_final=a.t_final, dt=a.dt,
        method=a.method, rtol=a.rtol, atol=a.atol)

    os.makedirs(a.out, exist_ok=True)
    traj_path = os.path.join(a.out, "trajectories.npy")
    meta_path = os.path.join(a.out, "trajectories_meta.npz")
    np.save(traj_path, X)
    np.save(os.path.join(a.out, "trajectories_indices.npy"),
            np.arange(F_values.size, dtype=np.int64))
    np.savez(meta_path, t_eval=t_eval, F_values=F_values, x0=x0, N=a.N)

    print(f"\ntrajectories {X.shape} -> {traj_path}")
    print(f"meta -> {meta_path}")
    print(f"{(t_eval.size - 1) // SUBSTEPS} hydro steps of {SUBSTEPS} "
          f"substeps, dt_hydro={a.dt * SUBSTEPS:g}")
    print(f"\nBuild the per-dataset feature matrices with:\n"
          f"  python -m scripts.run_lorenz96 --gen-dir {a.out} "
          f"--build-features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
