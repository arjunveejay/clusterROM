"""Network, hydro knots, the training/testing partition, and the per-dataset feature matrices.

The trace-cells dataset is one ``.npy`` in ``[idx | t | params | species]`` order, tracer-major, each tracer a contiguous block of ``SUBSTEPS * n_steps + 1`` rows.

Two feature matrices are derived from it once and cached in ``out_dir``, so the raw file is needed only when ``Config.build_features`` is set.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from scipy.linalg import qr as _qr
from sklearn.preprocessing import StandardScaler

from .config import (N_LEADING, N_PARAMS, NSEC, PARAM_NAMES, SPLIT_LOG_COLS,
                     SUBSTEPS, DUST_ATTENUATION, SELF_SHIELDING)
from .parser import Network, load_abundances


class ChemistryNetwork(Network):
    """The reaction network plus the params-row -> environment seam.

    The solver hands each interval's raw parameter row to ``env``. Keeping that mapping on the network lets the same ROM code drive a system with a different parameter set.
    """

    @staticmethod
    def env(p):
        return dict(nH=float(p[0]), T=float(p[1]), Tgrain=float(p[2]),
                    Av=float(p[3]), uv_flux=float(p[4]), Tcap_2body=True)

SPECIES_COL0 = N_LEADING + N_PARAMS          # first species column


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

def _feat_paths(out_dir):
    return (os.path.join(out_dir, "feature_matrix.npy"),
            os.path.join(out_dir, "feature_matrix_test.npy"))


def _idx_path(feat_path):
    return os.path.splitext(feat_path)[0] + "_indices.npy"


def materialize_feature_matrix(src_mm, starts, sel, rpt, out_path):
    """Restack the selected tracers onto a fixed per-step grid.

    Step ``k`` becomes source rows ``[SUBSTEPS*k .. SUBSTEPS*k + SUBSTEPS]``, sharing its right knot with the next step, so a basis is fitted over whole steps rather than the first 80% of each.

    ``idx`` is rewritten to the local position; global ids go to an ``_indices.npy`` sidecar.
    """
    sel = np.asarray(sel, dtype=np.int64)
    n_steps = (int(rpt) - 1) // SUBSTEPS
    tp = SUBSTEPS + 1
    rows_out = n_steps * tp
    np.save(_idx_path(out_path), sel)

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


def open_feature_matrix(feat_path, n_tracers, n_steps, n_species):
    """Memmap a feature matrix and regenerate its row provenance.

    Everything follows from the file shape, since blocks are contiguous. A tracer whose FOM solve failed left a zero-filled block, spotted by ``nH == 0`` since real densities are strictly positive.
    """
    mm = np.load(feat_path, mmap_mode="r")
    n_rows, n_cols = mm.shape
    n_params = int(n_cols) - N_LEADING - int(n_species)
    if n_params <= 0:
        raise ValueError(f"{feat_path}: {n_cols} columns is too few for "
                         f"{n_species} species. Wrong network?")
    tp, rem = divmod(n_rows, n_tracers * int(n_steps))
    if rem or tp < 1:
        raise ValueError(
            f"{feat_path}: {n_rows} rows is not divisible by n_tracers*n_steps "
            f"= {n_tracers}*{n_steps}; it does not match this dataset.")

    rows_per_tracer = int(n_steps) * tp
    sample_step = np.tile(
        np.repeat(np.arange(int(n_steps), dtype=np.int64), tp), n_tracers)
    sample_tracer = np.repeat(np.arange(n_tracers, dtype=np.int64), rows_per_tracer)
    first_rows = np.arange(n_tracers, dtype=np.int64) * rows_per_tracer
    valid_rows = np.repeat(np.asarray(mm[first_rows, N_LEADING]) != 0.0,
                           rows_per_tracer)

    print(f"Attached {feat_path}: {mm.shape} (samples/step={tp}), "
          f"{int(valid_rows[first_rows].sum())}/{n_tracers} tracers valid")
    return SimpleNamespace(
        memmap=mm, path=feat_path, sample_step=sample_step,
        sample_tracer=sample_tracer,
        valid_rows=valid_rows, n_species=int(n_species), n_params=n_params,
        param_slice=slice(N_LEADING, N_LEADING + n_params),
        species_slice=slice(N_LEADING + n_params, N_LEADING + n_params + n_species),
    )


# ---------------------------------------------------------------------------
# Hydro knots
# ---------------------------------------------------------------------------

def hydro_params(params_file):
    """Hydro parameters at the knots, ``(n_tracers * n_times, 5)``.

    Returns ``(params, n_rows, n_tracers)``, the last two being the trace-cells array's row and tracer counts, which fix the time grid.

    One file serves every network solved along the same trace cells: these are properties of the hydrodynamics, and only the species block downstream of them differs. It is shipped rather than derived -- the knot rows sit one per page of a several-hundred-GB dataset, so gathering them is slow and needs a file that is not otherwise required.
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"{params_file} is missing. It holds the hydro parameters at the "
            f"knots, which routing, the per-interval operators and the "
            f"train/test split all read, and it is expected to be present.")
    with np.load(params_file) as z:
        params = np.asarray(z["params"], dtype=np.float64)
        n_rows, _, n_tracers = (int(v) for v in z["key"][:3])
    if not np.isfinite(params).all() or not np.count_nonzero(params):
        raise ValueError(f"{params_file} is degenerate (zeros or non-finite).")
    if params.shape[0] != n_tracers * (n_rows // n_tracers - 1) // SUBSTEPS + n_tracers:
        raise ValueError(
            f"{params_file}: {params.shape[0]} rows do not match the "
            f"{n_tracers} tracers its key records.")
    print(f"  hydro params: {params_file}")
    return params, n_rows, n_tracers


# ---------------------------------------------------------------------------
# Training / testing partition
# ---------------------------------------------------------------------------

def pivoted_qr_split(params, n_tracers, n_times, n_train, n_test, seed):
    """Split tracers 50-50 at random, then order each half by pivoted QR.

    QR on the scaled parameter trajectories picks maximally diverse tracers; running it per half leaves the training and testing datasets equally diverse and identically distributed.
    """
    log_cols = [PARAM_NAMES.index(c) for c in SPLIT_LOG_COLS]
    flat = np.asarray(params, dtype=np.float64).copy()
    flat[:, log_cols] = np.log10(flat[:, log_cols])
    scaler = StandardScaler().fit(flat)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_tracers)
    mid = n_tracers // 2
    if n_train > mid or n_test > n_tracers - mid:
        raise ValueError(f"n_train <= {mid} and n_test <= {n_tracers - mid} "
                         f"are required for a 50-50 split of {n_tracers}.")

    def order(pool):
        X = scaler.transform(
            flat.reshape(n_tracers, n_times, N_PARAMS)[pool].reshape(-1, N_PARAMS)
        ).reshape(len(pool), -1)
        _, _, pivots = _qr(X.T, pivoting=True, mode="economic")
        return pool[pivots]

    return order(shuffled[:mid])[:n_train], order(shuffled[mid:])[:n_test]


def _failed_tracers(mm, starts):
    """Global ids of tracers whose FOM solve failed.

    A failed solve writes a full block of NaN, so one row per tracer spots it.
    """
    rows = np.asarray(mm[np.asarray(starts, dtype=np.int64), SPECIES_COL0:])
    return np.where(~np.isfinite(rows).all(axis=1))[0].astype(np.int64)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def build_network(cfg):
    """Construct the reaction network specified by ``cfg``.

    Returns ``(net, qoi_indices)``, where ``qoi_indices`` holds the species index of each entry of ``cfg.qoi``. Raises ``ValueError`` if a QoI is not a species of the network.
    """
    net = ChemistryNetwork(grains=True, self_shielding=SELF_SHIELDING, dust_attenuation=DUST_ATTENUATION)
    net.load_from_disk(cfg.network)
    net.drop_passive_species()
    net.initial_abundances = load_abundances(cfg.abundances)
    missing = [q for q in cfg.qoi if q not in net.species_map]
    if missing:
        raise ValueError(f"QoI {missing} are not in {cfg.network}.")
    return net, np.array([net.species_map[q] for q in cfg.qoi], dtype=int)


def load_context(cfg):
    """One-time setup shared by every fit and evaluation in a run."""
    # --- network ---
    net, qoi_indices = build_network(cfg)
    n_species = len(net.species)

    os.makedirs(cfg.out_dir, exist_ok=True)
    train_feat, test_feat = _feat_paths(cfg.out_dir)

    # --- hydro knots ---
    # Shipped, and shared by every network solved along these trace cells; it also carries the row and tracer counts, so the time grid is known without opening the trace-cells array.
    params, n_rows, n_tracers = hydro_params(cfg.params_file)

    # --- dataset geometry ---
    # The raw file's width belongs to whichever network wrote it, so take the species width from the feature matrix and only the layout from the raw.
    if cfg.build_features:
        mm = np.load(cfg.chemical_hydro_file, mmap_mode="r")
        n_cols = mm.shape[1]
    else:
        mm = None
        n_cols = int(np.load(train_feat, mmap_mode="r").shape[1])

    if n_cols - SPECIES_COL0 != n_species:
        raise ValueError(
            f"Data has {n_cols - SPECIES_COL0} species columns but "
            f"{cfg.network} defines {n_species}. Wrong out_dir, or the feature "
            f"matrices need rebuilding (build_features=True)?")
    if n_rows % n_tracers:
        raise ValueError(f"{n_rows} rows do not divide into {n_tracers} tracers.")
    rpt = n_rows // n_tracers
    if (rpt - 1) % SUBSTEPS:
        raise ValueError(f"rows_per_tracer={rpt} is not SUBSTEPS*n_steps+1.")
    n_steps = (rpt - 1) // SUBSTEPS
    n_times = n_steps + 1
    starts = np.arange(n_tracers, dtype=np.int64) * rpt
    knot_off = np.arange(0, rpt, SUBSTEPS)

    # dt_hydro is the span of every per-step solve; the evaluation grid comes from the stored snapshot times. A mismatch puts t_eval outside the span and scores every block as +inf, so refuse to start.
    if mm is not None:
        knot_t = np.asarray(mm[starts[0] + knot_off, 1], dtype=np.float64)
    else:
        tp = SUBSTEPS + 1
        knot_t = np.asarray(np.load(train_feat, mmap_mode="r")[0:64 * tp:tp, 1],
                            dtype=np.float64)
    dt_seen = float(np.median(np.diff(knot_t)))
    if abs(dt_seen - cfg.dt_hydro) > 1e-3 * cfg.dt_hydro:
        raise ValueError(
            f"dt_hydro_yr={cfg.dt_hydro_yr:g} does not match the data's knot "
            f"spacing {dt_seen / NSEC:.9g} yr. Set dt_hydro_yr to match.")
    print(f"  {n_tracers} tracers, {n_times} knots, {n_steps} steps, "
          f"dt_hydro={cfg.dt_hydro_yr:g} yr")

    # --- training / testing partition ---
    # Reuse a prior split from its sidecars: deterministic given rng_seed, but the pivoted QR is not free to redo every run.
    tr_side, te_side = _idx_path(train_feat), _idx_path(test_feat)
    if not cfg.build_features and os.path.exists(tr_side) and os.path.exists(te_side):
        train_idx, test_idx = np.load(tr_side), np.load(te_side)
        print(f"  loaded split from {tr_side}, {te_side}")
    else:
        train_idx, test_idx = pivoted_qr_split(
            params, n_tracers, n_times, cfg.n_train, cfg.n_test, cfg.rng_seed)
        # NaN species break k-means and poison any basis fitted on the cluster. Without the raw file, open_feature_matrix already drops them.
        if mm is not None:
            bad = _failed_tracers(mm, starts)
            if bad.size:
                print(f"  dropping {bad.size} tracer(s) with failed FOM solves")
                train_idx = train_idx[~np.isin(train_idx, bad)]
                test_idx = test_idx[~np.isin(test_idx, bad)]
    print(f"train tracers: {len(train_idx)},  test tracers: {len(test_idx)}")

    if cfg.build_features:
        materialize_feature_matrix(mm, starts, train_idx, rpt, train_feat)
        materialize_feature_matrix(mm, starts, test_idx, rpt, test_feat)

    # Per-tracer parameter histories, (n, n_times, 5).
    grid = params.reshape(n_tracers, n_times, N_PARAMS)
    return SimpleNamespace(
        net=net, qoi_indices=qoi_indices,
        cluster_indices=qoi_indices, n_species=n_species,
        param_names=list(PARAM_NAMES),
        dt_hydro=cfg.dt_hydro, n_times=n_times, n_steps=n_steps,
        train_indices=train_idx, train_tracers=grid[train_idx],
        test_indices=test_idx, test_tracers=grid[test_idx],
        train_feat_path=train_feat, test_feat_path=test_feat,
    )


# ---------------------------------------------------------------------------
# FOM trajectories
# ---------------------------------------------------------------------------

def _dedup_shared_knots(t, y):
    """Drop the right knots the feature matrix repeats between steps.

    The last sample of each step is the same instant as the next step's first; left in, every time-integrated norm counts it twice.
    """
    tp = SUBSTEPS + 1
    keep = np.ones(t.size, dtype=bool)
    keep[tp - 1:-1:tp] = False
    return t[keep], y[:, keep]


def load_trajectories(feat_path, n, n_species):
    """FOM trajectories by position in the dataset: the first ``n``, or exactly the positions listed if ``n`` is a sequence.

    Returns ``(t_foms, y_foms, x_eqs)``, the last stacking the initial states, in the order requested.
    """
    mm = np.load(feat_path, mmap_mode="r")
    n_params = int(mm.shape[1]) - N_LEADING - n_species
    positions = range(int(n)) if np.ndim(n) == 0 else [int(p) for p in n]
    # Read the position column once rather than once per tracer: it is 75 MB on a 2048-tracer OSU split, so a prefix load of 1311 would otherwise scan 96 GB.
    pos_col = np.asarray(mm[:, 0])
    t_foms, y_foms = [], []
    for pos in positions:
        lo, hi = np.searchsorted(pos_col, [pos, pos + 1])
        # Copy, not view: a view keeps a mapping of the whole file alive, and one per tracer adds up fast.
        block = np.array(mm[lo:hi])
        if block.shape[0] == 0 or block[0, N_LEADING] == 0.0:
            raise ValueError(f"{feat_path}: tracer {pos} has no stored FOM "
                             f"trajectory; the block is invalid.")
        t, y = _dedup_shared_knots(
            block[:, 1],
            block[:, N_LEADING + n_params:N_LEADING + n_params + n_species].T)
        t_foms.append(t)
        y_foms.append(y)
    x_eqs = np.stack([y[:, 0].copy() for y in y_foms])
    return t_foms, y_foms, x_eqs
