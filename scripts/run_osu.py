#!/usr/bin/env python
"""OSU (osu_09_2008) chemistry run: eta x tau sweep, hybrid sweep, evaluation, figures.

    python -m scripts.run_osu --out-dir experiments/osu --chemical-hydro-file datasets/grav_collapse/osu_09_2008/piecewise_constant/R0.1_M6_trace_cells.npy

Differs from run_nelson.py only in configuration, following the OSU notebooks: 5 QoIs, only nH log-transformed, and Tgrain/Av/uv_flux dropped from the clustering features.

After the pure-ROM sweep, the hybrid ROM/FOM solver is rolled out over the same fitted ensembles (--hybrid-etas x --hybrid-taus, all of them by default) and overlaid on the clustered-error panel. --no-hybrid-sweep skips it.

The run reads the provided feature matrices from out_dir (see the README). --build-features rebuilds them instead, and needs the raw chemistry dataset, which is not distributed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# The package root, so the drivers run from anywhere: `python run_nelson.py` inside scripts/, or by full path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import default_out_dir



def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="R0.1_M6.0", help="dataset identifier; names the run directory")
    p.add_argument("--out-dir", default=None, help="run directory; defaults to experiments/<case>/<model>")
    p.add_argument("--chemical-hydro-file", default=None, help="override the .npy holding the hydro parameters and the chemistry; defaults to datasets/grav_collapse/osu_09_2008/, and is read only by --build-features")
    p.add_argument("--build-features", action="store_true", help="rebuild the split feature matrices from the raw dataset")
    p.add_argument("--etas", type=float, nargs="+", default=[1e-5, 1e-8, 1e-11, 1e-14], help="residual SVD energy values to sweep")
    p.add_argument("--taus", type=float, nargs="+", default=[0.01, 0.10, 0.40], help="split tolerances to sweep")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--max-clusters", type=int, default=1024)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=2048)
    p.add_argument("--n-eval", type=int, default=2048, help="trajectories of the evaluated dataset integrated per combination")
    p.add_argument("--qoi", nargs="+", default=["CO", "C+", "e-", "O", "C"])
    p.add_argument("--n-workers", type=int, default=None, help="process-pool size; default is the core count")
    p.add_argument("--tracer", type=int, default=3, help="position within the evaluated split the per-tracer figures use")
    p.add_argument("--global-rank", type=int, default=5, help="rank of the single-basis baseline in the figures")
    p.add_argument("--figure-combo", type=float, nargs=2, metavar=("TAU", "ETA"), default=None, help="combination the per-trajectory figures use; default is the middle of each sweep")
    p.add_argument("--ranks", type=int, nargs="+", default=None, help="encoder ranks for the global-ROM baseline; default 1..14, ranks above the basis's rho are dropped")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--eval-split", default="test", choices=["test", "train"], help="which split to roll out; fits always train on the train split")
    p.add_argument("--overwrite", action="store_true", help="redo the run in place instead of refusing on an existing CSV")

    h = p.add_argument_group("hybrid sweep")
    h.add_argument("--no-hybrid-sweep", action="store_true", help="skip the hybrid rollout over the fitted ensembles")
    h.add_argument("--hybrid-etas", type=float, nargs="+", default=None, help="subset of --etas the hybrid is rolled out at; default all of them")
    h.add_argument("--hybrid-taus", type=float, nargs="+", default=None, help="subset of --taus the hybrid is rolled out at; default all of them")
    h.add_argument("--hybrid-thresh", type=float, default=1e-4, help="rewind trigger on the relative projection residual; inf disables the rewind")
    h.add_argument("--hybrid-lookback", type=int, default=10, help="hydro intervals a rewind steps back")
    h.add_argument("--hybrid-n-fom", type=int, default=11, help="hydro intervals each FOM window spans; must exceed --hybrid-lookback")
    return p


def _subset(chosen, swept, flag):
    """``chosen`` (or all of ``swept``), refusing values the pure-ROM sweep did not fit."""
    if chosen is None:
        return list(swept)
    missing = [v for v in chosen if not any(math.isclose(v, s) for s in swept)]
    if missing:
        raise SystemExit(f"{flag} {missing} not in the swept values {list(swept)}; the hybrid reuses the sweep's fitted ensembles")
    return list(chosen)


def hybrid_sweep(ctx, cfg, etas, taus, hybrid, fom, out_dir, eval_split="test", overwrite=False):
    """Roll the hybrid out at every (eta, tau) over the ensembles ``run_sweep`` saved.

    Nothing is refitted: each combination's cluster and ensemble are loaded from ``<out_dir>/sweep``, so the hybrid and pure-ROM rows describe the same bases. ``fom`` is the ``(tracers, x_eqs, t_foms, y_foms)`` tuple ``run_sweep`` returned, so both are scored on the same trajectories.

    Writes ``hybrid_sweep<sfx>.csv`` in the sweep CSV's row layout (so ``mean_series`` reads it unchanged), with the hybrid configuration and the tracer-averaged diagnostics as extra columns, plus per-combination metrics, per-tracer errors and per-tracer diagnostics under ``sweep/``.
    """
    import numpy as np

    from src.cluster import Cluster
    from src.config import combo_stem, split_suffix
    from src.evaluate import metrics_rows, per_tracer_rows, read_table, write_csv
    from src.hybrid import DIAG_KEYS, evaluate_hybrid
    from src.pipeline import split_paths
    from src.rom import EnsembleROM

    sweep_dir = os.path.join(out_dir, "sweep")
    sfx = split_suffix(eval_split)
    csv_path = os.path.join(out_dir, f"hybrid_sweep{sfx}.csv")
    if os.path.exists(csv_path) and not overwrite:
        raise FileExistsError(f"{csv_path} exists; pass --overwrite to redo the run in place.")

    tracers, x_eqs, t_foms, y_foms = fom
    split_indices = split_paths(ctx, eval_split)[2]
    qoi_names = [ctx.net.species[j] for j in ctx.qoi_indices]
    hyb_meta = dict(thresh=float(hybrid.thresh), lookback=int(hybrid.lookback), n_fom=int(hybrid.n_fom), positivity=hybrid.positivity)

    rows_all = []
    for eta in etas:
        cfg_eta = cfg.replace(eta=eta, positivity=hybrid.positivity)
        for tau in taus:
            stem = combo_stem(tau, eta)
            cl = Cluster.load(os.path.join(sweep_dir, f"cluster_{stem}.npz"))
            ens = EnsembleROM.load(os.path.join(sweep_dir, f"ensemble_{stem}.npz"), cl, ctx.net)
            ens.precompute_B_structure()

            res = evaluate_hybrid(ens, tracers, x_eqs, t_foms, y_foms, ctx.qoi_indices, qoi_names, ctx.dt_hydro, cfg_eta, hybrid=hybrid, n_workers=cfg.n_workers, verbose=False)
            diag = res["hybrid"]
            # Averaged over every attempted tracer; a failed one carries nan and drops out.
            diag_means = {f"mean_{k}": float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in diag.items()}
            print(f"[hybrid] {stem}: {res['n_valid']}/{res['n_tracers']} complete, k={cl.n_clusters}, FOM fraction {diag_means['mean_fom_frac']:.3f}, rewinds {diag_means['mean_rewinds']:.2f}", flush=True)

            combo = metrics_rows(res, qoi_names, tau_split=float(tau), eta=float(eta), eval_split=eval_split, n_clusters=int(cl.n_clusters), mean_rank=ens.mean_rank(), **hyb_meta, **diag_means)
            write_csv(os.path.join(sweep_dir, f"hybrid_metrics_{stem}{sfx}.csv"), combo)
            write_csv(os.path.join(sweep_dir, f"hybrid_pertracer_{stem}{sfx}.csv"), per_tracer_rows(res, split_indices))
            failed = set(res["failed_idx"])
            write_csv(os.path.join(sweep_dir, f"hybrid_diag_{stem}{sfx}.csv"), [dict(position=i, global_id=int(split_indices[i]), failed=int(i in failed), **{k: float(diag[k][i]) for k in DIAG_KEYS}) for i in range(res["n_tracers"])])
            rows_all += combo

    write_csv(csv_path, rows_all)
    print(f"[hybrid] wrote {csv_path}  ({len(rows_all)} rows)", flush=True)
    return read_table(csv_path)


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.out_dir = a.out_dir or default_out_dir("osu", a.model)

    from src.data import load_context
    from src.hybrid import HybridConfig
    from src.pipeline import CASE_FIGURES, case_config, draw_figures, rank_sweep, run_sweep, split_paths

    opts = CASE_FIGURES["osu"]
    # Built and validated before any solve, so a bad lookback/n_fom pair fails in seconds rather than after the sweep.
    hybrid = HybridConfig(positivity=opts["hybrid_positivity"], thresh=a.hybrid_thresh, lookback=a.hybrid_lookback, n_fom=a.hybrid_n_fom)
    hyb_etas = _subset(a.hybrid_etas, a.etas, "--hybrid-etas")
    hyb_taus = _subset(a.hybrid_taus, a.taus, "--hybrid-taus")

    fig_dir = os.path.join(a.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = case_config("osu", a.out_dir, chemical_hydro_file=a.chemical_hydro_file, build_features=a.build_features, qoi=a.qoi, n_train=a.n_train, n_test=a.n_test, n_eval=a.n_eval, n_clusters=a.n_clusters, max_clusters=a.max_clusters, min_cluster_size=a.min_cluster_size, n_workers=a.n_workers or (os.cpu_count() or 1))
    print(f"[osu] {len(a.etas)} eta x {len(a.taus)} tau -> {a.out_dir}", flush=True)
    ctx = load_context(cfg)

    # The global single-basis baseline panel (a) compares against, evaluated first.
    rank_df, _ens_global, rho = rank_sweep(ctx, cfg, a.ranks or list(range(1, 15)), a.out_dir, eval_split=a.eval_split, overwrite=a.overwrite)

    sweep_df, fm_train, fom, qoi_names = run_sweep(ctx, cfg, a.etas, a.taus, (True, False), a.out_dir, eval_split=a.eval_split, overwrite=a.overwrite)

    hybrid_df = None
    if not a.no_hybrid_sweep:
        print(f"[osu] hybrid: {len(hyb_etas)} eta x {len(hyb_taus)} tau, {hybrid.summary()}", flush=True)
        hybrid_df = hybrid_sweep(ctx, cfg, hyb_etas, hyb_taus, hybrid, fom, a.out_dir, eval_split=a.eval_split, overwrite=a.overwrite)

    if a.no_figures:
        print("[osu] done (figures skipped)")
        return 0

    draw_figures(a.out_dir, sweep_df, rank_df, qoi_names, a.taus, a.etas, n_eval=int(min(cfg.n_eval, len(split_paths(ctx, a.eval_split)[2]))), max_clusters=cfg.max_clusters, eval_split=a.eval_split, positivity_label=opts["positivity_label"], combo=tuple(a.figure_combo) if a.figure_combo else None, ctx=ctx, cfg=cfg, fm_train=fm_train, fom=fom, tracer=a.tracer, global_rank=a.global_rank, hybrid=hybrid, hybrid_df=hybrid_df, time_scale=opts["time_scale"], time_label=opts["time_label"])
    print(f"[osu] done -> {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
