#!/usr/bin/env python
"""Nelson chemistry run: eta x tau sweep, evaluation, figures.

    python -m scripts.run_nelson --out-dir experiments/R0.1_M6 \
        --chemical-hydro-file datasets/grav_collapse/nelson/piecewise_constant/R0.1_M6_trace_cells.npy

The run reads the provided feature matrices from out_dir (see the README). --build-features rebuilds them instead, and needs the raw chemistry dataset, which is not distributed.
"""
from __future__ import annotations

import argparse
import os
import sys

# The package root, so the drivers run from anywhere: `python run_nelson.py` inside scripts/, or by full path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import default_out_dir


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="R0.1_M6.0",
                   help="dataset identifier; names the run directory")
    p.add_argument("--out-dir", default=None,
                   help="run directory; defaults to experiments/<case>/<model>")
    p.add_argument("--chemical-hydro-file", default=None,
                   help="override the .npy holding the hydro parameters and the chemistry; "
                        "defaults to datasets/grav_collapse/nelson/, and is read only by --build-features")
    p.add_argument("--build-features", action="store_true",
                   help="rebuild the split feature matrices from the raw dataset")
    p.add_argument("--etas", type=float, nargs="+",
                   default=[1e-2, 1e-4, 1e-6, 1e-8],
                   help="residual SVD energy values to sweep")
    p.add_argument("--taus", type=float, nargs="+", default=[0.01, 0.10, 0.40],
                   help="split tolerances to sweep")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--max-clusters", type=int, default=1024)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=2048)
    p.add_argument("--n-eval", type=int, default=2048,
                   help="trajectories of the evaluated dataset integrated per combination")
    p.add_argument("--qoi", nargs="+", default=["CO", "e-"])
    p.add_argument("--n-workers", type=int, default=None,
                   help="process-pool size; default is the core count")
    p.add_argument("--tracer", type=int, default=0,
                   help="position within the evaluated split the per-tracer figures use")
    p.add_argument("--global-rank", type=int, default=5,
                   help="rank of the single-basis baseline in the figures")
    p.add_argument("--figure-combo", type=float, nargs=2,
                   metavar=("TAU", "ETA"), default=None,
                   help="combination the per-trajectory figures use; default is the middle of each sweep")
    p.add_argument("--ranks", type=int, nargs="+", default=None,
                   help="encoder ranks for the global-ROM baseline; default 1..14, "
                        "ranks above the basis's rho are dropped")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--eval-split", default="test", choices=["test", "train"],
                   help="which split to roll out; fits always train on the train split")
    p.add_argument("--overwrite", action="store_true",
                   help="redo the run in place instead of refusing on an existing CSV")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.out_dir = a.out_dir or default_out_dir("nelson", a.model)

    from src.data import load_context
    from src.hybrid import HybridConfig
    from src.pipeline import (CASE_FIGURES, case_config,
                              draw_figures, rank_sweep, run_sweep,
                              split_paths)

    fig_dir = os.path.join(a.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = case_config(
        "nelson", a.out_dir, chemical_hydro_file=a.chemical_hydro_file,
        build_features=a.build_features,
        qoi=a.qoi,
        n_train=a.n_train,
        n_test=a.n_test,
        n_eval=a.n_eval,
        n_clusters=a.n_clusters,
        max_clusters=a.max_clusters,
        min_cluster_size=a.min_cluster_size,
        n_workers=a.n_workers or (os.cpu_count() or 1))
    print(f"[nelson] {len(a.etas)} eta x {len(a.taus)} tau -> {a.out_dir}",
          flush=True)
    ctx = load_context(cfg)

    # The global single-basis baseline panel (a) compares against, evaluated first.
    rank_df, _ens_global, rho = rank_sweep(
        ctx, cfg, a.ranks or list(range(1, 15)), a.out_dir,
        eval_split=a.eval_split, overwrite=a.overwrite)

    sweep_df, fm_train, fom, qoi_names = run_sweep(
        ctx, cfg, a.etas, a.taus, (True, False), a.out_dir,
        eval_split=a.eval_split, overwrite=a.overwrite)

    if a.no_figures:
        print("[nelson] done (figures skipped)")
        return 0

    opts = CASE_FIGURES["nelson"]
    draw_figures(
        a.out_dir, sweep_df, rank_df, qoi_names, a.taus, a.etas,
        n_eval=int(min(cfg.n_eval, len(split_paths(ctx, a.eval_split)[2]))),
        max_clusters=cfg.max_clusters, eval_split=a.eval_split,
        positivity_label=opts["positivity_label"],
        combo=tuple(a.figure_combo) if a.figure_combo else None,
        ctx=ctx, cfg=cfg, fm_train=fm_train, fom=fom, tracer=a.tracer,
        global_rank=a.global_rank,
        hybrid=HybridConfig(positivity=opts["hybrid_positivity"]),
        time_scale=opts["time_scale"], time_label=opts["time_label"])
    print(f"[nelson] done -> {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
