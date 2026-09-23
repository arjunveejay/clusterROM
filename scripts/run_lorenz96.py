#!/usr/bin/env python
"""Lorenz-96 forcing-ensemble run: eta x tau sweep, evaluation, figures.

    python -m scripts.run_lorenz96 --build-features

The raw dataset is not shipped; generate it first with
``python -m src.lorenz96`` (see the Lorenz slurm script).

The error splitting criterion uses absolute error and the
positivity correction step is deactivated.
"""
from __future__ import annotations

import argparse
import os
import sys

# The package root, so the drivers run from anywhere: `python run_nelson.py` inside scripts/, or by full path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import default_out_dir

from src.config import REPO_ROOT


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="dataset identifier; defaults to the --gen-dir name")
    p.add_argument("--out-dir", default=None,
                   help="run directory; defaults to experiments/lorenz96/<model>")
    p.add_argument("--gen-dir",
                   default=os.path.join(REPO_ROOT, "datasets/lorenz96/N100_F0-2.5"),
                   help="generator output holding feature_matrix.npy and meta.npz")
    p.add_argument("--build-features", action="store_true",
                   help="rebuild the shifted source and the split feature matrices")
    p.add_argument("--etas", type=float, nargs="+",
                   default=[1e-2, 1e-4, 1e-6, 1e-8])
    p.add_argument("--taus", type=float, nargs="+", default=[0.005, 0.02, 0.05],
                   help="split tolerances in state units")
    p.add_argument("--n-clusters", type=int, default=16)
    p.add_argument("--max-clusters", type=int, default=2048)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--qoi", type=int, nargs="+", default=[0, 25, 50, 75],
                   help="state components the error is measured on")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--tracer", type=int, default=0,
                   help="position within the evaluated split the per-forcing figure uses")
    p.add_argument("--global-rank", type=int, default=8)
    p.add_argument("--figure-combo", type=float, nargs=2,
                   metavar=("TAU", "ETA"), default=None)
    p.add_argument("--ranks", type=int, nargs="+", default=None,
                   help="encoder ranks for the global-ROM baseline; default "
                        "every 4th up to N, ranks above rho are dropped")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--eval-split", default="test", choices=["test", "train"],
                   help="which split to roll out; fits always train on the train split")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    model = a.model or os.path.basename(a.gen_dir.rstrip("/"))
    a.out_dir = a.out_dir or default_out_dir("lorenz96", model)

    from src.lorenz96 import load_context
    from src.pipeline import (CASE_FIGURES, case_config,
                              draw_figures, rank_sweep, run_sweep,
                              split_paths)
    from src.hybrid import HybridConfig

    fig_dir = os.path.join(a.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = case_config(
        "lorenz96", a.out_dir, gen_dir=a.gen_dir,
        build_features=a.build_features,
        qoi=a.qoi,
        n_train=a.n_train,
        n_test=a.n_test,
        n_eval=a.n_eval,
        n_clusters=a.n_clusters,
        max_clusters=a.max_clusters,
        min_cluster_size=a.min_cluster_size,
        n_workers=a.n_workers or (os.cpu_count() or 1))
    print(f"[lorenz96] {len(a.etas)} eta x {len(a.taus)} tau "
          f"({cfg.error_norm} criterion) -> {a.out_dir}", flush=True)
    ctx = load_context(cfg)

    ranks = a.ranks or (list(range(2, ctx.n_species + 1, 4)) + [ctx.n_species])
    rank_df, _ens_global, rho = rank_sweep(
        ctx, cfg, ranks, a.out_dir, eval_split=a.eval_split,
        overwrite=a.overwrite)

    # Positivity is undefined for a signed state, so there is no on/off axis.
    sweep_df, fm_train, fom, qoi_names = run_sweep(
        ctx, cfg, a.etas, a.taus, (False,), a.out_dir,
        eval_split=a.eval_split, overwrite=a.overwrite)

    if a.no_figures:
        print("[lorenz96] done (figures skipped)")
        return 0

    opts = CASE_FIGURES["lorenz96"]
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
    print(f"[lorenz96] done -> {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
