#!/usr/bin/env python
"""Redraw a completed run's figures from its cached artifacts. Fits nothing.

    python -m scripts.make_figures --case nelson --out-dir experiments/R0.1_M6

The two summary panels come off the sweep CSVs alone and cost about a second,
so this is the loop to use when tuning a figure. The abundance and hybrid
figures each solve one trajectory against the cached bases, which costs tens
of seconds. Lorenz-96 needs ``--gen-dir`` for those two and skips them without
it; chemistry reads everything it needs from the run directory.

The sweep grid is read off the table, so the tolerances and etas do not have
to be repeated here.
"""
from __future__ import annotations

import argparse
import os
import sys

# The package root, so the drivers run from anywhere: `python run_nelson.py` inside scripts/, or by full path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.pipeline import default_out_dir


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, choices=["nelson", "osu", "lorenz96"],
                   help="which system the run was for")
    p.add_argument("--model", default="R0.1_M6.0",
                   help="dataset identifier; names the run directory")
    p.add_argument("--out-dir", default=None,
                   help="run directory; defaults to experiments/<case>/<model>")
    p.add_argument("--chemical-hydro-file", default=None,
                   help="override the chemistry array; unused unless the run directory is incomplete")
    p.add_argument("--gen-dir", default=None,
                   help="Lorenz-96 generator output; omit to draw only the summary panels")
    p.add_argument("--eval-split", default="test", choices=["test", "train"],
                   help="which pass's outputs to draw")
    p.add_argument("--only", nargs="+", default=None,
                   choices=["panel", "violin", "abundance", "hybrid"],
                   help="draw a subset (default: all four)")
    p.add_argument("--figure-combo", type=float, nargs=2, metavar=("TAU", "ETA"),
                   default=None,
                   help="combination the per-trajectory figures use; default is the middle of the grid")
    p.add_argument("--tracer", type=int, default=0,
                   help="position within the evaluated split the per-tracer figures use")
    p.add_argument("--global-rank", type=int, default=5,
                   help="rank of the single-basis baseline in the abundance figure")
    p.add_argument("--n-eval", type=int, default=None,
                   help="tracers the run evaluated; default is inferred from the table")
    p.add_argument("--max-clusters", type=int, default=1024,
                   help="the run's cluster cap, which panel (c) draws as a ceiling; "
                        "not recorded in the CSV, so pass it if it was not the default")
    p.add_argument("--n-workers", type=int, default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.out_dir = a.out_dir or default_out_dir(a.case, a.model)

    from src import plots as P
    from src.pipeline import (CASE_FIGURES, case_config,
                              draw_figures, load_case_context,
                              split_paths)
    from src.data import load_trajectories, open_feature_matrix
    from src.hybrid import HybridConfig

    sweep_df, rank_df, rho = P.load_frames(a.out_dir, eval_split=a.eval_split)
    taus = sorted(np.unique(sweep_df["tau_split"]))
    etas = sorted(np.unique(sweep_df["eta"]), reverse=True)
    # The run recorded how many it solved per combination; with no failures anywhere that is how many it attempted.
    n_eval = a.n_eval or int(np.max(sweep_df["n_solved"]))
    opts = CASE_FIGURES[a.case]
    print(f"[figures] {a.case} {a.eval_split}: {len(etas)} eta x {len(taus)} tau, "
          f"n_eval={n_eval}, rho={rho}", flush=True)

    ctx = cfg = fm_train = fom = None
    # Chemistry needs no dataset: the context comes from the parameters file and the per-tracer figures from the feature matrices.
    have_data = a.gen_dir if a.case == "lorenz96" else True
    if have_data:
        cfg = case_config(a.case, a.out_dir, chemical_hydro_file=a.chemical_hydro_file,
                          gen_dir=a.gen_dir, n_eval=n_eval,
                          n_workers=a.n_workers or (os.cpu_count() or 1))
        ctx = load_case_context(a.case, cfg)
        fm_train = open_feature_matrix(ctx.train_feat_path,
                                       len(ctx.train_indices), ctx.n_steps,
                                       ctx.n_species)
        feat_path, split_tracers, split_indices = split_paths(ctx, a.eval_split)
        n_load = int(min(a.tracer + 1, len(split_indices)))
        t_foms, y_foms, x_eqs = load_trajectories(feat_path, n_load,
                                                  ctx.n_species)
        fom = (split_tracers[:n_load], x_eqs, t_foms, y_foms)

    # First-appearance order, not sorted: it is the order the run wrote and the panels lay out in.
    qoi_names = ([ctx.net.species[j] for j in ctx.qoi_indices] if ctx is not None
                 else list(dict.fromkeys(sweep_df["qoi"].tolist())))

    fig_dir = draw_figures(
        a.out_dir, sweep_df, rank_df, qoi_names, taus, etas, n_eval,
        max_clusters=a.max_clusters,
        eval_split=a.eval_split, positivity_label=opts["positivity_label"],
        combo=tuple(a.figure_combo) if a.figure_combo else None,
        ctx=ctx, cfg=cfg, fm_train=fm_train, fom=fom, tracer=a.tracer,
        global_rank=a.global_rank,
        hybrid=HybridConfig(positivity=opts["hybrid_positivity"]),
        time_scale=opts["time_scale"], time_label=opts["time_label"],
        only=a.only)
    print(f"[figures] done -> {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
