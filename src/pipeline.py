"""The eta x tau sweep both drivers run, and the artifact layout it writes.

    <out_dir>/
      tol_eta_sweep<sfx>.csv     every combination, concatenated
      sweep/
        cluster_<stem>.npz       partition, shared by both datasets
        ensemble_<stem>.npz      bases, shared by both datasets
        metrics_<stem><sfx>.csv  that combination alone
        pertracer_<stem>_<on|off><sfx>.csv  per-tracer error next to the global tracer id
      figures/                   PDFs and text reports

``<sfx>`` is empty for the testing dataset and ``_train`` otherwise, so the two passes cannot overwrite each other's evaluation outputs.

One ``build_adaptive`` call covers every tolerance at a given eta: the base fit and first scoring pass do not depend on the tolerance. That is also the caching granularity -- an eta is reloaded only when every one of its tolerances is on disk, so a job killed mid-eta refits that eta in full.
"""
from __future__ import annotations

import os

import numpy as np

from .cluster import Cluster
from .config import NSEC, combo_stem, split_suffix
from .data import load_trajectories, open_feature_matrix
from .evaluate import (AGGREGATES, columns_of, evaluate, metrics_rows,
                                 per_tracer_rows, read_table, where, write_csv)
from .hybrid import conservation_matrix
from .rom import EnsembleROM, fit_global_rom


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
# Per-system settings in one place: the drivers and the figure scripts both configure from here, so a network path or a QoI list cannot drift between a sweep and the figures drawn from it.

CASES = ("nelson", "osu", "lorenz96")


def case_config(case, out_dir, chemical_hydro_file=None, gen_dir=None, **over):
    """The configuration ``case`` runs with, with ``over`` applied on top."""
    from .config import REPO_ROOT, Config
    from .config_lorenz96 import LorenzConfig

    if case == "lorenz96":
        base = dict(out_dir=out_dir, gen_dir=gen_dir)
        base.update(over)
        return LorenzConfig(**base)

    base = dict(out_dir=out_dir)
    if case == "osu":
        base.update(
            network=os.path.join(REPO_ROOT, "networks/osu_09_2008/gas_reactions.in"),
            abundances=os.path.join(REPO_ROOT, "networks/osu_09_2008/abundances.in"),
            chemical_hydro_file=os.path.join(
                REPO_ROOT, "datasets/grav_collapse/osu_09_2008/R0.1_M6_trace_cells.npy"),
            qoi=["CO", "C+", "e-", "O", "C"],
            log_cols=["nH"], drop_cols=["Tgrain", "Av", "uv_flux"])
    elif case != "nelson":
        raise ValueError(f"case must be one of {CASES}; got {case!r}")
    # Only an explicit path overrides the per-network default.
    if chemical_hydro_file:
        base["chemical_hydro_file"] = chemical_hydro_file
    base.update(over)
    return Config(**base)


def default_out_dir(case, model):
    """Run directory for a case and dataset: ``experiments/<case>/<model>``.

    One dataset per directory, so two models of the same system do not overwrite each other's artifacts.
    """
    from .config import REPO_ROOT
    return os.path.join(REPO_ROOT, "experiments", case, model)


def load_case_context(case, cfg):
    """``load_context`` for ``case``."""
    if case == "lorenz96":
        from .lorenz96 import load_context
    else:
        from .data import load_context
    return load_context(cfg)


# Figure conventions fixed by the system rather than the run: a signed state has no positivity axis, and chemistry times are plotted in kyr.
CASE_FIGURES = {
    "nelson": dict(positivity_label="on", time_scale=NSEC * 1e3,
                   time_label="Time [kyr]", hybrid_positivity=True),
    "osu": dict(positivity_label="on", time_scale=NSEC * 1e3,
                time_label="Time [kyr]", hybrid_positivity=True),
    "lorenz96": dict(positivity_label="off", time_scale=1.0,
                     time_label="Time", hybrid_positivity=False),
}


def _say(msg):
    print(f"[sweep] {msg}", flush=True)


def _build_eta(ctx, cfg, fm_train, base_cluster, taus, sweep_dir, overwrite):
    """Every combination at one eta, fitted in one pass or loaded from disk.

    Yields ``(tau, stem, ensemble, cluster)``.
    """
    stems = [combo_stem(tau, cfg.eta) for tau in taus]
    paths = [(os.path.join(sweep_dir, f"cluster_{s}"),
              os.path.join(sweep_dir, f"ensemble_{s}")) for s in stems]

    if not overwrite and all(os.path.exists(p + ".npz")
                             for pair in paths for p in pair):
        _say(f"eta={cfg.eta:g}: reusing {len(taus)} cached combination(s)")
        fitted = []
        for cl_path, ens_path in paths:
            cl = Cluster.load(cl_path + ".npz")
            fitted.append((EnsembleROM.load(ens_path + ".npz", cl, ctx.net), cl))
    else:
        # The tolerance list sets each one's split RNG stream, so record it.
        _say(f"eta={cfg.eta:g}: fitting {len(taus)} tolerance(s) "
             f"{[float(t) for t in taus]} on one shared base pass")
        fitted = EnsembleROM.build_adaptive(
            base_cluster, ctx.net, fm_train, ctx.n_steps, ctx.dt_hydro, cfg,
            taus, ctx.qoi_indices)
        for (cl_path, ens_path), (ens, cl) in zip(paths, fitted):
            cl.save(cl_path)
            ens.save(ens_path)

    for tau, stem, (ens, cl) in zip(taus, stems, fitted):
        yield tau, stem, ens, cl


def split_paths(ctx, eval_split):
    """``(feat_path, tracers, indices)`` of the dataset being evaluated."""
    if eval_split not in ("test", "train"):
        raise ValueError(f"eval_split must be 'test' or 'train'; got {eval_split!r}")
    if eval_split == "train":
        return ctx.train_feat_path, ctx.train_tracers, ctx.train_indices
    return ctx.test_feat_path, ctx.test_tracers, ctx.test_indices


def rank_sweep(ctx, cfg, ranks, out_dir, eval_split="test", overwrite=False):
    """Single-cluster global ROM error against basis size.

    The same build with one cluster and no refinement, so the trial space is a single global POD basis; sweeping the rank shows what the clustered ensemble buys at equal cost.

    The positivity projection is forced off regardless of ``cfg``: the global ROM's failure rate is what motivates the clustered ensemble, and correcting those tracers drives the count to ~0.

    Ranks above the basis's own ``rho`` are dropped rather than clamped; clamping would repeat the last point and read as convergence.

    Each row records ``rho``, ``residual_energy`` (``1 - sum`` of the retained variance ratios) and ``invariant_residual`` (``||W D U|| / (||W|| ||D U||)``). Read them together: once the residual energy reaches 0 the basis is exhausted, and a flat error tail there is not convergence.
    """
    sfx = split_suffix(eval_split)
    csv_path = os.path.join(out_dir, f"single_cluster_rank_sweep{sfx}.csv")
    if os.path.exists(csv_path) and not overwrite:
        raise FileExistsError(
            f"{csv_path} exists; pass --overwrite to redo the run in place.")

    fm_train = open_feature_matrix(ctx.train_feat_path, len(ctx.train_indices),
                                  ctx.n_steps, ctx.n_species)
    qoi_names = [ctx.net.species[j] for j in ctx.qoi_indices]
    # Fitted at full rank, then re-sliced per candidate; set_rank refreshes the cached B slices.
    ens = fit_global_rom(fm_train, ctx.net, cfg, ctx.cluster_indices,
                         ctx.param_names, ctx.n_species, cfg.n_workers)
    basis = next(iter(ens.roms.values())).basis
    rho = basis.max_rank
    evr = np.nan_to_num(basis.pca.explained_variance_ratio_)

    # A network without reactions has no invariants, so there is no conservation residual to report.
    W = conservation_matrix(ctx.net) if getattr(ctx.net, "reactions", None) else None
    nW = np.linalg.norm(W, 2) if W is not None else 0.0

    feat_path, split_tracers, split_indices = split_paths(ctx, eval_split)
    n_eval = int(min(cfg.n_eval, len(split_indices)))
    t_foms, y_foms, x_eqs = load_trajectories(feat_path, n_eval, ctx.n_species)
    tracers = split_tracers[:n_eval]

    requested = [int(r) for r in ranks]
    dropped = [r for r in requested if r > rho]
    if dropped:
        _say(f"rho={rho}: ranks {dropped} exceed the basis and are dropped "
             f"(residual energy is already "
             f"{max(1.0 - evr[:rho].sum(), 0.0):.3e} at rank {rho})")

    cfg_off = cfg.replace(positivity=False)
    rows = []
    for r in requested:
        if r > rho:
            continue
        basis.set_rank(r)
        res = evaluate(ens, tracers, x_eqs, t_foms, y_foms, ctx.qoi_indices,
                       qoi_names, ctx.dt_hydro, cfg_off, verbose=False)
        Rm = basis._R
        inv = (float(np.linalg.norm(W @ Rm, 2)
                     / (nW * np.linalg.norm(Rm, 2) + 1e-300))
               if W is not None else float("nan"))
        resid_energy = float(max(1.0 - evr[:r].sum(), 0.0))
        failed = int(res["n_tracers"] - res["n_solved"])
        for agg in AGGREGATES:
            for q in qoi_names:
                rows.append(dict(rank=int(r), rho=rho,
                                 residual_energy=resid_energy,
                                 invariant_residual=inv, agg=agg, qoi=q,
                                 n_solved=res["n_solved"],
                                 n_valid=res["n_valid"], failed=failed,
                                 **res[agg][q]))
        line = (f"rank {r:>3}/{rho}: solved {res['n_solved']}/{res['n_tracers']}"
                f" | resid_energy {resid_energy:.2e}")
        _say(line + (f" | invariant_residual {inv:.2e}" if W is not None else ""))

    write_csv(csv_path, rows)
    _say(f"wrote {csv_path}  ({len(rows)} rows)")
    return read_table(csv_path), ens, rho


def run_sweep(ctx, cfg, etas, taus, positivity_modes, out_dir,
              eval_split="test", overwrite=False):
    """Fit and evaluate every (eta, tau) combination; return the concatenated frame.

    ``positivity_modes`` is the positivity settings each fit is evaluated under, one CSV row group per mode. A signed state passes ``(False,)``.

    The fits do not depend on which dataset is evaluated -- ``build_adaptive`` always trains on the training dataset -- so ``eval_split`` only chooses what is integrated, and the cluster/ensemble artifacts stay unsuffixed and shared between passes. A second pass over the other dataset reuses them rather than refitting.

    Returns ``(sweep_df, fm_train, fom, qoi_names)``; the last two are what the per-tracer figures need, so they are not reloaded.
    """
    sweep_dir = os.path.join(out_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    sfx = split_suffix(eval_split)
    csv_path = os.path.join(out_dir, f"tol_eta_sweep{sfx}.csv")
    if os.path.exists(csv_path) and not overwrite:
        raise FileExistsError(
            f"{csv_path} exists; pass --overwrite to redo the run in place.")

    fm_train = open_feature_matrix(ctx.train_feat_path, len(ctx.train_indices),
                                  ctx.n_steps, ctx.n_species)
    qoi_names = [ctx.net.species[j] for j in ctx.qoi_indices]

    feat_path, split_tracers, split_indices = split_paths(ctx, eval_split)
    # Clamped to the dataset being evaluated: load_trajectories takes the first n of it, so an n_eval sized for the testing dataset would silently evaluate only part of the larger training one.
    n_eval = int(min(cfg.n_eval, len(split_indices)))
    _say(f"loading {n_eval} of {len(split_indices)} {eval_split} FOM trajectories")
    t_foms, y_foms, x_eqs = load_trajectories(feat_path, n_eval, ctx.n_species)
    tracers = split_tracers[:n_eval]

    # Tolerance- and eta-independent, so the base partition is fitted once.
    base_cluster = Cluster(ctx.param_names)
    base_cluster.fit(fm_train, ctx.cluster_indices, cfg.n_clusters,
                     cfg.log_cols, cfg.drop_cols, qoi_log=cfg.qoi_log,
                     seed=cfg.base_seed)

    rows_all = []
    for eta in etas:
        cfg_eta = cfg.replace(eta=eta)
        for tau, stem, ens, cl in _build_eta(ctx, cfg_eta, fm_train,
                                             base_cluster, taus, sweep_dir,
                                             overwrite):
            ens.precompute_B_structure()
            meta = dict(n_clusters=int(cl.n_clusters),
                        mean_rank=ens.mean_rank(),
                        **ens.summary_columns())
            combo = []
            for on in positivity_modes:
                label = "on" if on else "off"
                res = evaluate(ens, tracers, x_eqs, t_foms, y_foms,
                               ctx.qoi_indices, qoi_names, ctx.dt_hydro,
                               cfg_eta.replace(positivity=on), verbose=False)
                _say(f"{stem} positivity {label}: {res['n_valid']}"
                     f"/{res['n_tracers']} complete, k={cl.n_clusters}, "
                     f"mean rank {meta['mean_rank']:.2f}")
                combo += metrics_rows(res, qoi_names, tau_split=float(tau),
                                      eta=float(eta), positivity=label,
                                      eval_split=eval_split, **meta)
                write_csv(os.path.join(sweep_dir,
                                       f"pertracer_{stem}_{label}{sfx}.csv"),
                          per_tracer_rows(res, split_indices))
            write_csv(os.path.join(sweep_dir, f"metrics_{stem}{sfx}.csv"), combo)
            rows_all += combo

    write_csv(csv_path, rows_all)
    _say(f"wrote {csv_path}  ({len(rows_all)} rows)")
    return (read_table(csv_path), fm_train, (tracers, x_eqs, t_foms, y_foms),
            qoi_names)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def draw_figures(out_dir, sweep_df, rank_df, qoi_names, taus, etas, n_eval, max_clusters, eval_split="test", positivity_label="on", combo=None, ctx=None, cfg=None, fm_train=None, fom=None, tracer=0, global_rank=5, hybrid=None, time_scale=1.0, time_label="Time [s]", only=None, hybrid_df=None):
    """Draw the figures into ``<out_dir>/figures``; return that directory.

    The two summary panels come off the tables alone. The abundance and hybrid figures each need one trajectory solved, so they are skipped unless ``ctx``, ``cfg`` and ``fom`` are supplied. ``only`` restricts the set to the named figures. ``hybrid_df``, a hybrid sweep table, is overlaid on the clustered-error panel.
    """
    from . import plots as P
    import matplotlib.pyplot as plt

    want = set(only) if only else {"panel", "violin", "abundance", "hybrid"}
    fig_dir = os.path.join(out_dir, "figures")
    sweep_dir = os.path.join(out_dir, "sweep")
    os.makedirs(fig_dir, exist_ok=True)
    taus, etas = list(taus), list(etas)
    twin = taus[len(taus) // 2]

    # One positivity mode only: the sweep writes a row group per mode, and the panels average every row sharing an (eta, qoi), which would silently mix the corrected and uncorrected errors.
    sweep_df = where(sweep_df, positivity=positivity_label)

    if "panel" in want:
        P.sweep_panel(sweep_df, rank_df, qoi_names, n_eval=n_eval, max_clusters=max_clusters, show_tols=tuple(taus), twin_tol=twin, split=eval_split, savepath=os.path.join(fig_dir, "panel.pdf"), hybrid_df=hybrid_df)
    if "violin" in want:
        P.violin_panel(sweep_dir, taus, qoi_names, etas=etas,
                       label=positivity_label, eval_split=eval_split,
                       savepath=os.path.join(fig_dir, "violin.pdf"))
    plt.close("all")

    if not (want & {"abundance", "hybrid"}):
        return fig_dir
    if ctx is None or cfg is None or fom is None:
        print("  no context loaded; skipping the per-tracer figures")
        return fig_dir

    tau, eta = combo or (twin, etas[len(etas) // 2])
    stem = combo_stem(tau, eta)
    cfg_fig = cfg.replace(eta=eta)
    split_ids = split_paths(ctx, eval_split)[2]

    if "abundance" in want and fm_train is not None:
        P.tracer_figures(sweep_dir, stem, fig_dir, ctx, cfg_fig, fm_train,
                         *fom, tracer, global_rank, global_ids=split_ids)
    if "hybrid" in want:
        P.hybrid_figures(sweep_dir, stem, fig_dir, ctx, cfg_fig, *fom, tracer,
                         hybrid=hybrid, time_scale=time_scale,
                         time_label=time_label)
    plt.close("all")
    return fig_dir
