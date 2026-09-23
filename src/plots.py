"""The figures, all fed from artifacts a run already wrote.

``sweep_panel``            the four-panel summary for one evaluation split
``sweep_panel_splits``     the same, train and test side by side
``abundance_panel``         one tracer's species trajectories over its route
``hybrid_comparison_panel`` hybrid vs pure ROM vs reference, over route bands
``violin_panel``            per-tracer error distribution vs eta

The summary panels come off the two sweep CSVs alone. ``abundance_panel`` and the hybrid panel need one tracer actually solved, so they are much slower; ``tracer_figures`` and ``hybrid_figures`` load a fit and drive those two.

Artifact naming (``eta_tag``, ``combo_stem``) lives in :mod:`config`, so the compute path never imports matplotlib; this module imports it to find combos on disk.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import offset_copy

from . import diagnostics as D
from .config import combo_stem, split_suffix
from .evaluate import columns_of, pivot_mean, read_table, where
from .cluster import Cluster
from .hybrid import FOM, HybridConfig, solve_hybrid, solve_pure_rom
from .rom import EnsembleROM, fit_global_rom


LIGHT = {0.005: "#b39ddb", 0.01: "#7fc4f2", 0.05: "#c9b3e6", 0.10: "#f5a86a",
         0.20: "#f2b9c4", 0.40: "#a9d6a9"}

DARK = {0.005: "#4a148c", 0.01: "#1f77b4", 0.05: "#6a3d9a", 0.10: "#d9601b",
        0.20: "#c2185b", 0.40: "#2e7d32"}

TOL_PAIRS = [("#7fc4f2", "#1f77b4"),        # blue
             ("#f5a86a", "#d9601b"),        # orange
             ("#a9d6a9", "#2e7d32"),        # green
             ("#c9b3e6", "#6a3d9a"),        # purple
             ("#f2b9c4", "#c2185b"),        # pink
             ("#bdbdbd", "#525252")]

HYBRID_MARKER = "*"      # hybrid overlay on the clustered-error panel; the tau colour is kept

ETA_MIN = 1e-16

FAIL_COLOR = "tab:red"

GLOBAL_COLOR = "k"

SPLIT_CAPTION = {"test": "Testing", "train": "Training"}

SPLIT_WORD = {"test": "testing", "train": "training"}

PAPER_STYLE = {
    "text.usetex": False,
    "font.size": 18,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _first_by(tab, key, value):
    """``{key: value}`` taking the first row at each distinct key.

    A sweep table holds one row per (aggregate, QoI, statistic), so a per-combination column such as ``n_clusters`` repeats and only one occurrence is wanted.
    """
    out = {}
    for k, v in zip(tab[key], tab[value]):
        out.setdefault(k.item() if hasattr(k, "item") else k, v)
    return out


def drop_full_basis(rank_df):
    """Drop rank rows whose residual energy is exactly zero.

    A rank sweep reaching the full state dimension ends with ``eta(r) = 1 - cumsum(evr) == 0``. Panel (a)'s x axis is log-eta, so that point lands on the floor (~1e-16) and stretches the panel over sixteen decades, squeezing every other rank into the left quarter. It is not a result either: a ROM that keeps every mode is the full-order model.

    A no-op for sweeps that stop short of the state dimension, which is every chemistry run.
    """
    keep = rank_df["residual_energy"] > 0
    if keep.all():
        return rank_df
    dropped = sorted({int(r) for r in rank_df["rank"][~keep]})
    print(f"  dropping rank {dropped} from the rank sweep: residual energy is "
          f"0 (full basis), which has no place on a log-eta axis")
    return rank_df[keep]


def load_frames(out_dir, eval_split="test", full_basis=False):
    """The (sweep, rank sweep) pair a run wrote, plus the global basis's rho.

    The full-basis rank row is dropped unless ``full_basis`` is set; see :func:`drop_full_basis`.
    """
    sfx = split_suffix(eval_split)
    paths = (os.path.join(out_dir, f"single_cluster_rank_sweep{sfx}.csv"),
             os.path.join(out_dir, f"tol_eta_sweep{sfx}.csv"))
    missing = [q for q in paths if not os.path.exists(q)]
    if missing:
        have = sorted(f for f in os.listdir(out_dir)
                      if f.endswith(".csv")) if os.path.isdir(out_dir) else []
        raise FileNotFoundError(
            f"no {eval_split}-split results in {out_dir}: "
            f"{[os.path.basename(q) for q in missing]} absent. "
            f"CSVs present: {have or 'none'}")
    rank_df, sweep_df = (read_table(q) for q in paths)
    rho = int(rank_df["rho"][0])
    if not full_basis:
        rank_df = drop_full_basis(rank_df)
    return sweep_df, rank_df, rho


# ---------------------------------------------------------------------------
# Shared series and style
# ---------------------------------------------------------------------------


def use_paper_style():
    plt.rcParams.update(PAPER_STYLE)

def tol_colors(tols):
    """``{tau_split: (light, dark)}`` -- one distinct colour pair per tolerance.

    ``LIGHT``/``DARK`` are used verbatim when every requested tolerance is a key of both. Any other grid gets ``TOL_PAIRS`` assigned in ascending-tolerance order instead; mixing the two sources would hand two different tolerances the same colour.
    """
    tols = list(tols)
    if all(t in DARK for t in tols):
        return {t: (LIGHT[t], DARK[t]) for t in tols}
    order = sorted(tols)
    return {t: TOL_PAIRS[order.index(t) % len(TOL_PAIRS)] for t in tols}


def mean_series(df, tau, qoi, agg="l2-in-time"):
    """(eta, error) for one tolerance: mean over tracers, then over QoIs.

    ``stat="mean"`` is already the per-tracer mean that ``evaluate`` stores; averaging those across QoIs weights each QoI equally.
    """
    rows = where(df, **{"tau_split": tau, "agg": agg, "stat": "mean"})
    idx, y = pivot_mean(rows, index="eta", values="value", columns="qoi",
                        keys=list(qoi))
    x = np.asarray(idx, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]

def rank_series(df, tau):
    """(eta, mean basis size) for one tolerance."""
    lut = _first_by(where(df, **{"tau_split": tau}), "eta", "mean_rank")
    keys = sorted(lut)
    sub = np.array([lut[k] for k in keys], dtype=float)
    if not np.isfinite(sub).any():
        raise ValueError(
            f"no mean_rank values for tau_split={tau}; the rank axes would be "
            "blank.")
    x = np.asarray(np.array(keys, dtype=float), dtype=float)
    order = np.argsort(x)
    return x[order], sub[order]

def global_rank_series(sc_df, qoi, agg="l2-in-time"):
    """(rank, error) for the single-cluster global ROM sweep."""
    return pivot_mean(where(sc_df, agg=agg), index="rank", values="mean",
                      columns="qoi", keys=list(qoi))


# ---------------------------------------------------------------------------
# The four-panel figure
# ---------------------------------------------------------------------------


def _fail_label(failed, n_eval):
    """Failure-rate caption that never rounds a real failure down to "0%".

    A caption fires only because tracers failed, so it must not then report none. Below 1% the label keeps a decimal, and anything nonzero that would still round to 0.0 falls back to the raw count.
    """
    pct = 100.0 * failed / max(n_eval, 1)
    if pct >= 1.0:
        return f"{pct:.0f}%"
    if pct >= 0.05:
        return f"{pct:.1f}%"
    return f"{int(failed)}/{int(n_eval)}"

def _annotate_failures(ax, x, y, failed, n_eval, color, fontsize=18, dy_pad=0,
                       others=None):
    """Caption the failure rate beside each point that had unsolved tracers.

    Same convention as the global-ROM panel: inline red text, never a second axis. Which side of the point it sits on follows the curve's local shape -- above a local maximum, below a local minimum, and on the side away from the incoming segment on a monotone stretch -- so the caption lands in empty space rather than on the neighbouring segment.

    ``dy_pad`` shifts the caption a further ``|dy_pad|`` points away from the curve, in whichever direction it already chose; it is a scalar or one value per point. Callers annotating several curves on one axes pass an increasing pad so captions at a shared x do not overprint, and ``color`` should then be the curve's own colour, since vertical order alone does not say which curve a caption belongs to. Pad only where captions genuinely collide: a blanket per-curve pad pushes a lone caption far from the point it describes, far enough to flip it onto a neighbouring curve.

    ``others`` is the other curves' values at the same x, shape ``(m, n)``, NaN where a curve has no point there. When given, the side with the larger gap to the nearest neighbouring curve wins, so the caption sits nearest the curve it belongs to. The local-shape rule is then only the tie-break, for where the curves coincide and no side is roomier.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = np.asarray(failed, dtype=float)
    n = x.size
    pad_pts = np.broadcast_to(np.asarray(dy_pad, dtype=float), (n,))
    # Where the point sits in the axes, so a caption near an edge is flipped inward rather than riding over the twin axis above or off the bottom. Requires the caller to have set the limits already.
    ylo, yhi = ax.get_ylim()
    if ax.get_yscale() == "log":
        to_frac = lambda v: ((np.log10(v) - np.log10(ylo))
                             / max(np.log10(yhi) - np.log10(ylo), 1e-12))
    else:
        to_frac = lambda v: (v - ylo) / max(yhi - ylo, 1e-12)
    for i in range(n):
        if not np.isfinite(f[i]) or f[i] <= 0:
            continue
        prev = y[i - 1] if i > 0 else None
        nxt = y[i + 1] if i < n - 1 else None
        if prev is not None and nxt is not None:
            if y[i] >= prev and y[i] >= nxt:
                above = True
            elif y[i] <= prev and y[i] <= nxt:
                above = False
            else:
                above = y[i] < prev
        else:
            above = True
        frac = to_frac(y[i]) if np.isfinite(y[i]) and y[i] > 0 else 0.5
        # How much of the axes the caption itself will occupy, offset included. Judging the available room from the point alone is not enough once dy_pad separates several curves' captions: a padded one clears the threshold and still lands on the twin axis above.
        h_pts = ax.get_window_extent().height / ax.figure.dpi * 72.0
        pad = (6 + pad_pts[i] + fontsize) / max(h_pts, 1.0)
        if others is not None:
            gap_up, gap_dn = 1.0 - frac, frac
            for o in np.asarray(others, dtype=float):
                v = o[i]
                if not np.isfinite(v) or v <= 0:
                    continue
                d = to_frac(v) - frac
                if abs(d) < 0.3 * pad:
                    # Effectively on top of this point, so it blocks neither side. Counting it as a hard zero gap would make coincident curves look like they block one direction and leave the other wide open.
                    continue
                if d > 0:
                    gap_up = min(gap_up, d)
                else:
                    gap_dn = min(gap_dn, -d)
            # Only override the shape rule when one side is genuinely roomier; coincident curves leave both gaps at their full extent and keep the shape tie-break.
            if max(gap_up, gap_dn) > pad and abs(gap_up - gap_dn) > 0.25 * pad:
                above = gap_up > gap_dn
        if above and frac + pad > 0.97:
            above = False
        elif not above and frac - pad < 0.03:
            above = True
        va, dy = ("bottom", 6 + pad_pts[i]) if above else ("top", -6 - pad_pts[i])
        ax.annotate(_fail_label(f[i], n_eval),
                    xy=(x[i], y[i]), xytext=(0, dy),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=fontsize, color=color, clip_on=False)
    return bool(np.any(f > 0))

def _panel_global(ax_sc, sc_df, qoi, n_eval, agg="l2-in-time",
                  sc_tick_fs=18, sc_tick_sep=0.16,
                  split="test", corner_label=None):
    """Draw the global ROM error vs eta into ``ax_sc``.

    Split out of ``sweep_panel`` so one figure can carry a panel per evaluation split.

    ``split`` only sets the y-label's wording. The data is whatever ``sc_df`` holds, and ``n_eval`` must match the dataset it was evaluated over, or every failure-rate caption is scaled by the ratio.
    """
    # eta(r) = 1 - (retained variance at rank r), which the rank sweep already records as `residual_energy`; reading it avoids refitting the global basis just to recover its spectrum.
    rank_sc, y_sc = global_rank_series(sc_df, qoi, agg)
    m = _first_by(sc_df, "rank", "residual_energy")
    eta_sc = np.array([max(float(m[r]), ETA_MIN) for r in rank_sc])

    # Ranks above rho have no residual energy to be placed at: the basis is exhausted, and they are left NaN so this panel drops them rather than clipping them onto rho's eta, which would read as convergence. Dropped from the array, not just skipped by `plot` -- a NaN left in `eta_sc` poisons the tick-label span and blanks every label on both x axes.
    _ok = np.isfinite(eta_sc) & (eta_sc > 0)
    if not _ok.all():
        rank_sc = np.asarray(rank_sc)[_ok]
        y_sc = np.asarray(y_sc)[_ok]
        eta_sc = eta_sc[_ok]
    if eta_sc.size == 0:
        raise ValueError("no rank in the sweep has a finite residual energy; "
                         "there is nothing to place on the eta axis")

    ax_sc.plot(eta_sc, y_sc, marker="o", ls="-", lw=2, ms=6, color=GLOBAL_COLOR)
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.invert_xaxis()
    ax_sc.set_xlabel(r"Residual SVD energy $\eta$")
    ax_sc.set_ylabel(f"Mean {SPLIT_WORD[split]} error")
    ax_sc.grid(alpha=0.3, which="major")
    if corner_label:
        # Top right, inside the axes: the twin basis-size axis owns the space above.
        ax_sc.text(0.98, 0.97, corner_label, transform=ax_sc.transAxes,
                   ha="right", va="top", fontsize=20, color="0.25", zorder=6)
    # Consecutive ranks can map to almost the same eta, which collides their labels. Keep a label only where it is `sc_tick_sep` of the axis span clear of the last one kept; the tick itself always stays. The threshold has to be a fraction of the span rather than a fixed number of decades, since a decade is a different share of the axis on every case.
    ax_sc.set_xticks(eta_sc)
    _e = np.log10(eta_sc)
    _span = max(_e.max() - _e.min(), 1e-12)
    # Decide keep/drop in screen order (eta descending, the axis is inverted) so the surviving labels spread evenly left to right rather than thinning from whichever end the rank sweep started at.
    _order = np.argsort(eta_sc)[::-1]
    _keep, _last = np.zeros(_e.size, dtype=bool), None
    for i in _order:
        if _last is None or abs(_e[i] - _last) >= sc_tick_sep * _span:
            _keep[i] = True
            _last = _e[i]
    ax_sc.set_xticklabels([rf"$10^{{{e:.0f}}}$" if k else ""
                           for e, k in zip(_e, _keep)])
    ax_sc.xaxis.set_minor_locator(plt.NullLocator())
    ax_sc.tick_params(axis="x", labelsize=sc_tick_fs)

    rank_ax = ax_sc.twiny()
    rank_ax.set_xscale("log")
    rank_ax.set_xlim(ax_sc.get_xlim())
    rank_ax.set_xticks(eta_sc)
    # Same mask, so the rank label sits exactly above the eta it maps to and the twin axis cannot crowd where the bottom one was thinned.
    rank_ax.set_xticklabels([f"{int(r)}" if k else ""
                             for r, k in zip(rank_sc, _keep)])
    rank_ax.set_xlabel("Basis size (global)")
    rank_ax.xaxis.set_minor_locator(plt.NullLocator())
    rank_ax.tick_params(axis="x", labelsize=sc_tick_fs)
    rank_ax.grid(False)

    # Failure rate captioned inline. The side is decided by the curve's local shape -- above a local maximum, below a local minimum -- so the caption lands in the empty region beside an extremum rather than on a neighbouring segment. Clamped away from the axis edges, where it would ride over the twin basis-size ticks.
    fail_handle = None
    sf = _first_by(sc_df, "rank", "failed")
    eta_of = dict(zip(rank_sc, eta_sc))
    y_of = dict(zip(rank_sc, y_sc))
    order = np.argsort(eta_sc)[::-1]              # left-to-right on screen
    seq = [rank_sc[i] for i in order]
    pos = {int(r): k for k, r in enumerate(seq)}
    ys = np.array([y_of[r] for r in seq], dtype=float)
    ylo, yhi = ax_sc.get_ylim()
    span = np.log10(yhi) - np.log10(ylo)

    any_fail = False
    for r in sorted(sf):
        f = sf[r]
        r = int(r)
        if f <= 0 or r not in pos:
            continue
        any_fail = True
        k = pos[r]
        left = ys[k - 1] if k > 0 else ys[k]
        right = ys[k + 1] if k < len(ys) - 1 else ys[k]
        # Put the caption on whichever side the curve is not occupying:
        #   peak (above both neighbours)  -> above
        #   valley (below both)           -> below
        #   descending into this point    -> below (segment is above-left)
        #   ascending into this point     -> above (segment is below-left)
        if ys[k] >= left and ys[k] >= right:
            above = True
        elif ys[k] <= left and ys[k] <= right:
            above = False
        else:
            above = left < ys[k]
        frac = (np.log10(ys[k]) - np.log10(ylo)) / span
        if above and frac > 0.88:                 # no room under the ticks
            above = False
        elif not above and frac < 0.12:           # no room at the bottom
            above = True
        ax_sc.annotate(_fail_label(f, n_eval),
                       xy=(eta_of[r], ys[k]),
                       xytext=(0, 13 if above else -17),
                       textcoords="offset points", ha="center",
                       va="bottom" if above else "top",
                       color=FAIL_COLOR, fontsize=18,
                       annotation_clip=False)
    if any_fail:
        fail_handle = Line2D([], [], color=FAIL_COLOR, ls="none",
                             marker=r"$\%$", ms=12, label="Failures")

    _h = [Line2D([], [], color=GLOBAL_COLOR, marker="o", ls="-", lw=2,
                 label="Global basis")]
    if fail_handle is not None:
        _h.append(fail_handle)
    ax_sc.legend(handles=_h, frameon=False, loc="lower left")

def _panel_ensemble(ax_cl, df, tols, dark, qoi, n_eval, twin_tol,
                    agg="l2-in-time", ylim=(5e-5, 4e-1), hybrid_df=None,
                    hybrid_marker=HYBRID_MARKER,
                    hybrid_legend_loc="lower left", corner_label=None,
                    star_sizes=None, star_legend_color=None,
                    twin_tick_fs=None):
    """Draw the clustered ROM error vs eta into ``ax_cl``.

    Returns the eta positions of its x ticks. The rank panel shares them, so it has to be drawn after this one.
    """
    cl_any_fail = False
    cl_y, cl_fail, cl_series = [], [], []
    for i_t, t in enumerate(tols):
        x, y = mean_series(df, t, qoi, agg)
        cl_y.append(np.asarray(y, dtype=float))
        cl_series.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        ax_cl.plot(x, y, marker="o", ls="-", lw=2, ms=6,
                   color=dark[t])
        # Uncaptioned, a combination that lost tracers reads as an ordinary point on the error curve, when its error is a mean over the survivors only.
        # Index on eta rather than the raw column, since `x` is already in eta.
        rows = df[np.isclose(df["tau_split"], t)]
        g = {float(np.asarray(np.asarray(k, dtype=float))): v
             for k, v in _first_by(rows, "eta", "n_solved").items()}
        fail = np.array([n_eval - float(g[v]) if v in g else 0.0
                         for v in x])
        cl_fail.append((t, i_t, x, y, fail))
    ax_cl.set_xscale("log")
    ax_cl.set_yscale("log")
    ax_cl.invert_xaxis()
    ax_cl.set_xlabel(r"Residual SVD energy $\eta$")
    ax_cl.grid(alpha=0.3, which="major")
    if ylim is not None:
        # `ylim` is a minimum extent, not a hard clip: a case whose curves descend past the window would otherwise be cut off mid-descent, and the last segment would read as a plunge out of the axes instead of convergence.
        lo, hi = ylim
        v = np.concatenate(cl_y) if cl_y else np.empty(0)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            lo, hi = min(lo, v.min() / 3.0), max(hi, v.max() * 3.0)
        ax_cl.set_ylim(lo, hi)

    # Annotated only once the limits are final: _annotate_failures decides which side of a point to caption from where the point sits in the axes, so asking before set_ylim answers for the wrong window. Curves that fail at the same combination also get one text-height of extra offset each, in legend order, so coincident captions do not overprint.
    _stack = {}
    for t, i_t, x, y, fail in cl_fail:            # legend order
        pads = np.zeros(np.size(x))
        for j, (xj, fj) in enumerate(zip(np.asarray(x), np.asarray(fail))):
            if np.isfinite(fj) and fj > 0:
                k = round(float(np.log10(xj)), 6)
                pads[j] = 19 * _stack.get(k, 0)
                _stack[k] = _stack.get(k, 0) + 1
        # The other curves' values at this curve's etas, so a caption can go in the gap facing whichever neighbour is further away.
        others = []
        for k, (xk, yk) in enumerate(cl_series):
            if k == i_t:
                continue
            lut = dict(zip(np.round(np.log10(xk), 6), yk))
            others.append([lut.get(round(float(np.log10(xj)), 6), np.nan)
                           for xj in np.asarray(x)])
        cl_any_fail |= _annotate_failures(ax_cl, x, y, fail, n_eval,
                                          dark[t], dy_pad=pads, others=others)

    # Hybrid overlay, before the ticks are pinned: the hybrid sweep runs a subset of the same etas and adds no new positions, so the twin basis-size axis still lines up with the pure-ROM curves.
    hybrid_tols = []
    if hybrid_df is not None and len(hybrid_df):
        for k, t in enumerate(tols):
            if not (hybrid_df["tau_split"] == t).any():
                continue
            hx, hy = mean_series(hybrid_df, t, qoi, agg)
            if not np.size(hx):
                continue
            # star_sizes nests them smallest-tolerance-in-front, so two tolerances landing on the same value read as concentric stars rather than one.
            nested = star_sizes is not None
            ax_cl.plot(hx, hy, ls="none", marker=hybrid_marker,
                       ms=star_sizes[k] if nested else 18, mfc=dark[t],
                       mec=dark[t] if nested else "white", mew=1.2,
                       zorder=9 - k if nested else 5)
            hybrid_tols.append(t)

    cl_x, cl_rank = rank_series(df, twin_tol)
    ax_cl.set_xticks(cl_x)
    ax_cl.set_xticklabels([rf"$10^{{{np.log10(v):.0f}}}$" for v in cl_x])
    ax_cl.xaxis.set_minor_locator(plt.NullLocator())

    cl_rank_ax = ax_cl.twiny()
    cl_rank_ax.set_xscale("log")
    cl_rank_ax.set_xlim(ax_cl.get_xlim())
    cl_rank_ax.set_xticks(cl_x)
    cl_rank_ax.set_xticklabels([f"{r:.1f}" for r in cl_rank])
    twin_color = dark.get(twin_tol, TOL_PAIRS[0][1])
    cl_rank_ax.set_xlabel("Mean basis size (clustered)", color=twin_color)
    cl_rank_ax.tick_params(axis="x", colors=twin_color)
    cl_rank_ax.xaxis.label.set_color(twin_color)
    cl_rank_ax.grid(False)
    if twin_tick_fs:
        cl_rank_ax.tick_params(axis="x", labelsize=twin_tick_fs)
    _hl = [Line2D([], [], color=dark[t], marker="o", ls="-", lw=2,
                  label=r"$\tau_\mathrm{split} = $" + rf"{t}")
           for t in tols]
    _leg = ax_cl.legend(handles=_hl, frameon=False, loc="best")
    if hybrid_tols:
        # Its own legend at a fixed corner: appended to the tolerance legend it would follow loc="best", which can place it over a curve. add_artist keeps the first legend, since ax.legend() otherwise replaces it. One neutral entry rather than one per tolerance -- the marker colours already say which curve each belongs to.
        ax_cl.add_artist(_leg)
        # star_legend_color fills the key with one tolerance's colour instead of neutral grey, and takes the white outline with it.
        key = (dict(color=star_legend_color, ms=16) if star_legend_color else
               dict(color="0.35", ms=14, mec="white", mew=1.2))
        ax_cl.legend(handles=[
            Line2D([], [], marker=hybrid_marker, ls="none",
                   label="Hybrid ROM/FOM", **key)],
            frameon=False, loc=hybrid_legend_loc)
    if corner_label:
        # Top left, mirroring the global panel's top-right caption.
        ax_cl.text(0.02, 0.97, corner_label, transform=ax_cl.transAxes,
                   ha="left", va="top", fontsize=20, color="0.25", zorder=6)
    return cl_x

def _panel_clusters(ax_bar, fig, df, tols, dark, light,
                    max_clusters, max_eta_exp=8, bar_headroom=1.9,
                    caption=True):
    """Draw the clusters built per (eta, tau_split) into ``ax_bar``.

    A partition diagnostic: it describes the fit, which is trained on the training dataset regardless of what the ROM is later evaluated over, so a figure carrying both datasets needs only one of these.
    """
    etas = np.unique(np.asarray(np.asarray(df["eta"], dtype=float), dtype=float))
    etas = np.sort(etas[etas >= 10.0 ** (-max_eta_exp)])   # ascending eta
    pos = np.arange(len(etas))
    width = 0.8 / max(len(tols), 1)

    def combo_val(t, col):
        lut = {float(np.asarray(np.asarray(k, dtype=float))): v
               for k, v in _first_by(where(df, **{"tau_split": t}), "eta", col).items()}
        return np.array([float(lut[e]) if e in lut else np.nan for e in etas])

    for i, t in enumerate(tols):
        off = (i - (len(tols) - 1) / 2) * width
        n = combo_val(t, "n_clusters")
        u = np.nan_to_num(combo_val(t, "n_unsplittable"))
        ax_bar.bar(pos + off, n, width=width, facecolor="none",
                   edgecolor=dark[t], lw=1.8, zorder=3)
        ax_bar.bar(pos + off, n - u, width=width,
                   facecolor=light[t], edgecolor="none", zorder=2)

    ax_bar.set_xticks(pos)
    ax_bar.set_xticklabels([rf"$10^{{{np.log10(v):.0f}}}$" for v in etas])
    ax_bar.set_xlabel(r"Residual SVD energy $\eta$")
    ax_bar.set_ylabel("Clusters")
    ax_bar.grid(alpha=0.3, axis="y", which="major")
    ax_bar.set_axisbelow(True)
    ax_bar.axhline(max_clusters, color="0.35", ls=":", lw=1.5)
    ax_bar.set_ylim(0, max_clusters * bar_headroom)
    if caption:
        _tr = offset_copy(ax_bar.get_yaxis_transform(), fig=fig, y=4,
                          units="points")
        ax_bar.text(0.99, max_clusters, f"max clusters = {max_clusters}",
                    transform=_tr, va="bottom", ha="right", color="0.35",
                    fontsize=18)
    ax_bar.legend(
        handles=[Patch(facecolor=light[t],
                       edgecolor=dark[t], lw=1.8,
                       label=r"$\tau_\mathrm{split} = $" + rf"{t}")
                 for t in tols]
        + [Patch(facecolor="none", edgecolor="0.35", lw=1.8,
                 label="unsplittable")],
        loc="upper left", ncol=2, frameon=False)

def _panel_rank(ax_rank, df, tols, dark, cl_x):
    """Draw the mean basis size vs eta into ``ax_rank``.

    ``cl_x`` comes from :func:`_panel_ensemble`, whose x ticks this shares.
    """
    for t in tols:
        x, y = rank_series(df, t)
        ax_rank.plot(x, y, marker="o", ls="--", lw=2, ms=6,
                     color=dark[t])
    ax_rank.set_xscale("log")
    ax_rank.invert_xaxis()
    ax_rank.set_xlabel(r"Residual SVD energy $\eta$")
    ax_rank.set_ylabel("Mean basis size")
    ax_rank.grid(alpha=0.3, which="major")
    ax_rank.set_xticks(cl_x)
    ax_rank.set_xticklabels([rf"$10^{{{np.log10(v):.0f}}}$" for v in cl_x])
    ax_rank.xaxis.set_minor_locator(plt.NullLocator())
    ax_rank.legend(handles=[
        Line2D([], [], color=dark[t], marker="o", ls="--", lw=2,
               label=r"$\tau_\mathrm{split} = $" + rf"{t}")
        for t in tols], frameon=False, loc="best")

def sweep_panel(df, sc_df, qoi, n_eval, max_clusters,
                 show_tols=(0.01, 0.10, 0.40), twin_tol=0.10,
                 agg="l2-in-time", ylim=(5e-5, 4e-1), max_eta_exp=8,
                 bar_headroom=1.9, sc_tick_fs=18, sc_tick_sep=0.16,
                 figsize=(15, 10), savepath=None, split="test",
                 hybrid_df=None, hybrid_marker=HYBRID_MARKER,
                 hybrid_legend_loc="lower left"):
    """The four-panel sweep summary.

    Top left: global ROM error vs eta, with basis size on a twin axis and the failure rate annotated where solves failed.
    Top right: clustered ROM error vs eta, one curve per tau_split.
    Bottom left: clusters built per (eta, tau_split), with the unsplittable fraction shown as an outline against a fill, and the ``max_clusters`` ceiling marked.
    Bottom right: mean basis size vs eta.

    ``split`` sets the error axis wording only; ``n_eval`` must match the dataset ``df`` and ``sc_df`` were evaluated over. For training and testing side by side, see :func:`sweep_panel_splits`.
    """
    use_paper_style()
    qoi = list(qoi)
    tols = [t for t in sorted(np.unique(df["tau_split"])) if t in set(show_tols)]
    if not tols:
        raise ValueError(f"none of show_tols={show_tols} present in the sweep "
                         f"(have {sorted(np.unique(df['tau_split']))})")
    pal = tol_colors(tols)
    dark = {t: pal[t][1] for t in tols}
    light = {t: pal[t][0] for t in tols}

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_sc, ax_cl = axes[0]
    ax_bar, ax_rank = axes[1]

    # ---------------- global ROM ----------------
    _panel_global(ax_sc, sc_df, qoi, n_eval, agg=agg,
                  sc_tick_fs=sc_tick_fs, sc_tick_sep=sc_tick_sep, split=split)

    # ---------------- clustered ROM ----------------
    cl_x = _panel_ensemble(ax_cl, df, tols, dark, qoi, n_eval, twin_tol,
                           agg=agg, ylim=ylim, hybrid_df=hybrid_df,
                           hybrid_marker=hybrid_marker,
                           hybrid_legend_loc=hybrid_legend_loc)

    # ---------------- clusters built ----------------
    _panel_clusters(ax_bar, fig, df, tols, dark, light,
                    max_clusters, max_eta_exp=max_eta_exp,
                    bar_headroom=bar_headroom)

    # ---------------- mean basis size ----------------
    _panel_rank(ax_rank, df, tols, dark, cl_x)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92,
                        wspace=0.20, hspace=0.30)
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"saved figure -> {savepath}")
    return fig, axes

def sweep_panel_splits(df_train, sc_train, df_test, sc_test, qoi,
                        n_eval_train, n_eval_test, max_clusters,
                        show_tols=(0.01, 0.10, 0.40),
                        twin_tol=0.10, agg="l2-in-time", ylim=(5e-5, 4e-1),
                        max_eta_exp=8, bar_headroom=1.9, sc_tick_fs=18,
                        sc_tick_sep=0.16, figsize=(15, 15), savepath=None,
                        hybrid_train=None, hybrid_test=None,
                        hybrid_marker=HYBRID_MARKER,
                        hybrid_legend_loc="lower left", share_ylim=True,
                        caption_headroom=3.0, star_sizes=None,
                        star_legend_color=None, max_clusters_caption=True,
                        tick_fs=20, twin_tick_fs=18, row_gap=0.0):
    """The sweep summary with a training row above the test row.

    Rows 1 and 2 are the global and clustered error panels, drawn once per dataset; row 3 is the partition diagnostics. Those two describe the fit, which is trained on the training dataset whatever the ROM is later evaluated over, so they are drawn once from the testing frame -- the training frame carries identical ``n_clusters``, ``n_unsplittable`` and ``mean_rank``.

    Each row's caption names its split: top right on the global panel, top left on the clustered one, where each has empty space.

    ``n_eval_train`` and ``n_eval_test`` are separate on purpose. They are the denominators of the failure-rate captions, and passing one value for both scales an entire row's percentages silently, since the figure still draws.

    ``share_ylim`` puts both global panels on one error axis and both clustered panels on another, so a row-to-row comparison is not reading two different scales. Turn it off to let each row size to its own data.

    ``row_gap`` is additional vertical space, as a fraction of the figure height, between the training and testing rows. It is taken in equal parts from the height of the three rows, so the figure margins and the other row spacing are unchanged.
    """
    use_paper_style()
    qoi = list(qoi)
    tols = [t for t in sorted(np.unique(df_test["tau_split"])) if t in set(show_tols)]
    if not tols:
        raise ValueError(f"none of show_tols={show_tols} present in the test "
                         f"sweep (have {sorted(np.unique(df_test['tau_split']))})")
    missing = [t for t in tols if t not in set(df_train["tau_split"])]
    if missing:
        raise ValueError(
            f"train sweep is missing tau_split {missing}, which the test sweep "
            "has. The two rows would show different tolerance sets under the "
            "same legend.")
    pal = tol_colors(tols)
    dark = {t: pal[t][1] for t in tols}
    light = {t: pal[t][0] for t in tols}

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    (ax_sc_tr, ax_cl_tr), (ax_sc_te, ax_cl_te), (ax_bar, ax_rank) = axes

    rows = (("train", ax_sc_tr, ax_cl_tr, sc_train, df_train, n_eval_train,
             hybrid_train),
            ("test", ax_sc_te, ax_cl_te, sc_test, df_test, n_eval_test,
             hybrid_test))

    cl_x = None
    for split, ax_sc, ax_cl, sc_d, d, n_ev, hyb in rows:
        cap = SPLIT_CAPTION[split]
        _panel_global(ax_sc, sc_d, qoi, n_ev, agg=agg,
                      sc_tick_fs=sc_tick_fs, sc_tick_sep=sc_tick_sep,
                      split=split, corner_label=cap)
        # Every row's ticks come from its own frame, but the rank panel follows the test row, since it sits under it and shares its eta axis.
        x = _panel_ensemble(ax_cl, d, tols, dark, qoi, n_ev, twin_tol,
                            agg=agg, ylim=ylim, hybrid_df=hyb,
                            hybrid_marker=hybrid_marker,
                            hybrid_legend_loc=hybrid_legend_loc,
                            corner_label=cap, star_sizes=star_sizes,
                            star_legend_color=star_legend_color,
                            twin_tick_fs=twin_tick_fs)
        if split == "test":
            cl_x = x

    if share_ylim:
        # Union of each pair, applied after both are drawn: the failure captions are offset from their data points, so they follow the rescale.
        for pair, pad in (((ax_sc_tr, ax_sc_te), 1.0),
                          ((ax_cl_tr, ax_cl_te), caption_headroom)):
            a, b = pair
            lo = min(a.get_ylim()[0], b.get_ylim()[0])
            # The clustered panels caption top left, which is where their leftmost (largest-eta, highest-error) point and its failure label sit. Headroom moves the curve down rather than moving the caption somewhere the axes have no room for.
            hi = max(a.get_ylim()[1], b.get_ylim()[1]) * pad
            a.set_ylim(lo, hi)
            b.set_ylim(lo, hi)

    _panel_clusters(ax_bar, fig, df_test, tols, dark, light,
                    max_clusters, max_eta_exp=max_eta_exp,
                    bar_headroom=bar_headroom, caption=max_clusters_caption)
    _panel_rank(ax_rank, df_test, tols, dark, cl_x)

    # Last, so it overrides the ticks each panel set for itself.
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelsize=tick_fs)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.94,
                        wspace=0.20, hspace=0.45)
    if row_gap:
        _widen_first_gap(fig, axes, row_gap)
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"saved figure -> {savepath}")
    return fig, axes


def _widen_first_gap(fig, axes, gap):
    """Add ``gap`` (figure fraction) between the first two rows of ``axes``.

    Each row loses ``gap / n_rows`` of height; the top of the first row, the bottom of the last row and every other inter-row gap are preserved. Twin axes share their host's position, so every axes in the figure whose vertical extent matches a row is moved with it.
    """
    n = axes.shape[0]
    rows = [axes[r, 0].get_position() for r in range(n)]
    h = rows[0].height - gap / n
    if h <= 0:
        raise ValueError(f"row_gap={gap} leaves no height for the panels")
    # Stack from the bottom up: the last row keeps its bottom edge, each row above keeps its original gap to the one below, and the first gap is widened by `gap`.
    y0 = [0.0] * n
    y0[-1] = rows[-1].y0
    for r in range(n - 2, -1, -1):
        orig = rows[r].y0 - rows[r + 1].y1
        y0[r] = y0[r + 1] + h + orig + (gap if r == 0 else 0.0)
    for ax in fig.axes:
        b = ax.get_position()
        for r, row in enumerate(rows):
            if np.isclose(b.y0, row.y0) and np.isclose(b.height, row.height):
                ax.set_position([b.x0, y0[r], b.width, h])
                break


PANEL_KEYS = {"global_train": (0, 0), "clustered_train": (0, 1),
              "global_test": (1, 0), "clustered_test": (1, 1)}


def nudge_failure_labels(axes, moves):
    """Re-offset a sweep panel's failure-rate captions by their label text.

    ``moves`` maps a key of :data:`PANEL_KEYS` to ``{label: (dx, dy)}`` or ``{label: (dx, dy, va)}``, in offset points from the annotated point. Both rows carry these captions and their percentages differ, so the panel has to be named: the same string can appear in more than one. A label the panel does not carry is skipped, since the percentages depend on the run.
    """
    moved = []
    for key, labels in moves.items():
        if key not in PANEL_KEYS:
            raise KeyError(f"{key!r} is not a panel; use {sorted(PANEL_KEYS)}")
        i, j = PANEL_KEYS[key]
        ax = axes[i][j]
        for text in ax.texts:
            spec = labels.get(text.get_text())
            if spec is None:
                continue
            text.set_position(tuple(spec[:2]))
            if len(spec) > 2:
                text.set_va(spec[2])
            moved.append(f"{key}:{text.get_text()}")
    absent = {f"{k}:{lab}" for k, labs in moves.items() for lab in labs} - set(moved)
    if absent:
        print(f"  no failure caption to move for {sorted(absent)}")
    return moved


# ---------------------------------------------------------------------------
# Abundance panel
# ---------------------------------------------------------------------------


def _logfmt(y, _pos):
    """Power-of-ten tick label for a symlog axis."""
    if y == 0:
        return "0"
    e = int(np.round(np.log10(abs(y))))
    return f"$-10^{{{e}}}$" if y < 0 else f"$10^{{{e}}}$"


def _dex_error(d, peak, floor=1e-12, rom_key="y_rom_pos", t_key="tyr_rom_pos"):
    """RMS |log10(ROM) - log10(FOM)| per species: off by how many decades.

    Unlike a relative L2 it weights every snapshot equally, so a brief startup transient cannot dominate a species whose tail is tracked perfectly. Both trajectories are clipped at a floor set by each species' own FOM peak, so a ROM that reaches exactly zero scores a bounded value rather than infinity.
    """
    y_fom = np.asarray(d["y_fom"])
    t_fom = np.asarray(d["tyr"])
    y_rom = np.asarray(d[rom_key])
    t_rom = np.asarray(d[t_key])
    if t_rom.shape[0] != t_fom.shape[0]:
        # A solve that died early returns a short axis; align it as evaluate does, so every species is scored over the same span.
        y_rom = np.stack([np.interp(t_fom, t_rom, y_rom[j])
                          for j in range(y_rom.shape[0])])
    flo = (np.where(peak > 0, peak, 1.0) * floor)[:, None]
    dex = (np.log10(np.clip(y_rom, flo, None))
           - np.log10(np.clip(y_fom, flo, None)))
    return np.sqrt(np.mean(dex ** 2, axis=1))


def abundance_panel(d, d_global, species, qoi_indices, knots_only=True,
                    n_markers=30, n_ticks=6, figsize=(20, 13), savepath=None,
                    dex_floor=1e-12, ensemble=None, cluster_dims=None):
    """Per-tracer abundance figure: every species along one trajectory.

    Four panels: the QoIs (top left), then the non-QoI species split into three equal groups ordered by the clustered ROM's dex error -- smallest (top right), intermediate (bottom left), largest (bottom right). Each panel overlays, per species, the FOM (markers only, so the curves stay readable underneath), the clustered ROM (solid) and the global ROM (dashed), with shaded bands marking the cluster route.

    ``d`` comes from :func:`diagnostics.tracer_data` and ``d_global`` from the same call on a single-cluster ensemble at a fixed basis size.

    Pass ``ensemble`` (the ``EnsembleROM`` that produced ``d``) or an explicit ``cluster_dims`` mapping to get the shaded route summarised numerically in the returned info dict under ``route``: how many clusters the ROM routes through and their route-weighted mean basis size, via :func:`diagnostics.route_stats`. One band in the figure is one entry in that count.

    Each panel gets its own symlog scale: abundances within a group still span several decades, and a shared scale would flatten the small ones to a line.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator, SymmetricalLogLocator
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Rectangle

    use_paper_style()
    plt.rcParams.update({"font.size": 26, "axes.titlesize": 26,
                         "axes.labelsize": 26, "xtick.labelsize": 26,
                         "ytick.labelsize": 26, "legend.fontsize": 22})

    peak = np.asarray(d["peak"])
    qoi = list(qoi_indices)
    non_qoi = [i for i in range(len(species)) if i not in set(qoi)]

    rom_key = "y_rom_pos" if "y_rom_pos" in d else "y_rom"
    rom_tkey = "tyr_rom_pos" if "y_rom_pos" in d else "tyr_rom"
    dex = _dex_error(d, peak, dex_floor, rom_key, rom_tkey)

    ranked = sorted(non_qoi, key=lambda j: -dex[j])
    n = len(ranked)
    hi, mid, lo = ranked[:n // 3], ranked[n // 3:2 * n // 3], ranked[2 * n // 3:]

    # FOM is markers only -- no line -- so the ROM curves stay readable under it.
    VARIANTS = [("y_fom", "tyr", "none", 0.0, "FOM"),
                (rom_key, rom_tkey, "-", 1.8, "ROM")]
    MARKERS = {"y_fom": "o"}

    # Shading follows the clustered ROM's own route, so one band is one cluster the ROM actually integrated in.
    rom_seg = d.get("rom_pos", d["rom"])
    cc = {c: plt.cm.tab20(i % 20)
          for i, c in enumerate(np.unique(rom_seg["cp"]))}
    dt_step = d.get("dt_hydro_yr")
    colors = {j: plt.cm.tab20(j % 20) for j in range(len(species))}

    def xy(t, y):
        t = np.asarray(t)
        if knots_only and dt_step:
            m = D.knot_mask(t, dt_step)          # mask wants t in yr
            t, y = t[m], np.asarray(y)[m]
        return t / 1e3, y                          # -> kyr

    seg_k = {**rom_seg, "te": np.asarray(rom_seg["te"]) / 1e3}

    PANELS = [(qoi, "QoIs"), (lo, "Non-QoIs, smallest error"),
              (mid, "Non-QoIs, intermediate error"),
              (hi, "Non-QoIs, largest error")]

    fig, axes2d = plt.subplots(2, 2, figsize=figsize, sharex=True)
    for ax, (group, title) in zip(axes2d.flat, PANELS):
        if not group:
            ax.set_visible(False)
            continue
        for j in group:
            for ykey, tkey, ls, lw, _lab in VARIANTS:
                t_, y_ = xy(d[tkey], d[ykey][j])
                mk = MARKERS.get(ykey)
                ax.plot(t_, y_, ls=ls, lw=lw, color=colors[j], alpha=0.85,
                        zorder=5, marker=mk, ms=6 if mk else 0, mfc="none",
                        mew=1.4, markevery=max(1, len(t_) // n_markers))
            if d_global is not None:
                gk = "y_rom_pos" if "y_rom_pos" in d_global else "y_rom"
                gt = "tyr_rom_pos" if "y_rom_pos" in d_global else "tyr_rom"
                t_, y_ = xy(d_global[gt], d_global[gk][j])
                ax.plot(t_, y_, ls="--", lw=1.6, color=colors[j], alpha=0.85,
                        zorder=4)

        peaks = [peak[j] for j in group if peak[j] > 0]
        lt = max(min(peaks) * 1e-3, 1e-30) if peaks else 1e-30
        ax.set_yscale("symlog", linthresh=lt)
        ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
        loc = SymmetricalLogLocator(linthresh=lt, base=10)
        loc.numticks = n_ticks
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, pos, c=3 * lt: "" if y != 0 and abs(y) < c
            else _logfmt(y, pos)))
        for k in range(len(seg_k["seg"])):
            ax.axvspan(seg_k["te"][k], seg_k["te"][k + 1],
                       color=cc[seg_k["seg"][k]], alpha=0.12, lw=0, zorder=0)
        ax.set_title(title)

    # One ylabel per row on the left column. Each panel keeps its own symlog scale, so only the text is shared; this is not sharey.
    for row in axes2d:
        if row[0].get_visible():
            row[0].set_ylabel("Fractional abundance")
    for ax in axes2d[-1]:
        if ax.get_visible():
            ax.set_xlabel("Time [kyr]")

    # Each panel carries a species legend underneath, wrapped at 4 columns, so the gap between rows has to grow with the widest group.
    rows_needed = max(int(np.ceil(len(g) / 4)) for g, _ in PANELS if g)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.10,
                        wspace=0.20, hspace=0.38 + 0.22 * (rows_needed - 1))

    class _HandlerBands(HandlerBase):
        """Legend key that shows the shading palette itself: one sub-rectangle per cluster, drawn edge to edge so it reads as the band strip."""
        def __init__(self, band_colors, alpha=0.45, **kw):
            super().__init__(**kw)
            self.band_colors = band_colors
            self.alpha = alpha        # stronger than the panels' band tint, or the swatch is invisible at this size

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            m = max(len(self.band_colors), 1)
            w = width / m
            return [Rectangle((xdescent + i * w, -ydescent), w, height,
                              facecolor=c, alpha=self.alpha, lw=0,
                              transform=trans)
                    for i, c in enumerate(self.band_colors)]

    # First-visit order, so the key runs left to right in the same order the bands do along x. Cluster-id order is unrelated to the route and would put the palette in the wrong sequence.
    _seen = list(dict.fromkeys(np.asarray(seg_k["seg"]).tolist()))
    band_colors = [cc[c] for c in _seen if c in cc]
    cluster_key = Patch(label="Clusters")      # proxy; the handler draws it
    variants = VARIANTS + ([("g", "g", "--", 1.6, "Global basis")]
                           if d_global is not None else [])
    fig.legend(handles=[Line2D([], [], color="0.25", ls=ls, lw=lw, label=lab,
                               marker=MARKERS.get(yk), ms=6, mfc="none", mew=1.4)
                        for yk, _t, ls, lw, lab in variants] + [cluster_key],
               handler_map={cluster_key: _HandlerBands(band_colors)},
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005),
               frameon=True)

    # One species legend per panel, below the axes. The offset clears the tick labels and xlabel on the bottom row, which the top row does not have.
    for r, row in enumerate(axes2d):
        for ax, (group, _t) in zip(row, PANELS[2 * r:2 * r + 2]):
            if not group or not ax.get_visible():
                continue
            bb = ax.get_position()
            off = (0.09 if r == len(axes2d) - 1 else 0.02) + 0.012 * (rows_needed - 1)
            fig.legend(handles=[Line2D([], [], color=colors[j], lw=3,
                                       label=species[j]) for j in group],
                       loc="upper center",
                       bbox_to_anchor=(bb.x0 + bb.width / 2, bb.y0 - off),
                       bbox_transform=fig.transFigure,
                       ncol=min(len(group), 4), frameon=False)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig, axes2d, dict(dex_error=dex, groups=dict(hi=hi, mid=mid, lo=lo),
                             route=D.route_stats(d, ensemble=ensemble,
                                                 dims=cluster_dims))


# ---------------------------------------------------------------------------
# Hybrid figures
# ---------------------------------------------------------------------------
# Same alpha on both route registers, or a cluster reads differently above and below the axis. Stronger than abundance_panel's tint: the lines here come from a different palette than the bands, so no band is a pale copy of a curve.
BAND_ALPHA = 0.30
# FOM windows are the point of the figure, so they keep a stronger tint than the cluster bands rather than fading with them.
FOM_ALPHA_SCALE = 1.8
# Same map and keying as abundance_panel, so a cluster band looks the same in both figures. Qualitative maps only -- the index cycles, it does not ramp.
BAND_CMAP = plt.cm.tab20
# Okabe-Ito, colour-blind safe. Not BAND_CMAP: a QoI drawn from the band map would sit on a pale copy of its own colour. Okabe-Ito's yellow is left out, since it disappears against white at line weight.
LINE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
               "#56B4E9", "#000000", "#8B4513")
# The line-style keys (hybrid solid, ROM dashed) are drawn in one colour rather than grey, which would read as a further species. This is LINE_COLORS' third entry.
STYLE_COLOR = "#009E73"
ROM_CAPTION = "Clusters visited by the ROM"
HYBRID_CAPTION = "Clusters visited by the hybrid"


def hybrid_comparison_panel(result, t_ref, y_ref, *, qoi_indices, species,
                            rom=None, time_scale=1.0,
                            time_label="Time [s]",
                            tick_stride=3, band_alpha=BAND_ALPHA,
                            fom_color="k", band_cmap=None, fom_alpha=None,
                            line_colors=None, knots_only=True,
                            style_color=STYLE_COLOR,
                            rom_caption=ROM_CAPTION,
                            hybrid_caption=HYBRID_CAPTION,
                            strip_ratio=0.09, strip_gap=0.55,
                            caption_fontsize=None, legend_outside=True,
                            legend_ncol=None, ref_stride=None,
                            figsize=(14, 6), fontsize=16, ax=None,
                            strip_ax=None):
    """Hybrid vs pure ROM vs reference, over per-interval cluster route bands.

    QoI abundances only: open circles the reference, solid the hybrid, dashed the ROM.

    The hybrid's route shades the data area, with FOM intervals in ``fom_color``. The ROM's route is a detached strip under the panel, on its own axes, so a band under the data always belongs to the hybrid and one below the panel always to the ROM. Both registers draw from one cluster-to-colour map.

    ``time_scale`` divides every time; pass ``NSEC * 1e3`` with ``time_label="Time [kyr]"``. ``knots_only`` samples every series at the hydro knots, and ``ref_stride`` thins the reference markers further.

    Returns ``(fig, ax)`` with ``ax`` the data panel; the strip is ``fig.axes[1]`` when one was created.
    """
    qi = list(qoi_indices)
    names = [str(species[j]) for j in qi]
    # Keyed by position in `qoi_indices`, not by species index: the palette is short, and one QoI list gives the same species the same colour everywhere.
    lc = tuple(line_colors or LINE_COLORS)
    pal = {j: lc[i % len(lc)] for i, j in enumerate(qi)}

    y_ref = np.asarray(y_ref, dtype=float)
    t_ref = np.asarray(t_ref, dtype=float) / time_scale
    tk_h = np.asarray(result.t, dtype=float) / time_scale
    dt = result.dt_hydro / time_scale
    n_steps = result.n_steps

    want_strip = rom is not None and strip_ratio > 0
    if ax is not None:
        fig = ax.figure
    elif want_strip:
        # Not sharex: the strip clears its own ticks, and on a shared axis that call also strips the data panel's tick labels. Both axes get the same explicit xlim instead, which keeps them in register. `strip_gap` has to clear the data panel's tick labels and its xlabel, which both live in the gap.
        fig, (ax, strip_ax) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw=dict(height_ratios=[1.0, strip_ratio],
                             hspace=strip_gap))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Every series at the hydro knots, not at the intra-step sub-points: the sub-points carry no hydro information and multiply the vertex count.
    m_ref = _knot_sample(t_ref, dt) if knots_only else slice(None)
    m_h = _knot_sample(tk_h, dt) if knots_only else slice(None)
    t_r, y_r = t_ref[m_ref], y_ref[:, m_ref]

    stride = ref_stride or max(1, t_r.shape[0] // 40)
    handles = []
    for name, j in zip(names, qi):
        h, = ax.plot(t_r[::stride], y_r[j, ::stride], "o", color=pal[j],
                     ms=6, mfc="none", mew=1.6, ls="none", label=name, zorder=4)
        handles.append(h)
        if result.y.size:
            ax.plot(tk_h[m_h], result.y[j][m_h], "-", color=pal[j], lw=1.8,
                    zorder=4)
        if rom is not None and np.asarray(rom.y).size:
            t_rom = np.asarray(rom.t) / time_scale
            m_r = _knot_sample(t_rom, dt) if knots_only else slice(None)
            ax.plot(t_rom[m_r], np.asarray(rom.y)[j][m_r], "--", color=pal[j],
                    lw=1.1, alpha=0.8, zorder=4)

    _log_axis(ax, y_ref[qi], tick_stride=tick_stride)
    ax.set_ylabel("Fractional abundance", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    # Flush against t = 0: the default margin leaves a white gutter before the first band, which reads as an interval the solver did not route.
    ax.set_xlim(0.0, n_steps * dt)
    ax.margins(x=0)

    a_fom = (min(1.0, band_alpha * FOM_ALPHA_SCALE) if fom_alpha is None
             else fom_alpha)
    cmap = _band_colors(result.path, None if rom is None else rom.path,
                        band_cmap, fom_color)
    _spans(ax, result.path, n_steps, dt, cmap, band_alpha, a_fom)

    # The time axis stays on the data panel, above the bands: the strip is a legend for the route, not a second plot, so it carries no axis of its own.
    ax.set_xlabel(time_label, fontsize=fontsize)
    if strip_ax is not None and rom is not None:
        _route_strip(strip_ax, rom.path, n_steps, dt, cmap,
                     band_alpha=band_alpha, a_fom=a_fom, caption=rom_caption,
                     fontsize=caption_fontsize or fontsize)

    legend = [Line2D([], [], color=style_color, lw=1.8, ls="-", label="hybrid")]
    if rom is not None:
        legend.append(Line2D([], [], color=style_color, lw=1.1, ls="--",
                             label="ROM"))
    if result.fom_step.any():
        # Composited over white at the band's own alpha, or the key is a solid swatch against a tinted band.
        legend.append(Line2D([], [], color=_over_white(fom_color, a_fom), lw=8,
                             label="FOM window"))
    ax.grid(alpha=0.3)
    # Before the legends, not after: subplots_adjust moves the axes, and the axes-anchored legend moves with it while a figure-anchored one does not.
    if strip_ax is None:
        fig.tight_layout()
    else:
        # tight_layout reports the strip as incompatible and warns, so the margins are set directly; top leaves room for the two legend rows.
        fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.16)

    # The hybrid's own route is the shading, not a curve, so it gets a row to itself: one key showing every cluster colour it visited, in the tint the panel draws.
    band_key = swatch = None
    if hybrid_caption:
        # First-visit order, so the key runs left to right in the same order the bands do along x. Cluster-id order is unrelated to the route.
        hyb = list(dict.fromkeys(
            c for c in (int(v) for v in np.asarray(result.path))
            if c != int(FOM)))
        swatch = [_over_white(cmap[c], band_alpha) for c in hyb] or ["0.85"]
        band_key = Patch(label=hybrid_caption)

    entries = handles + legend
    hmap = {} if band_key is None else {band_key: _HandlerBands(swatch)}
    if legend_outside:
        # Above the axes, not over the data: the abundances run to the top of the panel, so an in-axes legend sits on the first decade of every QoI.
        main_leg = ax.legend(handles=entries, frameon=False, fontsize=fontsize,
                             ncol=legend_ncol or len(entries),
                             loc="lower center", bbox_to_anchor=(0.5, 1.003),
                             borderaxespad=0.0, columnspacing=1.4,
                             handletextpad=0.5)
        if band_key is not None:
            # The second row is placed off the first row's measured extent, since a fixed offset only holds at one font size.
            fig.canvas.draw()
            top = main_leg.get_window_extent().transformed(
                fig.transFigure.inverted()).y1
            band_leg = Legend(ax, [band_key], [hybrid_caption],
                              handler_map=hmap, frameon=False,
                              fontsize=fontsize, loc="lower center",
                              bbox_to_anchor=(0.5, top + 0.008),
                              bbox_transform=fig.transFigure, handlelength=3.0,
                              borderaxespad=0.0)
            ax.add_artist(band_leg)   # ax.legend() again would replace main_leg
            # add_artist clips to the axes patch, and this row sits entirely above the axes, so without this it is drawn and then clipped away.
            band_leg.set_clip_on(False)
            fig.canvas.draw()
            over = band_leg.get_window_extent().transformed(
                fig.transFigure.inverted()).y1 - 0.995
            if over > 0:              # would sit off the top of the figure
                band_leg.set_bbox_to_anchor((0.5, top + 0.008 - over),
                                            transform=fig.transFigure)
    else:
        ax.legend(handles=([band_key] if band_key else []) + entries,
                  handler_map=hmap, frameon=False, fontsize=fontsize,
                  ncol=legend_ncol or 4, loc="upper right")
    return fig, ax


class _HandlerBands(HandlerBase):
    """One legend key showing every cluster colour, edge to edge."""

    def __init__(self, band_colors, alpha=None, **kw):
        super().__init__(**kw)
        self.band_colors = list(band_colors)
        self.alpha = alpha       # None keeps the pre-composited colours opaque

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        n = max(len(self.band_colors), 1)
        w = width / n
        return [Rectangle((xdescent + i * w, -ydescent), w, height,
                          facecolor=c, alpha=self.alpha, lw=0, transform=trans)
                for i, c in enumerate(self.band_colors)]


def _over_white(color, alpha):
    """``color`` at ``alpha`` composited over white, as an opaque RGB.

    A legend swatch drawn at the band's alpha goes translucent over the legend's own background; this matches what the band looks like on the panel instead.
    """
    r, g, b = to_rgb(color)
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def _knot_sample(t, dt, rtol=1e-6):
    """Mask of the entries of ``t`` landing on a hydro-step boundary.

    Same rule as :func:`diagnostics.knot_mask`, in whatever unit ``t`` and ``dt`` share. Falls back to everything if the two disagree: an empty axis is worse than a dense one.
    """
    t = np.asarray(t, dtype=float)
    if not t.size or not np.isfinite(dt) or dt <= 0:
        return slice(None)
    frac = (t / dt) % 1.0
    m = (frac < rtol) | (frac > 1 - rtol)
    return m if m.sum() >= 2 else slice(None)


def _log_axis(ax, y_qoi, *, tick_stride):
    """Plain log y scale, keyed to the reference.

    Set from the reference alone, so a ROM excursion cannot decide the limits. A negative or zero value is off the panel: the trade for dropping symlog, whose linear window and zero tick take a visible slice of the axis to describe states the QoIs almost never reach.
    """
    pos = y_qoi[y_qoi > 0]
    if not pos.size:
        return
    lo = 10.0 ** np.floor(np.log10(pos.min()))
    hi = float(pos.max())
    ax.set_yscale("log")
    ax.set_ylim(lo, hi * 5)

    e_lo, e_hi = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    ax.set_yticks([10.0 ** e for e in range(e_lo, e_hi + 1, tick_stride)])
    ax.yaxis.set_minor_locator(plt.NullLocator())


def _band_colors(path, rom_path, band_cmap, fom_color):
    """One cluster -> colour map shared by the data panel and the ROM strip.

    Enumerated over the clusters on this tracer's route, as abundance_panel keys its own bands; FOM sits outside the cycle. Built once for both registers, or the same cluster would take a different colour above and below.
    """
    path = np.asarray(path, dtype=int)
    ids = (path if rom_path is None
           else np.r_[path, np.asarray(rom_path, dtype=int)])
    cm = band_cmap or BAND_CMAP
    n_col = int(getattr(cm, "N", 20))
    clusters = [int(c) for c in np.unique(ids) if int(c) != FOM]
    cmap = {c: cm(i % n_col) for i, c in enumerate(clusters)}
    cmap[int(FOM)] = fom_color
    return cmap


def _spans(ax, path, n_steps, dt, cmap, band_alpha, a_fom, ymin=0.0, ymax=1.0):
    """One ``axvspan`` per hydro interval, in axes-fraction height."""
    path = np.asarray(path, dtype=int)
    for k in range(min(n_steps, path.size)):
        c = int(path[k])
        ax.axvspan(k * dt, (k + 1) * dt, ymin=ymin, ymax=ymax, color=cmap[c],
                   alpha=(a_fom if c == int(FOM) else band_alpha),
                   lw=0, zorder=0)


def _route_strip(ax, rom_path, n_steps, dt, cmap, *, band_alpha, a_fom,
                 caption=None, fontsize=12, strip_alpha=None):
    """The ROM's route as a bare band strip -- no spines, ticks or scale.

    It is a key for the route, not a second plot, so every axis artist is removed and only the bands remain. Drawn at the data panel's own ``band_alpha`` by default, so one cluster looks identical in both registers and in the legend key.
    """
    a = band_alpha if strip_alpha is None else strip_alpha
    _spans(ax, rom_path, n_steps, dt, cmap, a, a_fom)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, n_steps * dt)
    ax.margins(x=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_visible(False)
    if caption:
        # Below the bands, not on them: centred inside the strip, the text crosses several cluster colours.
        ax.text(0.5, -0.45, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=fontsize, color="0.2", zorder=5)


# ---------------------------------------------------------------------------
# Trajectory and per-parameter error
# ---------------------------------------------------------------------------
#: One band per contiguous run of a cluster, at the alpha abundance_panel uses.
TRAJ_BAND_ALPHA = 0.12
#: Reference solid, clustered ROM dashed, global baseline dotted.
TRAJ_COLORS = {"reference": "0.15", "clustered": "#009f47", "global": "#ff0800"}
TRAJ_STYLE = {"reference": ("-", 2.2), "clustered": ("--", 1.8),
              "global": (":", 1.8)}


def _runs(route):
    """Contiguous runs of one value as ``(value, first, last)`` triples."""
    route = list(route)
    out, k0 = [], 0
    for k in range(1, len(route) + 1):
        if k == len(route) or route[k] != route[k0]:
            out.append((route[k0], k0, k - 1))
            k0 = k
    return out


def trajectory_panel(t, series, route, qoi, labels=None, band_alpha=TRAJ_BAND_ALPHA,
                     band_cmap=None, colors=None, figsize=(13, 7.5), ncols=2,
                     xlabel="$t$", savepath=None):
    """One panel per QoI: several solutions of one trajectory over its route.

    ``series`` maps a key of :data:`TRAJ_COLORS` -- ``"reference"``, ``"clustered"``, ``"global"`` -- to a ``(n_species, n_times)`` block, or to ``None`` for a solve that failed. Bands behind the curves mark contiguous runs of one cluster, so a band boundary is a re-routing.

    ``route`` is one cluster per interval and must be one shorter than ``t``. The palette cycles past its length; adjacent bands still differ, because the runs are contiguous by construction.
    """
    t = np.asarray(t, dtype=float)
    qoi = list(qoi)
    pal = dict(TRAJ_COLORS, **(colors or {}))
    cm = band_cmap or plt.cm.tab20
    n_col = int(getattr(cm, "N", 20))
    names = labels or [rf"$x_{{{q}}}$" for q in qoi]

    cc = {c: cm(i % n_col) for i, c in enumerate(sorted(set(route)))}
    segments = _runs(route)

    nrows = int(np.ceil(len(qoi) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True,
                             squeeze=False)
    axes = axes.ravel()
    for ax, q, name in zip(axes, qoi, names):
        for c, a, b in segments:
            ax.axvspan(t[a], t[min(b + 1, t.size - 1)], color=cc[c],
                       alpha=band_alpha, lw=0, zorder=0)
        for z, (key, y) in enumerate(series.items()):
            if y is None:
                continue
            ls, lw = TRAJ_STYLE.get(key, ("-", 1.8))
            ax.plot(t, np.asarray(y)[q], ls, lw=lw, color=pal[key], zorder=2 + z)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
    for ax in axes[len(qoi):]:
        ax.set_visible(False)
    axes[0].set_xlim(t[0], t[-1])              # shared, so this sets them all
    for ax in axes[max(0, len(qoi) - ncols):len(qoi)]:
        ax.set_xlabel(xlabel)

    # The cluster key draws the palette itself; one entry, not one per cluster.
    band_key = Patch(label="Clusters")
    handles = [Line2D([], [], color=pal[k], ls=TRAJ_STYLE[k][0],
                      lw=TRAJ_STYLE[k][1], label=lab)
               for k, lab in (("reference", "FOM"),
                              ("clustered", "Ensemble ROM"),
                              ("global", "Global ROM"))
               if series.get(k) is not None] + [band_key]
    fig.legend(handles=handles,
               handler_map={band_key: _HandlerBands(
                   [cc[c] for c in sorted(cc)], alpha=0.45)},
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))     # strip at the top for the legend
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"wrote {savepath}")
    return fig, axes


def error_vs_parameter_panel(splits, ranks, qoi, labels=None, agg="l2-in-time",
                             xlabel="$p$", figsize=(27, 8), savepath=None):
    """Per-trajectory error against the parameter, per split, plus basis sizes.

    ``splits`` maps a dataset name to ``(parameter values, {qoi: errors})``, both ordered by the parameter. ``ranks`` maps a cluster id to its basis size; the third panel bars them, which is where an ``increase_rank`` cluster shows up above what the variance threshold alone selected.
    """
    qoi = list(qoi)
    names = labels or [str(q) for q in qoi]
    fig, axes = plt.subplots(1, len(splits) + 1, figsize=figsize)
    err_axes, ax_rank = list(axes[:-1]), axes[-1]
    for ax in err_axes[1:]:
        ax.sharey(err_axes[0])                 # same decades across the splits

    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(qoi)))
    for ax, (split, (pvals, errs)) in zip(err_axes, splits.items()):
        for c, q, name in zip(colors, qoi, names):
            ax.plot(pvals, errs[q], "o-", color=c, ms=4, lw=1.2, label=name)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{split.capitalize()} set")
        ax.grid(alpha=0.3, which="both")
    err_axes[0].set_ylabel(r"Relative $L^2$ error")
    err_axes[0].legend(frameon=False, ncol=2)
    for ax in err_axes[1:]:
        ax.tick_params(labelleft=False)

    cs = sorted(ranks)
    sizes = [ranks[c] for c in cs]
    ax_rank.bar(range(len(cs)), sizes, color="0.55")
    ax_rank.set_xlabel("Cluster index")
    ax_rank.set_ylabel("Basis size")
    ax_rank.set_title(f"{len(cs)} clusters, mean basis size {np.mean(sizes):.1f}")
    ax_rank.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"wrote {savepath}")
    return fig, axes

# ---------------------------------------------------------------------------
# Per-tracer error distribution
# ---------------------------------------------------------------------------

def pertracer_path(sweep_dir, tau, eta, label="on", eval_split="test"):
    """Per-combination per-tracer CSV.

    ``label`` is the trailing token: the positivity mode for a pure-ROM sweep, or the solver tag for a hybrid one.
    """
    return os.path.join(
        sweep_dir,
        f"pertracer_{combo_stem(tau, eta)}_{label}"
        f"{split_suffix(eval_split)}.csv")


def etas_on_disk(sweep_dir, taus, label="on", eval_split="test"):
    """Every eta with a per-tracer file for *all* of ``taus``, largest first.

    Intersecting rather than unioning keeps the columns comparable across a row; a union would leave holes that read as if those runs had failed.
    """
    sfx = split_suffix(eval_split)
    per_tau = []
    for tau in taus:
        pat = os.path.join(
            sweep_dir, f"pertracer_tol{tau:g}_eta*_{label}{sfx}.csv")
        found = set()
        for path in glob.glob(pat):
            m = re.search(r"_eta([0-9.e+-]+)_", os.path.basename(path))
            if m:
                found.add(float(m.group(1)))
        per_tau.append(found)
    return sorted(set.intersection(*per_tau), reverse=True) if per_tau else []


def violin_panel(sweep_dir, taus, qoi, etas=None, label="on",
                 eval_split="test", agg="l2-in-time", floor=1e-16, ylim=None,
                 figsize=(15, 20), fs=28, eta_fs=24, stride=2,
                 err_label=r"$L^2$ error", savepath=None, rank_df=None,
                 rank_label="Mean basis size", rank_labelpad=12):
    """Per-tracer error distribution vs eta, one row per QoI, one column per tau.

    Reads the per-combo ``pertracer_*.csv`` files the sweep already wrote; nothing is re-solved, so changing ``etas`` or ``qoi`` is a redraw.

    Each violin is the distribution of ``log10`` per-tracer error over the tracers that completed; the solid rule is the median and the dashed one the p95. A summary statistic alone cannot say whether a bad mean is a heavy tail or a uniform shift, which is what these show.

    ``rank_df`` is a sweep frame carrying ``mean_rank`` per ``(tau_split, eta)``, such as the one passed to :func:`sweep_panel_splits`. When given, each top-row panel receives a twin x axis that labels every kept violin with the mean basis size of its tolerance and eta; an eta absent from the frame is left unlabelled. ``rank_labelpad`` is the gap in points between that axis's tick labels and its label.

    Each tolerance is captioned inside the top right of its top-row panel, where the errors are smallest and the violins leave room. The y axis is shared within a row only, so each quantity is scaled to its own error range; ``ylim``, when given, applies to every panel.
    """
    taus = list(taus)
    qoi = list(qoi)
    etas = ([float(e) for e in etas] if etas
            else etas_on_disk(sweep_dir, taus, label, eval_split))
    if not etas:
        raise FileNotFoundError(
            f"no {label}{split_suffix(eval_split)} per-tracer file shared by "
            f"tau={taus} under {sweep_dir}")

    use_paper_style()
    fig, axes = plt.subplots(len(qoi), len(taus), figsize=figsize,
                             sharex=True, sharey="row", squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(etas)))

    for j, tau in enumerate(taus):
        frames = {}
        for eta in etas:
            path = pertracer_path(sweep_dir, tau, eta, label, eval_split)
            if os.path.exists(path):
                frames[eta] = read_table(path)
        for i, q in enumerate(qoi):
            ax = axes[i][j]
            data, pos = [], []
            for kk, eta in enumerate(etas):
                df = frames.get(eta)
                if df is None:
                    continue
                col = f"{agg}|{q}"
                if col not in columns_of(df):
                    raise KeyError(
                        f"{col} not in "
                        f"{pertracer_path(sweep_dir, tau, eta, label, eval_split)}; "
                        f"QoIs there: "
                        f"{[c.split('|')[1] for c in columns_of(df) if '|' in c]}")
                # A frame may hold NaN where a solve was invalid; drop those rather than plot them at the floor.
                e = np.asarray(df[col], dtype=float)
                e = e[np.isfinite(e) & (e > 0)]
                if e.size == 0:
                    continue
                data.append(np.log10(np.maximum(e, floor)))
                pos.append(kk + 1)
            if not data:
                continue
            parts = ax.violinplot(data, positions=pos, widths=0.85,
                                  showextrema=False, showmedians=False)
            for body, kk in zip(parts["bodies"], [x - 1 for x in pos]):
                body.set_facecolor(colors[kk])
                body.set_alpha(0.8)
                body.set_edgecolor("none")
            for d, x in zip(data, pos):
                ax.hlines(np.median(d), x - 0.36, x + 0.36, color="k", lw=1.8)
                ax.hlines(np.percentile(d, 95), x - 0.36, x + 0.36,
                          color="k", lw=1.4, ls=(0, (2, 2)))
            if i == 0:
                ax.text(0.97, 0.95, rf"$\tau={tau:g}$", transform=ax.transAxes,
                        ha="right", va="top", fontsize=fs, zorder=6)
            if j == 0:
                ax.set_ylabel(f"{q}\n" + rf"$\log_{{10}}$ {err_label or agg}")
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.25)

    # Labels sit on major ticks at the kept positions and every violin keeps a minor tick, so dropping a label does not hide that the violin is there. sharex ties tick locations together, not the label objects, so the bottom row is labelled axes by axes.
    keep = list(range(0, len(etas), stride))
    if keep[-1] != len(etas) - 1:
        keep.append(len(etas) - 1)
    axes[0][0].set_xlim(0.4, len(etas) + 0.6)
    for ax in axes[-1]:
        ax.set_xticks([k + 1 for k in keep])
        ax.set_xticklabels(
            [rf"$10^{{{int(round(np.log10(etas[k])))}}}$" for k in keep])
        ax.set_xticks(np.arange(1, len(etas) + 1), minor=True)
        ax.set_xlabel(r"$\eta$", fontsize=fs)
        ax.tick_params(axis="x", which="major", labelsize=eta_fs)
        ax.tick_params(axis="x", which="minor", length=3)
    twins = []
    if rank_df is not None:
        # The twin axis carries the same kept positions as the eta axis, so each basis-size label sits over the violin whose eta is labelled below it.
        for tau, host in zip(taus, axes[0]):
            eta_r, rank_r = rank_series(rank_df, tau)

            def _label(eta):
                hit = np.isclose(eta_r, eta, rtol=1e-6, atol=0.0)
                r = rank_r[hit][0] if hit.any() else np.nan
                return f"{r:.1f}" if np.isfinite(r) else ""

            tw = host.twiny()
            tw.set_xlim(host.get_xlim())
            tw.set_xticks([k + 1 for k in keep])
            tw.set_xticklabels([_label(etas[k]) for k in keep])
            tw.set_xticks(np.arange(1, len(etas) + 1), minor=True)
            tw.set_xlabel(rank_label, fontsize=eta_fs, labelpad=rank_labelpad)
            tw.tick_params(axis="x", which="major", labelsize=eta_fs)
            tw.tick_params(axis="x", which="minor", length=3)
            tw.grid(False)
            twins.append((host, tw))
    for ax in axes.ravel():
        ax.tick_params(axis="y", labelsize=fs)
        ax.yaxis.label.set_fontsize(fs)

    fig.tight_layout()
    handles = [Line2D([], [], color="k", lw=1.8),
               Line2D([], [], color="k", lw=1.4, ls=(0, (2, 2)))]
    # Anchored to the top of the drawn top row, twin axes included, rather than the figure edge, since tight_layout leaves a margin above it that would otherwise separate the legend from the panels.
    renderer = fig.canvas.get_renderer()
    top = max(ax.get_tightbbox(renderer).y1
              for ax in [*axes[0], *(tw for _, tw in twins)])
    top = fig.transFigure.inverted().transform((0.0, top))[1]
    fig.legend(handles, ["median", "p95"], ncol=2, frameon=False,
               loc="lower right", bbox_to_anchor=(1.0, top), fontsize=eta_fs)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {savepath}")
    return fig, axes


# ---------------------------------------------------------------------------
# Per-tracer orchestrators
# ---------------------------------------------------------------------------

def _load_combo(sweep_dir, stem, ctx):
    """The (cluster, ensemble) pair for one combination, ready to solve."""
    ens_npz = os.path.join(sweep_dir, f"ensemble_{stem}.npz")
    if not os.path.exists(ens_npz):
        return None, None
    cl = Cluster.load(os.path.join(sweep_dir, f"cluster_{stem}.npz"))
    return cl, EnsembleROM.load(ens_npz, cl, ctx.net).precompute_B_structure()


def tracer_figures(sweep_dir, stem, fig_dir, ctx, cfg, fm_train, tracers,
                   x_eqs, t_foms, y_foms, tracer, global_rank=None,
                   global_ids=None, position=None):
    """Draw the abundance panel for one tracer of one fit.

    Solves the tracer twice, positivity on and off. ``global_rank`` adds the single-basis baseline as dashed curves; leave it ``None`` to omit it and skip the global refit, which is the slow part of this call.

    ``tracer`` indexes the lists passed in; ``position`` is that trajectory's position in the dataset, used to report it and to index ``global_ids``. They differ when the caller loaded one trajectory instead of a prefix.

    Returns ``(figure, info)``, left open so a caller can display it, or ``(None, None)`` if the combo was never fitted or the tracer not loaded.
    """
    os.makedirs(fig_dir, exist_ok=True)
    cl, ens = _load_combo(sweep_dir, stem, ctx)
    if ens is None:
        print(f"no fit for {stem} under {sweep_dir}; skipping the abundance figure")
        return None, None
    if tracer >= len(t_foms):
        print(f"tracer {tracer} was not loaded (only {len(t_foms)} FOM "
              f"trajectories); skipping the abundance figure")
        return None, None

    gids = ctx.test_indices if global_ids is None else global_ids
    pos = tracer if position is None else int(position)
    print(f"abundance figure: tracer {pos} (global id {gids[pos]}), {stem}")

    solve = dict(dt_hydro=ctx.dt_hydro, atol=cfg.atol, rtol=cfg.rtol,
                 method=cfg.ode_method)
    args = (tracers[tracer], x_eqs[tracer], t_foms[tracer], y_foms[tracer])
    d = D.tracer_data(ens, *args, **solve)

    d_global = None
    if global_rank:
        ens_global = fit_global_rom(fm_train, ctx.net, cfg, ctx.qoi_indices,
                                    ctx.param_names, global_rank, cfg.n_workers)
        d_global = D.tracer_data(ens_global, *args, **solve)
    fig, _, info = abundance_panel(
        d, d_global, ctx.net.species, ctx.qoi_indices, ensemble=ens,
        savepath=os.path.join(fig_dir, "abundance.pdf"))
    return fig, info


def hybrid_figures(sweep_dir, stem, fig_dir, ctx, cfg, tracers, x_eqs, t_foms,
                   y_foms, tracer, hybrid=None, time_scale=1.0,
                   time_label="Time [s]", position=None):
    """Draw the hybrid comparison panel for one tracer.

    Solves that tracer twice: the hybrid, and the pure ROM it is compared against. ``position`` reports the trajectory's place in the dataset when ``tracer`` is an index into a one-element list rather than that position.

    Returns ``(figure, result)``, left open so a caller can display it, or ``(None, None)`` if the combo was never fitted.
    """
    os.makedirs(fig_dir, exist_ok=True)
    cl, ens = _load_combo(sweep_dir, stem, ctx)
    if ens is None:
        print(f"no fit for {stem} under {sweep_dir}; skipping the hybrid figure")
        return None, None
    if tracer >= len(t_foms):
        print(f"tracer {tracer} was not loaded; skipping the hybrid figure")
        return None, None
    print(f"hybrid figure: tracer {tracer if position is None else position}, "
          f"{stem}")

    hybrid = hybrid or HybridConfig()
    pt, x0 = tracers[tracer], x_eqs[tracer]
    res = solve_hybrid(ens, cfg, pt, x0, ctx.dt_hydro, t_eval=t_foms[tracer],
                       hybrid=hybrid)
    rom = solve_pure_rom(ens, cfg, pt, x0, ctx.dt_hydro,
                         t_eval=t_foms[tracer], positivity=hybrid.positivity)

    fig, _ = hybrid_comparison_panel(
        res, t_foms[tracer], y_foms[tracer], qoi_indices=ctx.qoi_indices,
        species=ctx.net.species, rom=rom, time_scale=time_scale,
        time_label=time_label)
    path = os.path.join(fig_dir, "hybrid_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")

    return fig, res
