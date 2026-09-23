"""Per-tracer trajectories and route diagnostics for the figures.

Reports which clusters a tracer routed through, how well each basis represented the state handed to it, and what the positivity projection changed. Solves ODEs, so it is far slower than reading a stored trajectory.
"""
from __future__ import annotations

import numpy as np



def knot_mask(t_yr, dt_yr, rtol=1e-6):
    """Which entries of ``t_yr`` land on a hydro-step boundary (a knot).

    Trajectories carry intra-step sub-points too, but the knots are where the piecewise solve hands state across, so the diagnostics live there.
    """
    frac = (np.asarray(t_yr, dtype=np.float64) / dt_yr) % 1.0
    return (frac < rtol) | (frac > 1 - rtol)


def steps_to_snapshots(step_path, t_axis, dt_hydro):
    """Expand one cluster per hydro step to one per snapshot on ``t_axis``."""
    step_path = np.asarray(step_path)
    step = np.clip((np.asarray(t_axis) / dt_hydro).astype(np.int64),
                   0, step_path.size - 1)
    return step_path[step]


def route_segments(cluster_path, t_yr):
    """Run-length-encode a cluster path into bands the figures shade from.

    Returns the path, each run's cluster, and the run boundaries in years.
    """
    cp = np.asarray(cluster_path)
    t_yr = np.asarray(t_yr)
    if cp.size == 0:                      # a solve that failed on its first interval
        return dict(cp=cp, seg=cp[:0], te=t_yr[:0])
    edges = np.r_[0, np.flatnonzero(np.diff(cp)) + 1, cp.size]
    return dict(cp=cp, seg=cp[edges[:-1]],
                te=t_yr[np.clip(edges, 0, cp.size - 1)])


def route_stats(d, ensemble=None, dims=None, which="rom", key=None):
    """How much of a cluster ensemble one tracer's route actually touches.

    ``d`` is one entry of :func:`tracer_data`, the same dict the abundance panels shade from.

    ``which="rom"`` uses the route the ROM integrated on (``rom_pos`` when the positivity-on solve is present, matching what :func:`plots.abundance_panel` draws); ``which="fom"`` uses the route the FOM trajectory would have taken. ``key`` overrides ``which`` with an exact segment key, so ``key="rom"`` forces the positivity-off solve's route even when the other is present.

    Basis dimensions come from ``ensemble`` (``roms[c].basis.rank``) or from an explicit ``dims`` mapping ``{cluster: dim}``; with neither, only the counts are returned.

    ``mean_dim`` is weighted by how often each cluster appears in the route, so it is the basis size the tracer sees on average rather than the ensemble's mean basis size. ``mean_dim_time`` weights by time spent instead, which differs whenever the snapshot axis is not uniform in time; ``mean_dim_unweighted`` gives every visited cluster equal weight.
    """
    if key is not None:                      # an exact segment dict was named
        which = key
        seg = d[key]
    elif which == "fom":
        seg = d["fom"]
    elif which == "rom":
        seg = d.get("rom_pos", d["rom"])
    else:
        raise ValueError(f"which must be 'rom' or 'fom'; got {which!r}")

    cp = np.asarray(seg["cp"])
    clusters, counts = np.unique(cp, return_counts=True)
    clusters = [int(c) for c in clusters]
    # Time in each cluster: sum the run-length segments' durations, so a cluster revisited three times accumulates all three.
    te = np.asarray(seg["te"], dtype=float)
    dur = {c: 0.0 for c in clusters}
    for c, t0, t1 in zip(np.asarray(seg["seg"]), te[:-1], te[1:]):
        dur[int(c)] += float(t1 - t0)

    out = dict(
        which=which,
        clusters=clusters,
        n_clusters=len(clusters),
        counts={c: int(n) for c, n in zip(clusters, counts)},
        fractions={c: float(n) / cp.size for c, n in zip(clusters, counts)},
        time={c: dur[c] for c in clusters},
        n_switches=int(len(np.asarray(seg["seg"])) - 1),   # route transitions
    )

    if dims is None and ensemble is not None:
        dims = {int(c): int(ensemble.roms[int(c)].basis.rank) for c in clusters}
    if dims is None:
        return out

    dims = {int(c): int(dims[int(c)]) for c in clusters}
    w = np.array([out["counts"][c] for c in clusters], dtype=float)
    wt = np.array([out["time"][c] for c in clusters], dtype=float)
    v = np.array([dims[c] for c in clusters], dtype=float)
    out.update(
        dims=dims,
        mean_dim=float(np.average(v, weights=w)),
        mean_dim_time=float(np.average(v, weights=wt)) if wt.sum() > 0 else float("nan"),
        mean_dim_unweighted=float(v.mean()),
        min_dim=int(v.min()),
        max_dim=int(v.max()),
        total_dim=int(v.sum()),          # dofs held across the visited clusters
    )
    return out


def projection_residual(ensemble, y, t_yr, cluster_path, dt_yr):
    """Relative projection residual of each interval's entry state.

    ``||v - R L v|| / ||v||`` with ``v = x - x_bar``, in the basis's own scaled units: how well the basis about to be integrated in represents the state handed to it. A spike marks a bad cluster switch.

    Returns ``(t_yr_at_knots, residual)``, one entry per interval.
    """
    t_yr = np.asarray(t_yr)
    cluster_path = np.asarray(cluster_path)
    m = min(y.shape[1], cluster_path.size, t_yr.size)
    if m == 0:
        return np.empty(0), np.empty(0)
    knot = np.flatnonzero(knot_mask(t_yr[:m], dt_yr))
    if knot.size < 2:
        return np.empty(0), np.empty(0)

    # Read the cluster from inside the interval, not at the knot: int(t/dt) at an exact knot time rounds down to the previous interval, pairing the entry state with the basis that produced it and zeroing the residual.
    mid = knot[:-1] + np.maximum(np.diff(knot) // 2, 1)
    out = np.full(mid.size, np.nan)
    for k, cid in enumerate(cluster_path[np.minimum(mid, m - 1)]):
        rom = ensemble.roms.get(int(cid))
        if rom is None:
            continue
        b = rom.basis
        v = y[:, knot[k]] - b._x_bar
        err = v - b._R @ (b._L @ v)
        out[k] = (np.linalg.norm(err / b._scale)
                  / (np.linalg.norm(v / b._scale) + 1e-300))
    return t_yr[knot[:-1]], out


def dex_error(y_fom, t_fom_yr, y_rom, t_rom_yr, peak, floor=1e-12):
    """RMS ``|log10(ROM) - log10(FOM)|`` per species: off by how many decades.

    Weights every snapshot equally, so a startup transient cannot dominate a well-tracked tail. Both trajectories are clipped at a floor relative to each species' FOM peak, so a ROM reaching zero scores a bounded 12 dex rather than infinity.
    """
    y_fom, y_rom = np.asarray(y_fom), np.asarray(y_rom)
    t_fom_yr, t_rom_yr = np.asarray(t_fom_yr), np.asarray(t_rom_yr)
    if t_rom_yr.shape[0] != t_fom_yr.shape[0]:
        # A solve that died early returns a short axis; align it as evaluate does, so every species is scored over the same span.
        y_rom = np.stack([np.interp(t_fom_yr, t_rom_yr, y_rom[j])
                          for j in range(y_rom.shape[0])])
    flo = (np.where(peak > 0, peak, 1.0) * floor)[:, None]
    dex = (np.log10(np.clip(y_rom, flo, None))
           - np.log10(np.clip(y_fom, flo, None)))
    return np.sqrt(np.mean(dex ** 2, axis=1))


def tracer_data(ensemble, tracer, x0, t_fom, y_fom, dt_hydro, atol, rtol,
                method="BDF", positivity=(True, False), time_scale=1.0):
    """Solve one tracer once per entry in ``positivity``, in the panel's shape.

    Keys follow what :func:`plots.abundance_panel` reads: the positivity-on solve lands under ``y_rom_pos``/``tyr_rom_pos``/``rom_pos`` and the positivity-off one under ``y_rom``/``tyr_rom``/``rom``. Given both, the panel draws the positivity-on one.

    ``time_scale`` divides every time axis, e.g. seconds per year for chemistry; ``dt_hydro`` itself stays in the solver's units.
    """
    out = {"tyr": np.asarray(t_fom) / time_scale,
           "y_fom": np.asarray(y_fom),
           "peak": np.asarray(y_fom).max(axis=1),
           "dt_hydro_yr": dt_hydro / time_scale}

    for on in positivity:
        sfx = "_pos" if on else ""
        t, y, path, info = ensemble.solve_tracer(
            tracer, x0, dt_hydro, atol, rtol, t_eval=t_fom, positivity=on,
            method=method)
        out[f"tyr_rom{sfx}"] = t / time_scale
        out[f"y_rom{sfx}"] = y
        out[f"ok{sfx}"] = info is None
        if t.size == 0:
            # Nothing to draw, and the reason is in `info`; report it here rather than let it surface as an index error downstream.
            raise RuntimeError(
                f"the positivity-{'on' if on else 'off'} solve returned no "
                f"trajectory for this tracer: {info!r}")
        # Per-solve segments: one that dies early covers only its own prefix, so sharing them would squeeze the other route into the left edge.
        out[f"rom{sfx}"] = route_segments(
            steps_to_snapshots(path, t, dt_hydro), t / time_scale)

    # The panel does d.get("rom_pos", d["rom"]), which evaluates d["rom"] eagerly, so the positivity-off keys must exist even when only the positivity-on solve ran.
    if "rom" not in out:
        for k in ("rom", "y_rom", "tyr_rom", "ok"):
            out[k] = out[f"{k}_pos"]
    return out
