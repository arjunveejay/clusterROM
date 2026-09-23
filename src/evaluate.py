"""Solve the ensemble over a tracer set and reduce the errors to statistics.

Also the CSV layer. Results are passed around as lists of dicts, one per row, and read back as numpy structured arrays, so nothing here needs pandas.

A structured array supports the access the figures want -- ``tab["eta"]`` for a column and ``tab[tab["qoi"] == "CO"]`` for a row subset. Unlike ``genfromtxt``, :func:`read_table` keeps field names verbatim, which the per-tracer columns (``l2-in-time|CO``) depend on.
"""
from __future__ import annotations

import csv

import numpy as np
from scipy.interpolate import interp1d

AGGREGATES = ("mean-in-time", "max-in-time", "l2-in-time")
STATS = ("mean", "std", "median", "p90", "p95", "p99", "max")

_FLOOR = 1e-30


def _stats(x):
    """Per-QoI summary of an ``(n_valid, n_qoi)`` error array."""
    return {"mean": x.mean(axis=0), "std": x.std(axis=0),
            "median": np.median(x, axis=0),
            "p90": np.percentile(x, 90, axis=0),
            "p95": np.percentile(x, 95, axis=0),
            "p99": np.percentile(x, 99, axis=0),
            "max": x.max(axis=0)}


def evaluate(ensemble, tracers, x_eqs, t_foms, y_foms, qoi_indices, qoi_names,
             dt_hydro, cfg, verbose=True):
    """Solve every tracer and reduce the per-species errors three ways.

    ``mean-in-time`` and ``max-in-time`` reduce the pointwise relative error over time; ``l2-in-time`` is ``||y_rom - y_fom|| / ||y_fom||`` along the time axis. Only tracers that completed every hydro step are scored.
    """
    n = len(t_foms)
    results = ensemble.solve_tracers(
        tracers[:n], x_eqs, dt_hydro, cfg.atol, cfg.rtol, list(t_foms),
        cfg.n_workers, positivity=cfg.positivity,
        method=cfg.ode_method, verbose=verbose)
    t_roms, y_roms, _paths, infos = map(list, zip(*results))

    expected_end = (np.asarray(tracers[0]).shape[0] - 1) * dt_hydro
    valid = [i for i in range(n)
             if infos[i] is None and t_roms[i].size
             and np.isclose(t_roms[i][-1], expected_end)]

    errors, l2_rel = [], []
    for i in valid:
        t_fom, y_fom, y_rom = t_foms[i], y_foms[i], y_roms[i]
        if t_roms[i].shape[0] != t_fom.shape[0]:
            y_rom = interp1d(t_roms[i], y_rom, axis=1, bounds_error=False,
                             fill_value="extrapolate")(t_fom)
        resid = y_rom - y_fom
        errors.append(np.abs(resid) / (np.abs(y_fom) + _FLOOR))
        l2_rel.append(np.sqrt(np.sum(resid ** 2, axis=1))
                      / (np.sqrt(np.sum(y_fom ** 2, axis=1)) + _FLOOR))

    aggs = {"mean-in-time": np.stack([e.mean(axis=1) for e in errors]),
            "max-in-time": np.stack([e.max(axis=1) for e in errors]),
            "l2-in-time": np.stack(l2_rel)}

    out = {"n_tracers": n, "n_valid": len(valid), "valid_idx": list(valid),
           "n_solved": sum(1 for i in infos if i is None), "per_tracer": {}}
    for tag, agg in aggs.items():
        x = agg[:, qoi_indices]                       # (n_valid, n_qoi)
        st = _stats(x)
        out[tag] = {name: {s: float(st[s][j]) for s in STATS}
                    for j, name in enumerate(qoi_names)}
        out["per_tracer"][tag] = {name: x[:, j].copy()
                                  for j, name in enumerate(qoi_names)}
    return out


def metrics_rows(res, qoi_names, **meta):
    """One row dict per (aggregate, QoI, statistic), tagged with ``meta``."""
    return [dict(agg=agg, qoi=q, stat=stat, value=value,
                 n_solved=res["n_solved"], n_valid=res["n_valid"], **meta)
            for agg in AGGREGATES for q in qoi_names
            for stat, value in res[agg][q].items()]


def per_tracer_rows(res, global_ids):
    """Unreduced per-tracer errors, next to each tracer's global id.

    The summary statistics cannot say whether a bad mean is a heavy tail or a uniform shift; these can.
    """
    pos = np.asarray(res["valid_idx"], dtype=int)
    gid = np.asarray(global_ids)[pos]
    cols = {f"{agg}|{q}": v for agg, per_qoi in res["per_tracer"].items()
            for q, v in per_qoi.items()}
    return [dict(position=int(pos[i]), global_id=int(gid[i]),
                 **{k: float(v[i]) for k, v in cols.items()})
            for i in range(pos.size)]


def write_csv(path, rows):
    """Write row dicts to ``path``. The first row fixes the column order."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def read_table(path):
    """Read a CSV into a structured array, one field per column.

    A column parses as float where every value does, and as a string otherwise; an empty cell in a numeric column becomes ``nan``.
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        raw = list(reader)
    if not raw:
        raise ValueError(f"{path} has a header but no rows")

    columns, dtypes = [], []
    for j, name in enumerate(header):
        vals = [r[j] for r in raw]
        try:
            col = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            col = np.array(vals, dtype=str)
        columns.append(col)
        dtypes.append((name, col.dtype))

    tab = np.empty(len(raw), dtype=dtypes)
    for name, col in zip(header, columns):
        tab[name] = col
    return tab


def columns_of(tab):
    """Field names of a table read by :func:`read_table`."""
    return tab.dtype.names


def where(tab, **equals):
    """Rows of ``tab`` matching every ``column=value`` pair."""
    mask = np.ones(tab.size, dtype=bool)
    for col, val in equals.items():
        mask &= tab[col] == val
    return tab[mask]


def pivot_mean(tab, index, values, columns, keys):
    """``(index values, mean over keys)`` of ``values``, grouped by ``index``.

    One row per distinct ``index``, averaged over the ``keys`` of ``columns`` so every key counts equally. Index values are returned sorted.
    """
    idx = np.unique(tab[index])
    out = np.full(idx.size, np.nan)
    for i, v in enumerate(idx):
        rows = tab[tab[index] == v]
        per_key = [rows[rows[columns] == k][values] for k in keys]
        per_key = [float(a.mean()) for a in per_key if a.size]
        if per_key:
            out[i] = float(np.mean(per_key))
    return idx, out
