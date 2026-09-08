"""Stage 4: normalisation statistics and climatology.

Neural networks train badly when their inputs live on wildly different scales.
Here, sea surface temperature is around 28, sea level anomaly around 0.05, and
salinity around 35. Fed in raw, the network's gradients would be dominated by
whichever variable happens to have the largest numbers, and salinity's useful
signal would be swamped.

The fix is standardisation: subtract the mean, divide by the standard
deviation, so every input arrives centred on 0 with a spread of 1.

Two rules this module enforces, both of which are easy to get wrong and both
of which quietly inflate your final scores if you do:

  1. Statistics are computed on the TRAINING SPLIT ONLY. Using the whole
     record would leak information about the validation and test years into
     training, and the reported skill would be optimistic.

  2. Land is excluded. Averaging over NaN-filled land cells would corrupt
     every statistic.

It also computes a day-of-year climatology - the average conditions for each
calendar day. Predicting the DEPARTURE from normal is a much better-posed
problem than predicting absolute temperature, because the seasonal cycle is
large, obvious, and already known.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from .config import Config
from .build_cube import load_mask, open_cube

STATS_FILE = "norm_stats.json"
CLIMATOLOGY_FILE = "climatology.nc"


def compute(cfg: Config, split: str = "train") -> dict:
    """Per-variable mean and standard deviation over the training split.

    The target is handled per depth level, not as one number: temperature at
    5 m varies by many degrees across the basin and seasons, while at 1000 m
    it barely moves. A single shared statistic would make the deep levels
    look almost constant to the network, and their errors would vanish from
    the loss.
    """
    mask = load_mask(cfg)
    train_days = set(cfg.split_dates(split))

    stats: dict[str, object] = {
        "split": split,
        "n_days": 0,
        "surface": {},
        "target": {},
    }

    # ------------------------------------------------------------- surface
    surface = open_cube(cfg, "surface")
    sel = _select_days(surface, train_days)
    stats["n_days"] = int(sel.sizes["time"])

    for var in surface.data_vars:
        values = sel[var].values
        values = values[:, mask] if values.ndim == 3 else values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        stats["surface"][str(var)] = {
            "mean": float(finite.mean()),
            "std": float(finite.std()) or 1.0,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "valid_fraction": float(finite.size / values.size),
        }
    surface.close()

    # -------------------------------------------------------------- target
    try:
        target = open_cube(cfg, "target")
    except FileNotFoundError:
        target = None

    if target is not None:
        tvar = cfg["target"]["variable"]
        sel_t = _select_days(target, train_days)
        depths = [float(d) for d in sel_t.depth.values]
        per_depth = []

        for i, depth in enumerate(depths):
            values = sel_t[tvar].isel(depth=i).values[:, mask]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                # Below the sea floor everywhere: keep a placeholder so the
                # array of statistics stays aligned with the depth axis.
                per_depth.append(
                    {"depth": depth, "mean": 0.0, "std": 1.0, "valid_fraction": 0.0}
                )
                continue
            per_depth.append(
                {
                    "depth": depth,
                    "mean": float(finite.mean()),
                    "std": float(finite.std()) or 1.0,
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "valid_fraction": float(finite.size / values.size),
                }
            )

        stats["target"] = {"variable": tvar, "levels": per_depth}
        target.close()

    dest = cfg.processed_dir / STATS_FILE
    dest.write_text(json.dumps(stats, indent=2))
    return stats


def _select_days(ds: xr.Dataset, wanted: set) -> xr.Dataset:
    """Restrict a cube to a set of calendar dates."""
    import pandas as pd

    times = pd.DatetimeIndex(ds.time.values).date
    keep = np.array([t in wanted for t in times])
    if not keep.any():
        raise ValueError(
            "None of the split's days are present in the cube. "
            "Check that time.splits in the config overlaps the days you built."
        )
    return ds.isel(time=np.where(keep)[0])


def climatology(cfg: Config, split: str = "train", smooth_days: int = 15) -> Path:
    """Average conditions for each calendar day of the year.

    Computed from the training split only, for the same leakage reason as the
    normalisation statistics. Smoothed across neighbouring days because with
    only a few years of data each individual calendar day is noisy - 1 March
    should not look meaningfully different from 2 March.
    """
    import pandas as pd

    mask = load_mask(cfg)
    train_days = set(cfg.split_dates(split))

    surface = open_cube(cfg, "surface")
    sel = _select_days(surface, train_days).load()
    doy = pd.DatetimeIndex(sel.time.values).dayofyear

    clim = sel.assign_coords(dayofyear=("time", doy)).groupby("dayofyear").mean("time")

    # Circular smoothing so 31 December and 1 January stay continuous.
    if smooth_days > 1:
        clim = clim.pad(dayofyear=smooth_days, mode="wrap")
        clim = clim.rolling(dayofyear=smooth_days, center=True, min_periods=1).mean()
        clim = clim.isel(dayofyear=slice(smooth_days, -smooth_days))

    clim = clim.where(xr.DataArray(mask, dims=("lat", "lon")))
    clim.attrs.update(
        description=f"day-of-year climatology from the {split} split",
        smoothing_days=smooth_days,
    )

    dest = cfg.processed_dir / CLIMATOLOGY_FILE
    encoding = {v: {"zlib": True, "complevel": 4} for v in clim.data_vars}
    clim.to_netcdf(dest, encoding=encoding)
    surface.close()
    return dest


def load_stats(cfg: Config) -> dict:
    p = cfg.processed_dir / STATS_FILE
    if not p.exists():
        raise FileNotFoundError(f"stats missing: {p}\nRun: oceanembed stats")
    return json.loads(p.read_text())
