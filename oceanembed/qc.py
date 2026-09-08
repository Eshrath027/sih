"""Stage 5: quality control - prove the cube is trustworthy before training.

A model trained on a quietly broken dataset produces confident nonsense, and
the failure is very hard to diagnose later from the loss curve alone. This
stage answers the questions worth asking before any GPU time is spent:

  * Which days are missing, and are the gaps clustered in one period?
  * How much of each field is missing, and does that change over time?
  * Are the values physically plausible at every depth?
  * Is the land mask consistent across variables?
  * Does the mean temperature profile actually look like an ocean?

Writes a JSON report plus diagnostic figures to reports/.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from .config import Config
from .build_cube import load_mask, open_cube
from .harmonize import VALID_RANGES, daily_path


def coverage(cfg: Config) -> dict:
    """Which requested days actually made it into the cube."""
    requested = cfg.dates()
    present = [d for d in requested if daily_path(cfg, d).exists()]
    missing = [d for d in requested if not daily_path(cfg, d).exists()]

    # Group consecutive missing days, so a two-month outage reads as one gap
    # rather than sixty separate lines.
    gaps: list[dict] = []
    for day in missing:
        if gaps and (day - gaps[-1]["_last"]).days == 1:
            gaps[-1]["_last"] = day
            gaps[-1]["days"] += 1
        else:
            gaps.append({"start": day.isoformat(), "_last": day, "days": 1})
    for g in gaps:
        g["end"] = g.pop("_last").isoformat()

    return {
        "requested": len(requested),
        "present": len(present),
        "missing": len(missing),
        "completeness": round(len(present) / max(len(requested), 1), 4),
        "gaps": sorted(gaps, key=lambda g: -g["days"])[:20],
    }


def field_report(cfg: Config) -> dict:
    """Per-variable missing-data fractions and value ranges."""
    mask = load_mask(cfg)
    n_ocean = int(mask.sum())
    out: dict[str, object] = {"ocean_cells": n_ocean, "surface": {}, "target": {}}

    surface = open_cube(cfg, "surface")
    for var in surface.data_vars:
        values = surface[var].values
        ocean = values[:, mask]
        finite = np.isfinite(ocean)
        vals = ocean[finite]
        lo, hi = VALID_RANGES.get(str(var), (-np.inf, np.inf))
        out["surface"][str(var)] = {
            "missing_fraction": round(float(1 - finite.mean()), 5),
            "min": round(float(vals.min()), 4) if vals.size else None,
            "max": round(float(vals.max()), 4) if vals.size else None,
            "mean": round(float(vals.mean()), 4) if vals.size else None,
            "within_plausible_range": bool(vals.size and vals.min() >= lo and vals.max() <= hi),
        }
    surface.close()

    try:
        target = open_cube(cfg, "target")
    except FileNotFoundError:
        return out

    tvar = cfg["target"]["variable"]
    levels = []
    for i, depth in enumerate(target.depth.values):
        ocean = target[tvar].isel(depth=i).values[:, mask]
        finite = np.isfinite(ocean)
        vals = ocean[finite]
        levels.append(
            {
                "depth": float(depth),
                "missing_fraction": round(float(1 - finite.mean()), 5),
                "mean": round(float(vals.mean()), 3) if vals.size else None,
                "min": round(float(vals.min()), 3) if vals.size else None,
                "max": round(float(vals.max()), 3) if vals.size else None,
            }
        )
    out["target"] = {"variable": tvar, "levels": levels}

    # Physical sanity: below the mixed layer the ocean must get colder with
    # depth. A warmer-below-colder result means depths are reversed or the
    # interpolation is broken.
    means = [lv["mean"] for lv in levels if lv["mean"] is not None]
    inversions = [
        (levels[i]["depth"], levels[i + 1]["depth"])
        for i in range(len(means) - 1)
        if means[i + 1] > means[i] + 0.05
    ]
    out["monotonic_cooling_with_depth"] = not inversions
    out["temperature_inversions"] = inversions

    target.close()
    return out


def figures(cfg: Config) -> list[Path]:
    """Diagnostic plots: a map, the mean profile, and a time series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    mask = load_mask(cfg)
    out_dir = cfg.reports_dir

    surface = open_cube(cfg, "surface")

    # --- map of every surface input on one day -------------------------
    variables = list(surface.data_vars)
    ncol = 3
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, var in zip(axes, variables):
        field = surface[var].isel(time=0).values
        im = ax.pcolormesh(surface.lon, surface.lat, field, shading="auto", cmap="viridis")
        ax.set_title(f"{var}", fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes[len(variables):]:
        ax.axis("off")
    day = str(surface.time.values[0])[:10]
    fig.suptitle(f"Surface inputs, {day}", fontsize=12)
    fig.tight_layout()
    p = out_dir / "surface_inputs.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # --- basin-mean SST through time -----------------------------------
    if "sst" in surface:
        series = surface["sst"].where(xr.DataArray(mask, dims=("lat", "lon"))).mean(dim=("lat", "lon"))
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(surface.time.values, series.values, lw=0.9)
        ax.set_ylabel("basin mean SST (degC)")
        ax.set_title("Basin-mean sea surface temperature - look for jumps or flat spots")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / "sst_timeseries.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)
    surface.close()

    # --- mean temperature profile --------------------------------------
    try:
        target = open_cube(cfg, "target")
    except FileNotFoundError:
        return written

    tvar = cfg["target"]["variable"]
    profile = [
        float(np.nanmean(target[tvar].isel(depth=i).values[:, mask]))
        for i in range(target.sizes["depth"])
    ]
    fig, ax = plt.subplots(figsize=(4.5, 6))
    ax.plot(profile, target.depth.values, "o-")
    ax.invert_yaxis()
    ax.set_xlabel("temperature (degC)")
    ax.set_ylabel("depth (m)")
    ax.set_title("Mean profile\n(should show a warm mixed layer\nand a sharp thermocline)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "mean_profile.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)
    target.close()

    return written


def run(cfg: Config, make_figures: bool = True) -> dict:
    """Full QC pass: report plus figures."""
    report = {
        "coverage": coverage(cfg),
        "fields": field_report(cfg),
    }
    if make_figures:
        report["figures"] = [str(p) for p in figures(cfg)]

    dest = cfg.reports_dir / "qc_report.json"
    dest.write_text(json.dumps(report, indent=2, default=str))
    report["_report_path"] = str(dest)
    return report
