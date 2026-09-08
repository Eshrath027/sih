"""Stage 2: turn scattered raw downloads into one clean file per day.

Input:  raw files in whatever grid, naming and layout each product ships with.
Output: data/interim/daily/YYYY-MM-DD.nc - every input variable and the
        target, canonical names, identical 100x240 grid, consistent units,
        one shared land mask.

Doing this per-day, to individual files, is deliberate. Downloads and
processing get interrupted; a day that is already written is skipped, so
progress is never lost. It also means a single bad day can be inspected,
deleted and rebuilt without touching anything else.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from .config import Config
from . import grid
from .sources import cmems, oisst, synthetic

# Physically possible ranges. Anything outside these means the wrong variable,
# the wrong units, or a corrupt file - all of which are far better caught here
# than after they have been averaged into a training set.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "sst": (-2.5, 40.0),      # degC
    "sla": (-2.0, 2.0),       # m
    "sss": (20.0, 42.0),      # psu
    "uo": (-4.0, 4.0),        # m/s
    "vo": (-4.0, 4.0),        # m/s
    "uwnd": (-60.0, 60.0),    # m/s
    "vwnd": (-60.0, 60.0),    # m/s
    "thetao": (-2.5, 40.0),   # degC
}


def daily_path(cfg: Config, day: date) -> Path:
    d = cfg.interim_dir / "daily"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.isoformat()}.nc"


def check_units(ds: xr.Dataset, day: date) -> list[str]:
    """Flag variables whose values fall outside physical plausibility.

    Returns warnings rather than raising: one odd day should not halt a
    multi-year build, but it must be visible in the QC report.
    """
    problems = []
    for var, (lo, hi) in VALID_RANGES.items():
        if var not in ds:
            continue
        values = ds[var].values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            problems.append(f"{day}: {var} is entirely missing")
            continue
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin < lo or vmax > hi:
            problems.append(
                f"{day}: {var} outside plausible range "
                f"[{lo}, {hi}] - found [{vmin:.3f}, {vmax:.3f}]"
            )
    return problems


def build_day(
    cfg: Config,
    day: date,
    use_synthetic: bool = False,
    include_target: bool = True,
) -> tuple[xr.Dataset, list[str]]:
    """Assemble every variable for one day onto the canonical grid."""
    wanted = cfg.input_variables
    pieces: list[xr.Dataset] = []

    if use_synthetic:
        pieces.append(synthetic.open_day(cfg, day, wanted))
        if include_target:
            pieces.append(synthetic.open_day_target(cfg, day))
    else:
        # SST comes from NOAA and needs no credentials.
        if "sst" in wanted:
            pieces.append(oisst.open_day(cfg, day))

        # Everything else is a CMEMS group. Fetch each group once even when
        # it supplies two variables (currents, winds).
        groups = []
        for var in wanted:
            if var == "sst":
                continue
            group = cmems.group_for(var)
            if group not in groups:
                groups.append(group)

        for group in groups:
            pieces.append(cmems.open_day(cfg, day, group))

        if include_target:
            raw_target = cmems.open_day(cfg, day, "glorys")
            # Build a NEW dataset from the interpolated array rather than
            # assigning back into the old one. Assignment would align the
            # 15-level result against the dataset's existing 36 native levels,
            # quietly leaving the original axis in place and filling the new
            # depths with NaN - which then poisons the merge below.
            thetao = grid.interp_depth(raw_target["thetao"], cfg.depths)
            pieces.append(thetao.to_dataset(name="thetao"))

    # join="exact" would be too strict (the target legitimately adds a depth
    # axis the surface pieces lack), but every piece must already agree on
    # lat/lon, and no surface variable may carry a depth axis by this point.
    for piece in pieces:
        stray = [
            str(v) for v in piece.data_vars
            if "depth" in piece[v].dims and str(v) != cfg["target"]["variable"]
        ]
        if stray:
            raise ValueError(
                f"surface variables {stray} still carry a depth axis; "
                "they would be broadcast across every depth level in the merge"
            )

    ds = xr.merge(pieces, combine_attrs="drop_conflicts")

    # One land mask, shared by every variable. Cells that any surface input
    # calls land are land everywhere, so a model never sees a cell that has
    # temperature but no salinity.
    surface_vars = [v for v in wanted if v in ds]
    if surface_vars:
        mask = np.ones(grid.grid_shape(cfg), dtype=bool)
        for var in surface_vars:
            mask &= np.isfinite(ds[var].values)
        ds["ocean_mask"] = (("lat", "lon"), mask)
        ds["ocean_mask"].attrs["long_name"] = "ocean cells common to all surface inputs"

    ds = ds.expand_dims(time=[np.datetime64(day, "ns")])
    ds.attrs.update(
        title="OceanEmbed harmonized daily fields, North Indian Ocean",
        resolution="0.25 degree",
        domain=f"{cfg['domain']['lat_min']}-{cfg['domain']['lat_max']}N, "
               f"{cfg['domain']['lon_min']}-{cfg['domain']['lon_max']}E",
        synthetic="yes" if use_synthetic else "no",
    )

    return ds, check_units(ds, day)


def write_day(
    cfg: Config,
    day: date,
    use_synthetic: bool = False,
    include_target: bool = True,
    force: bool = False,
) -> tuple[Path, list[str]]:
    """Build and save one harmonized day. Skips days already written."""
    dest = daily_path(cfg, day)
    if dest.exists() and not force:
        return dest, []

    ds, problems = build_day(cfg, day, use_synthetic, include_target)

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds.data_vars
        if ds[var].dtype.kind == "f"
    }

    tmp = dest.with_suffix(".part.nc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(dest)

    return dest, problems


def harmonize_range(
    cfg: Config,
    days: list[date],
    use_synthetic: bool = False,
    include_target: bool = True,
    force: bool = False,
    progress: bool = True,
) -> dict[str, object]:
    """Harmonize a list of days, tolerating individual failures.

    A missing day in a multi-year satellite record is normal. Recording it and
    continuing is far more useful than aborting the whole build.
    """
    written, skipped, failed = [], [], {}
    problems: list[str] = []

    iterator = days
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(days, desc="harmonizing", unit="day")
        except ImportError:
            pass

    for day in iterator:
        try:
            path = daily_path(cfg, day)
            if path.exists() and not force:
                skipped.append(day)
                continue
            _, day_problems = write_day(cfg, day, use_synthetic, include_target, force)
            written.append(day)
            problems.extend(day_problems)
        except Exception as exc:
            failed[day.isoformat()] = f"{type(exc).__name__}: {exc}"

    return {
        "written": len(written),
        "skipped": len(skipped),
        "failed": failed,
        "problems": problems,
    }
