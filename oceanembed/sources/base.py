"""Common interface every data source implements.

A source knows two things: how to download its raw files, and how to hand
back one day of data using the project's canonical variable names, already
placed on the canonical grid. Nothing above this layer needs to know that
OISST ships one file per day over plain HTTP while GLORYS is an authenticated
server-side subset of a 1/12-degree reanalysis.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import xarray as xr

from ..config import Config
from .. import grid


class Source(Protocol):
    """What every source module must provide."""

    name: str

    def fetch(self, cfg: Config, days: list[date], force: bool = False) -> list[Path]:
        """Download whatever raw files are needed to cover `days`.

        Must be idempotent: already-present, valid files are skipped so an
        interrupted download can simply be re-run.
        """
        ...

    def open_day(self, cfg: Config, day: date) -> xr.Dataset:
        """One day of data, canonical names, on the canonical grid."""
        ...


# ----------------------------------------------------------------- helpers

def select_day(ds: xr.Dataset, day: date) -> xr.Dataset:
    """Pull a single day out of a dataset that may span a longer period.

    Products differ in how they timestamp a daily average - midnight, midday,
    or the end of the day - so this matches on the calendar date rather than
    an exact timestamp. Sub-daily data (the hourly wind product) is averaged
    to a daily mean, which is what the pipeline standardises on.
    """
    if "time" not in ds.dims and "time" not in ds.coords:
        return ds

    stamps = ds.time.values.astype("datetime64[D]")
    wanted = np.datetime64(day, "D")
    hits = np.where(stamps == wanted)[0]
    if hits.size == 0:
        raise KeyError(f"no time step for {day} in this file")

    sel = ds.isel(time=hits)
    if sel.sizes["time"] > 1:
        # Hourly or multi-step data: collapse to the daily mean.
        sel = sel.mean(dim="time", keep_attrs=True)
    else:
        sel = sel.isel(time=0)
    return sel.drop_vars("time", errors="ignore")


def to_canonical(
    ds: xr.Dataset,
    cfg: Config,
    rename: dict[str, str],
    method: str = "auto",
) -> xr.Dataset:
    """Rename source variables, subset the domain, and regrid.

    `rename` maps the product's own variable names onto the canonical names
    used everywhere downstream (sst, sla, sss, uo, vo, uwnd, vwnd, thetao).
    """
    ds = grid.standardize_coords(ds)

    present = {src: dst for src, dst in rename.items() if src in ds.variables}
    missing = set(rename) - set(present)
    if missing:
        raise KeyError(
            f"variables {sorted(missing)} not found in source file; "
            f"available: {sorted(map(str, ds.data_vars))}"
        )
    ds = ds[list(present)].rename(present)

    ds = grid.subset_domain(ds, cfg)

    out = xr.Dataset(attrs=ds.attrs)
    for name in ds.data_vars:
        out[name] = grid.regrid(ds[name], cfg, method=method)
    return out


def squeeze_singletons(ds: xr.Dataset) -> xr.Dataset:
    """Drop length-1 dimensions that carry no information.

    Surface products routinely ship a depth axis of length 1 holding the value
    0, because the surface is nominally a depth: OISST calls it `zlev`, the
    CMEMS salinity product calls it `depth`. Either way it is noise here.

    Dropping a singleton `depth` matters more than it looks. If salinity keeps
    a 1-element depth axis, merging it with GLORYS - which has a real 15-level
    depth axis - broadcasts every surface field across all 15 levels, silently
    inflating the dataset and corrupting the merge. Only lat and lon are
    protected, since a genuinely 1-cell grid would be a different bug.

    The 3D target never passes through here, so its depth axis is safe.
    """
    for dim in list(ds.dims):
        if dim in ("lat", "lon"):
            continue
        if ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
    return ds


def take_surface_level(ds: xr.Dataset) -> xr.Dataset:
    """Reduce a surface product to a single, shallowest level.

    Surface products are not reliably two-dimensional. The CMEMS current
    product ships two levels, 0 m and 15 m, because near-surface shear
    matters to its users; the salinity product ships a single dummy level at
    0 m. Neither should reach the pipeline with a depth axis.

    Left alone, this is a silent corruption rather than an error: merging a
    current field that still has a depth axis against the 3D temperature
    target makes xarray union the two sets of depth values, and every surface
    variable is then broadcast across levels it was never measured at.

    Only the target keeps its depth axis, and the target never comes through
    this function.
    """
    if "depth" not in ds.dims and "depth" not in ds.coords:
        return ds
    if "depth" in ds.dims:
        ds = ds.sortby("depth").isel(depth=0)
    return ds.drop_vars("depth", errors="ignore")


def already_have(path: Path, min_bytes: int = 1024) -> bool:
    """Whether a raw file exists and is plausibly complete.

    Guards against the classic interrupted-download case where a zero-byte or
    truncated file is left behind and silently poisons the pipeline.
    """
    return path.exists() and path.stat().st_size >= min_bytes
