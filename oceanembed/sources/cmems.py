"""Copernicus Marine (CMEMS) - sea level, salinity, currents, winds, GLORYS.

Six of the eight inputs plus the training target all live behind one free
Copernicus Marine account, which is why the whole pipeline needs only a single
registration.

The key reason to use their toolbox rather than plain HTTP: it performs the
subset on THEIR server. We ask for the North Indian Ocean box and the dates we
want, and receive only that. GLORYS in particular is a global 1/12-degree
daily reanalysis with 50 depth levels - many terabytes in full - and we pull
a few gigabytes of it.

Downloads are chunked by year rather than by day. One request per year per
variable group is dramatically faster than 2,192 separate requests, and each
yearly file is small enough that losing one to an interruption is cheap.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import xarray as xr

from ..config import Config, load_credentials
from .base import (
    already_have,
    select_day,
    squeeze_singletons,
    take_surface_level,
    to_canonical,
)

name = "cmems"


class CredentialsMissing(RuntimeError):
    """Raised when a CMEMS download is attempted without a login configured."""


# Canonical variable naming. CMEMS product variable names on the left, the
# names this project uses on the right.
RENAMES: dict[str, dict[str, str]] = {
    "sla": {"sla": "sla"},
    "sss": {"sos": "sss"},
    "currents": {"uo": "uo", "vo": "vo"},
    "winds": {"eastward_wind": "uwnd", "northward_wind": "vwnd"},
    "glorys": {"thetao": "thetao"},
}

# Which canonical variables each source group provides. Used to work out
# which groups need fetching for a given set of requested variables.
PROVIDES: dict[str, list[str]] = {
    "sla": ["sla"],
    "sss": ["sss"],
    "currents": ["uo", "vo"],
    "winds": ["uwnd", "vwnd"],
    "glorys": ["thetao"],
}


def group_for(variable: str) -> str:
    """Which download group supplies a canonical variable."""
    for group, provided in PROVIDES.items():
        if variable in provided:
            return group
    raise KeyError(f"no CMEMS group provides {variable!r}")


def check_credentials() -> tuple[str, str]:
    """Return the CMEMS username and password, or explain how to set them."""
    user, password = load_credentials()
    if not user or not password:
        raise CredentialsMissing(
            "Copernicus Marine credentials not found.\n\n"
            "  1. Register (free): https://data.marine.copernicus.eu/register\n"
            "  2. Create a file named .env in the project root containing:\n\n"
            "       COPERNICUSMARINE_SERVICE_USERNAME=your_username\n"
            "       COPERNICUSMARINE_SERVICE_PASSWORD=your_password\n\n"
            "Until then, run with --synthetic to exercise the pipeline on\n"
            "stand-in data of the correct shape and units."
        )
    return user, password


# Longest span put in a single file. Keeps any one download small enough that
# losing it to a dropped connection is cheap, and keeps files openable without
# a lot of memory.
MAX_DAYS_PER_FILE = 92


def local_path(cfg: Config, group: str, start: date, end: date) -> Path:
    d = cfg.raw_source_dir(group)
    return d / f"{group}_{start:%Y%m%d}_{end:%Y%m%d}.nc"


def plan_chunks(days: list[date]) -> list[tuple[date, date]]:
    """Group the wanted days into contiguous download ranges.

    This matters enormously when days are subsampled. Requesting one file per
    calendar year would span January to December regardless of how few days in
    between are actually wanted - with --every 30 that downloads all 365 days
    to keep 12, roughly a thirtyfold waste. GLORYS at 18 MB/day turns a 3 GB
    job into a 100 GB one.

    So instead: find runs of consecutive days and request exactly those. Dense
    selections still collapse into a few large ranges; sparse ones become many
    small requests, which is the correct trade.
    """
    if not days:
        return []

    ordered = sorted(days)
    runs: list[list[date]] = [[ordered[0]]]
    for day in ordered[1:]:
        if (day - runs[-1][-1]).days == 1:
            runs[-1].append(day)
        else:
            runs.append([day])

    chunks: list[tuple[date, date]] = []
    for run in runs:
        for i in range(0, len(run), MAX_DAYS_PER_FILE):
            block = run[i : i + MAX_DAYS_PER_FILE]
            chunks.append((block[0], block[-1]))
    return chunks


def find_file(cfg: Config, group: str, day: date) -> Path | None:
    """Locate the downloaded file whose date range contains `day`."""
    for path in sorted(cfg.raw_source_dir(group).glob(f"{group}_*_*.nc")):
        try:
            _, lo, hi = path.stem.rsplit("_", 2)
            start = datetime.strptime(lo, "%Y%m%d").date()
            end = datetime.strptime(hi, "%Y%m%d").date()
        except ValueError:
            continue
        if start <= day <= end:
            return path
    return None


def fetch(
    cfg: Config,
    days: list[date],
    group: str,
    force: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Download exactly the days requested, grouped into contiguous ranges."""
    import copernicusmarine

    user, password = check_credentials()
    spec = cfg["sources"][group]
    domain = cfg["domain"]

    written: list[Path] = []
    chunks = plan_chunks(days)

    for start, end in chunks:
        dest = local_path(cfg, group, start, end)
        if already_have(dest) and not force:
            written.append(dest)
            continue

        kwargs = dict(
            dataset_id=spec["dataset_id"],
            variables=list(spec["variables"]),
            # A small margin so regridding has source cells just beyond the
            # boundary and does not produce a NaN edge.
            minimum_longitude=domain["lon_min"] - 0.5,
            maximum_longitude=domain["lon_max"] + 0.5,
            minimum_latitude=domain["lat_min"] - 0.5,
            maximum_latitude=domain["lat_max"] + 0.5,
            start_datetime=f"{start}T00:00:00",
            end_datetime=f"{end}T23:59:59",
            output_filename=dest.name,
            output_directory=str(dest.parent),
            username=user,
            password=password,
            overwrite=force,
            dry_run=dry_run,
        )

        # Only the 3D target needs a depth range; asking for depth on a
        # surface-only product is an error.
        if group == "glorys":
            kwargs["minimum_depth"] = 0.0
            # A little beyond 1000 m so the deepest level interpolates rather
            # than extrapolating off the end of the column.
            kwargs["maximum_depth"] = 1200.0

        copernicusmarine.subset(**kwargs)
        written.append(dest)

    return written


def open_day(cfg: Config, day: date, group: str) -> xr.Dataset:
    """One day of one CMEMS variable group, on the canonical grid."""
    path = find_file(cfg, group, day)
    if path is None:
        raise FileNotFoundError(
            f"No downloaded {group} file covers {day}.\n"
            f"Run: oceanembed fetch --source {group}"
        )

    with xr.open_dataset(path) as ds:
        ds = select_day(ds, day)
        if group != "glorys":
            # Surface groups must arrive strictly 2D. Only the target keeps a
            # depth axis; see take_surface_level for why this matters.
            ds = take_surface_level(squeeze_singletons(ds))
        out = to_canonical(ds, cfg, rename=RENAMES[group])
        return out.load()
