"""NOAA OISST v2.1 - daily sea surface temperature.

The one input that needs no login. NOAA serves one global NetCDF per day over
plain HTTPS at exactly the 0.25-degree resolution the problem statement asks
for, on cell centres that our canonical grid was chosen to match. So SST
reaches the model with no interpolation of any kind.

Product: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import requests
import xarray as xr

from ..config import Config
from .base import already_have, select_day, squeeze_singletons, to_canonical

name = "oisst"

# Files older than ~2 weeks are 'final'; very recent days are served as a
# preliminary version under a different filename.
_PRELIM_SUFFIX = "_preliminary"

_TIMEOUT = 120
_RETRIES = 3
_BACKOFF = 5.0


def url_for(cfg: Config, day: date, preliminary: bool = False) -> str:
    template = cfg["sources"]["sst"]["url_template"]
    url = template.format(ym=day.strftime("%Y%m"), ymd=day.strftime("%Y%m%d"))
    if preliminary:
        url = url.replace(".nc", f"{_PRELIM_SUFFIX}.nc")
    return url


def local_path(cfg: Config, day: date) -> Path:
    d = cfg.raw_source_dir(name) / day.strftime("%Y")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"oisst-avhrr-v02r01.{day.strftime('%Y%m%d')}.nc"


def fetch(cfg: Config, days: list[date], force: bool = False) -> list[Path]:
    """Download the daily OISST files covering `days`.

    Skips files already on disk, so an interrupted run is resumed simply by
    running the command again.
    """
    written: list[Path] = []
    session = requests.Session()

    for day in days:
        dest = local_path(cfg, day)
        if already_have(dest) and not force:
            written.append(dest)
            continue

        ok = False
        # Try the final file first, then the preliminary name for recent days.
        for preliminary in (False, True):
            url = url_for(cfg, day, preliminary=preliminary)
            if _download(session, url, dest):
                ok = True
                break

        if not ok:
            raise RuntimeError(f"could not download OISST for {day} (tried final and preliminary)")
        written.append(dest)

    return written


def _download(session: requests.Session, url: str, dest: Path) -> bool:
    """Stream one file to disk. Returns False on a clean 404, raises on worse."""
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(_RETRIES):
        try:
            with session.get(url, stream=True, timeout=_TIMEOUT) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            # Rename only after a complete write, so a partial file is never
            # mistaken for a finished download.
            tmp.replace(dest)
            return True
        except requests.RequestException:
            tmp.unlink(missing_ok=True)
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_BACKOFF * (attempt + 1))

    return False


def open_day(cfg: Config, day: date) -> xr.Dataset:
    """Sea surface temperature for one day, on the canonical grid."""
    path = local_path(cfg, day)
    if not path.exists():
        raise FileNotFoundError(f"OISST file missing for {day}: {path}\nRun: oceanembed fetch --source sst")

    with xr.open_dataset(path) as ds:
        ds = squeeze_singletons(select_day(ds, day))
        out = to_canonical(ds, cfg, rename={"sst": "sst"})
        out["sst"].attrs.update(units="degC", long_name="sea surface temperature", source="NOAA OISST v2.1")
        return out.load()
