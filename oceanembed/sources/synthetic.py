"""Physically-motivated stand-in data, for building and testing the pipeline.

Purpose. Six of the eight inputs and the training target sit behind a
Copernicus Marine login. Rather than leave every downstream stage untested
until that account exists, this module generates fields with the correct
shape, units, coordinates and land mask, so `build-cube`, `stats`, `qc` and
`pack` can all be run and debugged immediately. Swapping in the real
downloads later changes nothing else.

It is deliberately not random noise. The fake subsurface temperature is
generated from the fake surface fields through a simplified but real physical
relationship:

  * Temperature is near-uniform through a wind-mixed surface layer, then
    decays toward cold deep water below it.
  * Stronger winds stir a deeper mixed layer.
  * A raised sea surface (positive sea level anomaly) means a thicker warm
    layer, so the thermocline sits deeper and the water at depth is warmer.

That last point is the actual physical mechanism the real model has to
discover. Encoding it here means a network trained on synthetic data SHOULD
reach a low error - so if it cannot, the bug is in the model code, not the
data. That makes this a genuine end-to-end test rather than a placeholder.

Values are reproducible: the random seed is derived from the date, so the
same day always produces the same field across runs and machines.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from ..config import Config
from .. import grid

name = "synthetic"

# Rough temperature of deep water below the thermocline, degrees C.
T_DEEP = 3.5


def _rng(day: date, salt: int = 0) -> np.random.Generator:
    """Deterministic per-day generator, so runs are reproducible."""
    return np.random.default_rng(day.toordinal() * 1000 + salt)


def _smooth_field(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth random field, standardised to zero mean and unit variance.

    Real ocean fields are spatially correlated - eddies and fronts, not
    per-pixel static - so white noise is blurred to the requested length
    scale in grid cells.
    """
    field = gaussian_filter(rng.standard_normal(shape), sigma=scale, mode="wrap")
    std = field.std()
    return field / std if std > 0 else field


def ocean_mask(cfg: Config) -> np.ndarray:
    """Land/sea mask for the canonical grid.

    Uses the real OISST land mask if any OISST file has been downloaded,
    since a realistic coastline makes every downstream masking bug visible.
    Falls back to a crude analytic approximation of the Indian subcontinent
    if no real file is available yet.
    """
    from pathlib import Path

    from ..config import ROOT

    candidates = sorted(Path(cfg.raw_source_dir("oisst")).rglob("*.nc"))
    candidates += sorted(ROOT.glob("oisst-*.nc"))

    for path in candidates:
        try:
            with xr.open_dataset(path) as ds:
                da = grid.standardize_coords(ds["sst"])
                # Drop time/zlev so only lat/lon remain.
                while da.ndim > 2:
                    da = da.isel({da.dims[0]: 0}, drop=True)
                da = grid.regrid(grid.subset_domain(da, cfg), cfg)
                return np.isfinite(da.values)
        except Exception:
            continue

    # Fallback: a rough India-shaped landmass, good enough to exercise the
    # masking logic when no real file exists.
    lat, lon = grid.target_coords(cfg)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    land = (
        ((lo > 68) & (lo < 88) & (la > 8) & (la < 30) & (la > 8 + 0.9 * (lo - 68)))
        | ((lo < 51) & (la > 10))
        | ((lo > 92) & (la > 20))
    )
    return ~land


def _seasonal(day: date) -> float:
    """Annual cycle phase, peaking in the boreal spring."""
    doy = day.timetuple().tm_yday
    return math.sin(2 * math.pi * (doy - 100) / 365.25)


def surface_fields(cfg: Config, day: date) -> dict[str, np.ndarray]:
    """All seven synthetic surface inputs for one day."""
    lat, lon = grid.target_coords(cfg)
    shape = (len(lat), len(lon))
    la = lat[:, None] * np.ones((1, len(lon)))
    season = _seasonal(day)

    r = _rng(day)

    # Warm at the equator, cooler to the north, plus a seasonal swing and
    # mesoscale structure.
    sst = (
        29.5
        - 0.16 * (la - 5.0)
        + 1.2 * season
        + 0.9 * _smooth_field(shape, 6.0, r)
    )

    # Sea level anomaly: eddies, order +/- 20 cm.
    sla = 0.10 * _smooth_field(shape, 8.0, _rng(day, 1))

    # Salinity: the Bay of Bengal is markedly fresher than the Arabian Sea
    # because of river discharge, which is the regional signal that makes
    # salinity worth including as an input at all.
    lo = lon[None, :] * np.ones((len(lat), 1))
    bay_of_bengal = 1 / (1 + np.exp(-(lo - 82) / 3.0))
    sss = 35.6 - 2.6 * bay_of_bengal - 0.35 * _smooth_field(shape, 7.0, _rng(day, 2))

    # Surface currents, order tens of cm/s.
    uo = 0.35 * _smooth_field(shape, 7.0, _rng(day, 3))
    vo = 0.28 * _smooth_field(shape, 7.0, _rng(day, 4))

    # Monsoon winds: reversing seasonally, which is the defining feature of
    # this basin.
    uwnd = 4.5 * season + 2.5 * _smooth_field(shape, 10.0, _rng(day, 5))
    vwnd = 2.0 * season + 2.5 * _smooth_field(shape, 10.0, _rng(day, 6))

    return {
        "sst": sst.astype("float32"),
        "sla": sla.astype("float32"),
        "sss": sss.astype("float32"),
        "uo": uo.astype("float32"),
        "vo": vo.astype("float32"),
        "uwnd": uwnd.astype("float32"),
        "vwnd": vwnd.astype("float32"),
    }


def subsurface_from_surface(
    cfg: Config, fields: dict[str, np.ndarray]
) -> np.ndarray:
    """Build a temperature profile at every grid point from the surface state.

    This is the simplified physics described in the module docstring, and it
    is the reason synthetic mode is a real test: the answer is genuinely
    predictable from the inputs, so a working model must be able to find it.
    """
    depths = np.asarray(cfg.depths, dtype="float64")
    sst = fields["sst"].astype("float64")
    sla = fields["sla"].astype("float64")

    wind_speed = np.hypot(fields["uwnd"], fields["vwnd"]).astype("float64")

    # Stronger wind stirs a deeper, well-mixed surface layer.
    mixed_layer = 15.0 + 3.5 * wind_speed

    # A raised sea surface means a thicker warm layer, so temperature falls
    # off more slowly with depth.
    decay_scale = np.clip(110.0 + 500.0 * sla, 40.0, 400.0)

    z = depths[:, None, None]
    below = np.clip(z - mixed_layer[None, :, :], 0.0, None)
    shape = np.exp(-below / decay_scale[None, :, :])

    temp = T_DEEP + (sst[None, :, :] - T_DEEP) * shape
    return temp.astype("float32")


def open_day(cfg: Config, day: date, variables: list[str] | None = None) -> xr.Dataset:
    """One day of synthetic surface data, on the canonical grid."""
    lat, lon = grid.target_coords(cfg)
    mask = ocean_mask(cfg)
    fields = surface_fields(cfg, day)

    wanted = variables or cfg.input_variables
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    for var in wanted:
        if var not in fields:
            continue
        values = np.where(mask, fields[var], np.nan)
        ds[var] = (("lat", "lon"), values.astype("float32"))

    ds.attrs["synthetic"] = "yes"
    return ds


def open_day_target(cfg: Config, day: date) -> xr.Dataset:
    """One day of synthetic subsurface temperature, on the canonical grid."""
    lat, lon = grid.target_coords(cfg)
    depths = np.asarray(cfg.depths, dtype="float32")
    mask = ocean_mask(cfg)

    fields = surface_fields(cfg, day)
    temp = subsurface_from_surface(cfg, fields)
    temp = np.where(mask[None, :, :], temp, np.nan)

    ds = xr.Dataset(
        {"thetao": (("depth", "lat", "lon"), temp.astype("float32"))},
        coords={"depth": depths, "lat": lat, "lon": lon},
    )
    ds["thetao"].attrs.update(units="degC", long_name="sea water potential temperature")
    ds.attrs["synthetic"] = "yes"
    return ds
