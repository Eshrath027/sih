"""The canonical grid, and everything that puts data onto it.

Every dataset in this project arrives on a different grid: OISST is already
0.25 degrees, GLORYS is 1/12 degree, the wind product is 0.125 degree. Nothing
downstream can stack them into one array until they share a grid, so this
module defines that single target grid and the operations that map onto it.

The target grid is deliberately chosen to match NOAA OISST's native cell
centres (x.125, x.375, x.625, x.875), which means sea surface temperature -
the most important input - passes through untouched with no interpolation
error at all.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .config import Config


# --------------------------------------------------------------- the grid

def target_coords(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude cell centres of the canonical grid.

    Cells are defined by their edges; the coordinate reported is the centre,
    which is why each value sits half a resolution step inside the bound.
    For the default configuration this yields 100 latitudes (5.125 .. 29.875)
    and 240 longitudes (45.125 .. 104.875).
    """
    d = cfg["domain"]
    res = float(d["resolution"])
    lat = np.arange(d["lat_min"] + res / 2, d["lat_max"], res)
    lon = np.arange(d["lon_min"] + res / 2, d["lon_max"], res)
    return lat.astype("float64"), lon.astype("float64")


def target_grid(cfg: Config) -> xr.Dataset:
    """An empty Dataset carrying only the canonical coordinates.

    Useful as a reindex/interp target and as the thing every harmonized file
    is checked against.
    """
    lat, lon = target_coords(cfg)
    return xr.Dataset(
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )


def grid_shape(cfg: Config) -> tuple[int, int]:
    lat, lon = target_coords(cfg)
    return len(lat), len(lon)


# ------------------------------------------------------- coordinate tidying

def standardize_coords(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename assorted coordinate spellings to 'lat'/'lon'/'time'/'depth'.

    Ocean products are inconsistent about this: latitude/nav_lat/y/lat all
    appear in the wild. Normalising here means the rest of the pipeline only
    ever deals with one spelling.
    """
    renames = {}
    for name in list(ds.coords) + list(getattr(ds, "dims", [])):
        low = str(name).lower()
        if low in ("latitude", "nav_lat", "y") and "lat" not in ds.coords:
            renames[name] = "lat"
        elif low in ("longitude", "nav_lon", "x") and "lon" not in ds.coords:
            renames[name] = "lon"
        elif low in ("t", "time_counter") and "time" not in ds.coords:
            renames[name] = "time"
        elif low in ("deptht", "lev", "level", "z", "zlev") and "depth" not in ds.coords:
            renames[name] = "depth"
    if renames:
        ds = ds.rename(renames)

    # Latitude sometimes arrives north-to-south. Force ascending so slicing
    # and binning behave predictably everywhere else.
    if "lat" in ds.coords and ds.sizes.get("lat", 0) > 1:
        if float(ds.lat[0]) > float(ds.lat[-1]):
            ds = ds.isel(lat=slice(None, None, -1))

    # Longitude conventions differ (0..360 vs -180..180). The North Indian
    # Ocean domain (45E..105E) is positive in both, so we only need to make
    # sure negative-convention files are converted before selection.
    if "lon" in ds.coords:
        lon = ds.lon.values
        if lon.size and float(np.nanmin(lon)) < 0:
            ds = ds.assign_coords(lon=(lon % 360))
            ds = ds.sortby("lon")

    return ds


def subset_domain(ds: xr.Dataset | xr.DataArray, cfg: Config, pad: float = 0.5):
    """Cut out the North Indian Ocean box, with a small margin.

    The margin matters: interpolating onto the target grid needs source cells
    slightly beyond the boundary, otherwise the outermost row and column come
    back as NaN.
    """
    d = cfg["domain"]
    return ds.sel(
        lat=slice(d["lat_min"] - pad, d["lat_max"] + pad),
        lon=slice(d["lon_min"] - pad, d["lon_max"] + pad),
    )


# ------------------------------------------------------------- regridding

def _source_resolution(coord: np.ndarray) -> float:
    """Median spacing of a coordinate axis, in degrees."""
    if coord.size < 2:
        return float("inf")
    return float(np.median(np.abs(np.diff(coord))))


def regrid(da: xr.DataArray, cfg: Config, method: str = "auto") -> xr.DataArray:
    """Put a DataArray onto the canonical grid.

    method='auto' picks the physically correct operation by comparing source
    and target resolution:

      finer than target  -> conservative area-average ('conservative')
          Correct for downsampling. Averaging every fine cell inside a coarse
          cell conserves the quantity and, critically, ignores land cells
          rather than smearing them into the ocean. Bilinear interpolation
          would instead sample a few points and throw the rest away.

      same as target     -> direct reindex ('nearest')
          A no-op when the grids already align, as with OISST.

      coarser than target -> bilinear interpolation ('linear')
          The only sane choice when upsampling.
    """
    da = standardize_coords(da)
    tgt_lat, tgt_lon = target_coords(cfg)
    res = float(cfg["domain"]["resolution"])

    src_res = max(
        _source_resolution(da.lat.values),
        _source_resolution(da.lon.values),
    )

    if method == "auto":
        if src_res < res * 0.9:
            method = "conservative"
        elif src_res < res * 1.1:
            method = "nearest"
        else:
            method = "linear"

    if method == "conservative":
        out = _regrid_conservative(da, tgt_lat, tgt_lon, res)
    elif method == "nearest":
        # Grids nominally match; snap to the canonical coordinates so tiny
        # floating-point differences do not stop arrays concatenating later.
        out = da.reindex(lat=tgt_lat, lon=tgt_lon, method="nearest", tolerance=res / 2)
    elif method == "linear":
        out = da.interp(lat=tgt_lat, lon=tgt_lon, method="linear")
    else:
        raise ValueError(f"unknown regrid method: {method!r}")

    out = out.assign_coords(lat=tgt_lat, lon=tgt_lon)
    out.attrs = dict(da.attrs)
    out.attrs["regrid_method"] = method
    return out


def _regrid_conservative(
    da: xr.DataArray, tgt_lat: np.ndarray, tgt_lon: np.ndarray, res: float
) -> xr.DataArray:
    """Area-average fine source cells into coarse target cells.

    Each source cell is assigned to the target cell containing its centre,
    then averaged. NaN (land) cells are excluded from both the sum and the
    count, so a coastal target cell reflects only the ocean inside it. A
    target cell with no ocean source cells comes back as NaN, which is the
    correct answer for a cell that is entirely land.
    """
    src_lat = da.lat.values
    src_lon = da.lon.values
    n_lat, n_lon = len(tgt_lat), len(tgt_lon)

    # Grid origin is the outer edge of the first cell, half a step below the
    # first centre.
    lat0 = tgt_lat[0] - res / 2
    lon0 = tgt_lon[0] - res / 2

    i_lat = np.floor((src_lat - lat0) / res).astype(np.int64)
    i_lon = np.floor((src_lon - lon0) / res).astype(np.int64)

    keep_lat = (i_lat >= 0) & (i_lat < n_lat)
    keep_lon = (i_lon >= 0) & (i_lon < n_lon)
    if not keep_lat.any() or not keep_lon.any():
        raise ValueError(
            "Source grid does not overlap the target domain. "
            "Check the longitude convention and the subset bounds."
        )

    da = da.isel(lat=np.where(keep_lat)[0], lon=np.where(keep_lon)[0])
    i_lat = i_lat[keep_lat]
    i_lon = i_lon[keep_lon]

    # Move lat/lon to the trailing axes so the leading axes (time, depth) can
    # be flattened and looped over cheaply.
    other_dims = [d for d in da.dims if d not in ("lat", "lon")]
    da = da.transpose(*other_dims, "lat", "lon")
    values = np.asarray(da.values, dtype="float64")

    lead_shape = values.shape[: len(other_dims)]
    n_lead = int(np.prod(lead_shape)) if lead_shape else 1
    flat = values.reshape(n_lead, len(i_lat), len(i_lon))

    # Flat index of the target cell each (source lat, source lon) falls into.
    bin_index = (i_lat[:, None] * n_lon + i_lon[None, :]).ravel()

    out = np.full((n_lead, n_lat * n_lon), np.nan, dtype="float64")
    for k in range(n_lead):
        plane = flat[k].ravel()
        valid = np.isfinite(plane)
        total = np.bincount(
            bin_index, weights=np.where(valid, plane, 0.0), minlength=n_lat * n_lon
        )
        count = np.bincount(
            bin_index, weights=valid.astype("float64"), minlength=n_lat * n_lon
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(count > 0, total / count, np.nan)

    out = out.reshape(*lead_shape, n_lat, n_lon)
    coords = {d: da.coords[d] for d in other_dims if d in da.coords}
    coords["lat"] = tgt_lat
    coords["lon"] = tgt_lon
    return xr.DataArray(
        out.astype("float32"),
        dims=(*other_dims, "lat", "lon"),
        coords=coords,
        name=da.name,
    )


# ------------------------------------------------------------ depth levels

def interp_depth(da: xr.DataArray, depths: list[float]) -> xr.DataArray:
    """Interpolate a 3D field onto the 15 required standard depth levels.

    GLORYS provides 50 unevenly spaced native levels; the problem statement
    demands a specific set of 15. Linear interpolation in depth is the
    standard treatment and is accurate because the native levels are finer
    than the target levels through the upper ocean, where temperature
    actually varies.
    """
    da = standardize_coords(da)
    if "depth" not in da.dims:
        raise ValueError(f"expected a 'depth' dimension, found dims {da.dims}")

    da = da.sortby("depth")
    targets = np.asarray(depths, dtype="float64")

    # The shallowest native level sits a metre or two below the surface,
    # so requesting 0 m would extrapolate to NaN. Clamping to the shallowest
    # available level treats the top level as the surface value, which is the
    # usual convention.
    shallowest = float(da.depth.min())
    clamped = np.where(targets < shallowest, shallowest, targets)

    out = da.interp(depth=clamped, method="linear")
    # Report the depths that were actually requested, not the clamped ones.
    out = out.assign_coords(depth=targets)
    out.depth.attrs.update(units="m", positive="down", long_name="depth below sea surface")
    return out


# --------------------------------------------------------------- land mask

def build_land_mask(da: xr.DataArray) -> xr.DataArray:
    """Boolean ocean mask (True where there is water) from a gridded field.

    Land is stored as NaN in every product we use, so a cell that is finite
    is a cell with ocean in it. Collapsing over time guards against a single
    cloudy or missing day carving fake islands into the mask.
    """
    dims = [d for d in da.dims if d not in ("lat", "lon")]
    finite = np.isfinite(da)
    if dims:
        finite = finite.any(dim=dims)
    return finite.rename("ocean_mask")
