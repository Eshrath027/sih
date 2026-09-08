"""Show every dataset in its RAW form, then show them combined.

show_one_row.py shows the finished row. This shows where each of those numbers
came from: the actual file, the name the variable has inside it, the grid it
arrived on, and the value before and after regridding.

Run:  .venv/bin/python scripts/show_all_sources.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oceanembed.config import load_config, date_range   # noqa: E402
from oceanembed import grid                             # noqa: E402
from oceanembed.sources import cmems, oisst             # noqa: E402

# Each source: the group name, which canonical variables it supplies, and a
# one-line description of what it physically measures.
SOURCES = [
    ("oisst",    ["sst"],           "NOAA satellite infrared thermometer"),
    ("sla",      ["sla"],           "Satellite radar altimeter (sea surface height)"),
    ("sss",      ["sss"],           "Salinity, blended satellite and in-situ"),
    ("currents", ["uo", "vo"],      "Surface currents, from height and wind"),
    ("winds",    ["uwnd", "vwnd"],  "Scatterometer winds at 10 m"),
    ("glorys",   ["thetao"],        "GLORYS reanalysis - THE ANSWER KEY"),
]

# The name each variable carries INSIDE the downloaded file, which is not the
# name this project uses for it.
NATIVE_NAME = {
    "sst": "sst", "sla": "sla", "sss": "sos",
    "uo": "uo", "vo": "vo",
    "uwnd": "eastward_wind", "vwnd": "northward_wind",
    "thetao": "thetao",
}


def spacing(values: np.ndarray) -> float:
    return float(np.median(np.abs(np.diff(values)))) if values.size > 1 else float("nan")


def raw_file_for(cfg, group: str, day: date) -> Path | None:
    if group == "oisst":
        p = oisst.local_path(cfg, day)
        return p if p.exists() else None
    return cmems.find_file(cfg, group, day)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=15.125)
    ap.add_argument("--lon", type=float, default=68.125)
    ap.add_argument("--every", type=int, default=112)
    args = ap.parse_args()

    cfg = load_config()

    day = None
    for d in date_range(cfg.start, cfg.end)[:: args.every]:
        if oisst.local_path(cfg, d).exists():
            day = d
            break
    if day is None:
        print("Nothing downloaded yet.", file=sys.stderr)
        return 1

    print(f"\n{'=' * 78}")
    print(f"  EVERY DATASET, SEPARATELY  ->  THEN COMBINED")
    print(f"  day {day}   at {args.lat}N, {args.lon}E")
    print(f"{'=' * 78}")

    combined: dict[str, float] = {}
    profile = None

    for group, provides, description in SOURCES:
        print(f"\n{'-' * 78}")
        print(f"  {group.upper()}   {description}")
        print(f"{'-' * 78}")

        path = raw_file_for(cfg, group, day)
        if path is None:
            print(f"  not downloaded yet  (provides: {', '.join(provides)})")
            continue

        with xr.open_dataset(path) as ds:
            print(f"  file          {path.name}")
            print(f"  size          {path.stat().st_size / 1e6:.2f} MB")
            print(f"  variables in  {sorted(str(v) for v in ds.data_vars)}")

            std = grid.standardize_coords(ds)
            lat_step = spacing(std.lat.values)
            lon_step = spacing(std.lon.values)
            print(f"  its own grid  {std.sizes.get('lat')} x {std.sizes.get('lon')} "
                  f"cells, spacing {lat_step:.4f} deg")
            if "depth" in std.dims:
                print(f"  depth levels  {std.sizes['depth']} "
                      f"(from {float(std.depth.min()):.1f} to {float(std.depth.max()):.1f} m)")
            if "time" in std.dims:
                print(f"  time steps    {std.sizes['time']}")

            # Value BEFORE regridding: nearest cell on the product's own grid.
            for var in provides:
                native = NATIVE_NAME[var]
                if native not in std:
                    continue
                da = std[native]
                while "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                probe = da
                if "depth" in probe.dims:
                    probe = probe.isel(depth=0, drop=True)
                near = probe.sel(lat=args.lat, lon=args.lon, method="nearest")
                got_lat = float(near.lat)
                got_lon = float(near.lon)
                print(f"\n  '{native}' before regridding:")
                print(f"     nearest cell is at {got_lat:.4f}N, {got_lon:.4f}E "
                      f"(off by {abs(got_lat-args.lat):.4f}, {abs(got_lon-args.lon):.4f} deg)")
                print(f"     value = {float(near):.4f}")

        # Value AFTER regridding, on the canonical grid.
        try:
            if group == "oisst":
                out = oisst.open_day(cfg, day)
            else:
                out = cmems.open_day(cfg, day, group)
                if group == "glorys":
                    out["thetao"] = grid.interp_depth(out["thetao"], cfg.depths)

            for var in provides:
                if var not in out:
                    continue
                da = out[var]
                if "depth" in da.dims:
                    profile = da.sel(lat=args.lat, lon=args.lon, method="nearest").values
                    print(f"\n  '{var}' after regridding onto the 100x240 grid:")
                    print(f"     exact cell {args.lat}N, {args.lon}E, "
                          f"{len(cfg.depths)} depths -> {profile[0]:.3f} C at the surface")
                else:
                    value = float(da.sel(lat=args.lat, lon=args.lon, method="nearest"))
                    combined[var] = value
                    print(f"\n  '{var}' after regridding onto the 100x240 grid:")
                    print(f"     exact cell {args.lat}N, {args.lon}E   value = {value:.4f}")
        except Exception as exc:
            print(f"  (could not regrid: {type(exc).__name__})")

    # ---------------------------------------------------------- combined
    print(f"\n{'=' * 78}")
    print("  COMBINED  -  all of the above, one row, one grid")
    print(f"{'=' * 78}\n")

    print(f"  {'variable':<8} {'value':>11}   came from")
    print("  " + "-" * 60)
    origin = {"sst": "oisst", "sla": "sla", "sss": "sss",
              "uo": "currents", "vo": "currents",
              "uwnd": "winds", "vwnd": "winds"}
    for var in cfg.input_variables:
        if var in combined:
            print(f"  {var:<8} {combined[var]:>11.4f}   {origin[var]}")
        else:
            print(f"  {var:<8} {'--':>11}   {origin[var]} (not downloaded yet)")

    if profile is not None:
        print(f"\n  thetao   15 depths from glorys:")
        print("     " + "  ".join(f"{float(t):.2f}" for t in profile))
    else:
        print(f"\n  thetao   -- (glorys not downloaded yet)")

    ready = len(combined)
    print(f"\n  {ready} of {len(cfg.input_variables)} inputs ready.")
    print(f"""
  THE POINT:
    Six separate files, six different grids, six different variable names.
    After regridding they all sit on the SAME 100 x 240 grid, so the value
    at row i column j means the same place in every one of them.
    That is what makes them stackable into a single array.
""")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
