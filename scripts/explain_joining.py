"""A worked example: how separate ocean datasets become one training array.

Run:  .venv/bin/python scripts/explain_joining.py

Every dataset in this project arrives as its own file, with its own variable
name, its own grid, and its own units. Nothing can be fed to a model until
they are joined. This script shows exactly how that join works, using the real
OISST file in the project root plus stand-ins for the products that need a
login.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oceanembed.config import load_config          # noqa: E402
from oceanembed import grid                        # noqa: E402
from oceanembed.sources import synthetic           # noqa: E402

OISST_FILE = ROOT / "oisst-avhrr-v02r01.19810901.nc"

# The single grid cell we follow all the way through, in the Arabian Sea.
SPOT_LAT, SPOT_LON = 15.125, 68.125


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


# ==========================================================================
def part1_separate_files() -> None:
    rule("PART 1  -  Each parameter is a SEPARATE file with its own shape")

    print("""
Every product is downloaded independently. They do not know about each other.
Here is what each one looks like on arrival:
""")

    with xr.open_dataset(OISST_FILE) as ds:
        sst_raw = ds["sst"]
        print(f"  SST     file: {OISST_FILE.name}")
        print(f"          variable name : 'sst'")
        print(f"          dimensions    : {dict(sst_raw.sizes)}")
        print(f"          lat spacing   : {float(ds.lat[1] - ds.lat[0]):.4f} deg")
        print(f"          units         : {sst_raw.attrs.get('units')}")

    print(f"""
  SLA     file: sla_2018.nc                (Copernicus, needs login)
          variable name : 'sla'            <-- DIFFERENT NAME
          dimensions    : time, latitude, longitude
          lat spacing   : 0.1250 deg       <-- DIFFERENT GRID
          units         : m

  GLORYS  file: glorys_2018.nc             (Copernicus, needs login)
          variable name : 'thetao'         <-- DIFFERENT NAME AGAIN
          dimensions    : time, depth, latitude, longitude
          lat spacing   : 0.0833 deg       <-- DIFFERENT GRID AGAIN
          units         : degC             <-- and it has a DEPTH axis

Three files. Three variable names. Three different grids. One of them is 3D.
They cannot be stacked into an array in this state.""")


# ==========================================================================
def part2_the_join_key() -> None:
    rule("PART 2  -  What actually connects them: (time, lat, lon)")

    print("""
This is the key idea. Think of each dataset as a spreadsheet whose primary key
is a place and a date:

    date         lat      lon      sst
    2018-03-15   15.125   68.125   28.41

    date         lat      lon      sla
    2018-03-15   15.125   68.125   0.07

    date         lat      lon      depth   thetao
    2018-03-15   15.125   68.125   0       28.39
    2018-03-15   15.125   68.125   50      24.10
    2018-03-15   15.125   68.125   100     18.55

Joining them is a database JOIN on (date, lat, lon). Same place, same day,
different measurement -> one row with many columns.

The complication is that the keys DO NOT MATCH between products, because the
grids differ. That is the entire problem regridding solves.""")


# ==========================================================================
def part3_the_mismatch() -> None:
    rule("PART 3  -  Why the keys do not match, with real numbers")

    with xr.open_dataset(OISST_FILE) as ds:
        oisst_lats = ds.lat.sel(lat=slice(14.8, 15.8)).values

    # DUACS sea level sits on a 0.125 degree grid, offset from OISST's.
    sla_lats = np.arange(14.8125, 15.8, 0.125)

    print("\n  OISST latitudes near our spot (0.25 deg grid):")
    print("     " + "  ".join(f"{v:8.4f}" for v in oisst_lats))

    print("\n  Sea level latitudes over the same span (0.125 deg grid):")
    print("     " + "  ".join(f"{v:8.4f}" for v in sla_lats))

    print(f"""
  Look at them. OISST has a cell centred at {oisst_lats[1]:.4f}.
  Sea level has NO cell there at all - it has cells at {sla_lats[2]:.4f}
  and {sla_lats[3]:.4f}, either side of it.

  So 'the sea level at the same place as this SST value' does not exist
  as a stored number. It has to be COMPUTED. That computation is regridding.""")


# ==========================================================================
def part4_regridding() -> None:
    rule("PART 4  -  Regridding: making the keys line up")

    cfg = load_config()
    tgt_lat, tgt_lon = grid.target_coords(cfg)

    print(f"""
  We define ONE canonical grid and force every dataset onto it:

      latitude   {tgt_lat[0]} to {tgt_lat[-1]}   ({len(tgt_lat)} cells)
      longitude  {tgt_lon[0]} to {tgt_lon[-1]}  ({len(tgt_lon)} cells)
      spacing    0.25 deg

  That grid was chosen to match OISST's native cells exactly, so SST needs
  no conversion at all. Everything else gets mapped onto it.

  For a FINER source (sea level at 0.125 deg, GLORYS at 0.0833 deg) we
  AREA-AVERAGE every source cell whose centre falls inside the target cell:
""")

    # Demonstrate the area-average with a real 2x2 block of fine cells.
    fine = np.array([[0.062, 0.068],
                     [0.071, 0.079]])
    print("      one 0.25 deg target cell contains a 2x2 block of 0.125 deg cells")
    print(f"          {fine[0,0]:.3f}   {fine[0,1]:.3f}")
    print(f"          {fine[1,0]:.3f}   {fine[1,1]:.3f}")
    print(f"      area-average  ->  {fine.mean():.4f} m   <-- the value stored on our grid")

    print("""
  Why average rather than just pick the nearest? Two reasons:
    * averaging uses all the information, not one arbitrary sample
    * it skips NaN land cells, so a coastal cell reflects only the water
      inside it instead of being contaminated by land

  GLORYS nests exactly 3x3 inside our cells, so each target value is the
  mean of 9 native cells. That alignment is not luck - the canonical grid
  was chosen for it.""")


# ==========================================================================
def part5_the_join(cfg) -> None:
    rule("PART 5  -  The join, executed on real data")

    day = date(2018, 3, 15)

    # SST: real data from the file on disk, regridded (a no-op for OISST).
    with xr.open_dataset(OISST_FILE) as ds:
        # standardize_coords renames OISST's 'zlev' axis to 'depth'; both it
        # and 'time' are length 1 here, so drop them to leave a plain map.
        sst = grid.standardize_coords(ds["sst"])
        sst = sst.isel(time=0, depth=0, drop=True)
        sst = grid.regrid(grid.subset_domain(sst, cfg), cfg)

    # The remaining variables use stand-ins, since they need a login.
    other = synthetic.open_day(cfg, day, ["sla", "sss", "uo", "vo", "uwnd", "vwnd"])
    target = synthetic.open_day_target(cfg, day)

    print(f"""
  Each piece is now on the identical 100 x 240 grid, so xarray can align
  them automatically on their shared lat/lon coordinates:

      merged = xr.merge([sst, sla, sss, currents, winds, glorys])

  Here is the result at our single grid cell, lat {SPOT_LAT}, lon {SPOT_LON}:
""")

    picks = {"sst": sst}
    for v in other.data_vars:
        picks[str(v)] = other[v]

    print("      SURFACE INPUTS (what the model sees)")
    print(f"      {'variable':<10}{'value':>12}   units")
    print("      " + "-" * 44)
    units = {"sst": "degC", "sla": "m", "sss": "psu", "uo": "m/s",
             "vo": "m/s", "uwnd": "m/s", "vwnd": "m/s"}
    for name in ["sst", "sla", "sss", "uo", "vo", "uwnd", "vwnd"]:
        val = float(picks[name].sel(lat=SPOT_LAT, lon=SPOT_LON, method="nearest"))
        print(f"      {name:<10}{val:>12.4f}   {units[name]}")

    print("\n      TARGET (what the model must predict)")
    print(f"      {'depth':>8}{'thetao':>12}   units")
    print("      " + "-" * 44)
    col = target["thetao"].sel(lat=SPOT_LAT, lon=SPOT_LON, method="nearest")
    for d, t in zip(cfg.depths, col.values):
        print(f"      {d:>7.0f}m{float(t):>12.4f}   degC")

    print("""
  That is one training example. Seven numbers in, fifteen numbers out,
  all describing the same 25 km square of ocean on the same day.""")


# ==========================================================================
def part6_scaling_up(cfg) -> None:
    rule("PART 6  -  Scaling that up to the full array")

    n_lat, n_lon = grid.grid_shape(cfg)
    n_in = len(cfg.input_variables)
    n_depth = len(cfg.depths)
    n_days = 2192

    print(f"""
  The previous section showed ONE cell. The pipeline does all {n_lat * n_lon:,}
  cells at once, for every day, producing two arrays:

      X  ({n_days}, {n_in}, {n_lat}, {n_lon})   inputs
         day   variable   lat   lon

      Y  ({n_days}, {n_depth}, {n_lat}, {n_lon})   target
         day    depth     lat   lon

  X[d, v, i, j]  is variable v at grid cell (i, j) on day d.
  Y[d, k, i, j]  is temperature at depth k, same cell, same day.

  The two arrays share their last three axes, which is what 'joined' means
  in practice: index both at [d, :, i, j] and you get the surface state and
  the temperature profile for the same place and time.

  The pipeline stage that performs this join is oceanembed/harmonize.py.
  It reads every source for one day, regrids each onto the canonical grid,
  and calls xr.merge - which aligns them on lat/lon automatically because
  by then they all carry identical coordinates.""")


# ==========================================================================
def main() -> int:
    if not OISST_FILE.exists():
        print(f"Expected the OISST file at {OISST_FILE}", file=sys.stderr)
        return 1

    cfg = load_config()

    part1_separate_files()
    part2_the_join_key()
    part3_the_mismatch()
    part4_regridding()
    part5_the_join(cfg)
    part6_scaling_up(cfg)

    print(f"\n{'=' * 74}")
    print("  In one sentence: the coordinates ARE the connection. Regridding")
    print("  forces every dataset onto identical coordinates, and once they")
    print("  match, merging is automatic.")
    print(f"{'=' * 74}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
