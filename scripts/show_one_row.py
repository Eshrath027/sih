"""Print ONE training example in full: every number, for one place, one day.

The dataset is four-dimensional, which makes it hard to picture. This strips
it down to a single grid cell on a single day - one row of 7 inputs and one
row of 15 outputs - so the shape of the problem is obvious.

Run:  .venv/bin/python scripts/show_one_row.py
      .venv/bin/python scripts/show_one_row.py --lat 18 --lon 89   (Bay of Bengal)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oceanembed.config import load_config, date_range   # noqa: E402
from oceanembed.sources import cmems, oisst, synthetic  # noqa: E402

UNITS = {
    "sst": "degC", "sla": "m", "sss": "psu",
    "uo": "m/s", "vo": "m/s", "uwnd": "m/s", "vwnd": "m/s",
}
MEANING = {
    "sst":  "sea surface temperature",
    "sla":  "sea level anomaly (the surface bulge)",
    "sss":  "sea surface salinity",
    "uo":   "surface current, eastward",
    "vo":   "surface current, northward",
    "uwnd": "wind, eastward",
    "vwnd": "wind, northward",
}
GROUP = {"sla": "sla", "sss": "sss", "uo": "currents", "vo": "currents",
         "uwnd": "winds", "vwnd": "winds"}


def first_ready_day(cfg, every: int) -> date | None:
    """A day whose SST file is on disk, so the example uses real data."""
    for day in date_range(cfg.start, cfg.end)[::every]:
        if oisst.local_path(cfg, day).exists():
            return day
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=15.125)
    ap.add_argument("--lon", type=float, default=68.125)
    ap.add_argument("--every", type=int, default=112)
    args = ap.parse_args()

    cfg = load_config()
    day = first_ready_day(cfg, args.every)
    if day is None:
        print("No downloaded days yet. Run the fetch command first.", file=sys.stderr)
        return 1

    print(f"\n{'=' * 78}")
    print(f"  ONE TRAINING EXAMPLE")
    print(f"  day {day}   at latitude {args.lat}N, longitude {args.lon}E")
    print(f"  (one 25 km square of ocean, on one day)")
    print(f"{'=' * 78}")

    # ------------------------------------------------------------- inputs
    values: dict[str, float] = {}
    source: dict[str, str] = {}

    try:
        ds = oisst.open_day(cfg, day)
        values["sst"] = float(ds["sst"].sel(lat=args.lat, lon=args.lon, method="nearest"))
        source["sst"] = "REAL"
    except Exception:
        source["sst"] = "pending"

    for var in ["sla", "sss", "uo", "vo", "uwnd", "vwnd"]:
        try:
            ds = cmems.open_day(cfg, day, GROUP[var])
            values[var] = float(ds[var].sel(lat=args.lat, lon=args.lon, method="nearest"))
            source[var] = "REAL"
        except Exception:
            source[var] = "pending"

    # Fill anything not yet downloaded so the row is complete and readable.
    fake = synthetic.open_day(cfg, day, cfg.input_variables)
    for var in cfg.input_variables:
        if var not in values:
            values[var] = float(fake[var].sel(lat=args.lat, lon=args.lon, method="nearest"))

    print("\n  X  -  THE INPUTS  (what the satellite saw)")
    print("  " + "-" * 74)
    print(f"  {'#':>2}  {'name':<6} {'value':>11}  {'units':<6} {'source':<8} meaning")
    print("  " + "-" * 74)
    for i, var in enumerate(cfg.input_variables):
        print(f"  {i:>2}  {var:<6} {values[var]:>11.4f}  {UNITS[var]:<6} "
              f"{source[var]:<8} {MEANING[var]}")

    row = "  ".join(f"{values[v]:.3f}" for v in cfg.input_variables)
    print(f"\n  As a single row of 7 numbers:")
    print(f"     [ {row} ]")

    # ------------------------------------------------------------- target
    try:
        tds = cmems.open_day(cfg, day, "glorys")
        from oceanembed import grid
        col = grid.interp_depth(tds["thetao"], cfg.depths)
        temps = col.sel(lat=args.lat, lon=args.lon, method="nearest").values
        tsource = "REAL"
    except Exception:
        tds = synthetic.open_day_target(cfg, day)
        temps = tds["thetao"].sel(lat=args.lat, lon=args.lon, method="nearest").values
        tsource = "pending (stand-in shown)"

    print(f"\n  Y  -  THE ANSWER  (temperature underneath)   source: {tsource}")
    print("  " + "-" * 74)
    print(f"  {'#':>2}  {'depth':>8}  {'temperature':>12}")
    print("  " + "-" * 74)
    for i, (d, t) in enumerate(zip(cfg.depths, temps)):
        bar = "#" * max(0, int((float(t) - 3) / 30 * 34))
        print(f"  {i:>2}  {d:>7.0f}m  {float(t):>10.3f} C  {bar}")

    row = "  ".join(f"{float(t):.2f}" for t in temps)
    print(f"\n  As a single row of 15 numbers:")
    print(f"     [ {row} ]")

    # ------------------------------------------------------------- summary
    print(f"\n{'=' * 78}")
    print("  SO ONE EXAMPLE IS:")
    print(f"     7 numbers in   ->   15 numbers out")
    print()
    print("  The model does all 24,000 squares at once, for every day:")
    print(f"     X  (days, 7, 100, 240)      Y  (days, 15, 100, 240)")
    print()
    print("  This row is X[d, :, i, j] and Y[d, :, i, j] for one d, i, j.")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
