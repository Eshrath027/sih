"""Look inside a packed .npz training file.

The final dataset is raw numbers with no labels attached, which makes it hard
to check by eye. This prints what is in a shard, converts the standardised
values back into real units, and can draw the maps to a PNG.

Run:
    .venv/bin/python scripts/view_npz.py                    # summary of all shards
    .venv/bin/python scripts/view_npz.py --file data/export/oceanembed_1994.npz
    .venv/bin/python scripts/view_npz.py --day 0 --plot     # draw maps to reports/
    .venv/bin/python scripts/view_npz.py --day 0 --lat 15 --lon 68   # one point
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPORT = ROOT / "data" / "export"
SPLIT_NAME = {0: "train", 1: "val", 2: "test"}

# The canonical grid, so a row/column index can be turned back into a place.
LAT0, LON0, STEP = 5.125, 45.125, 0.25


def load_manifest() -> dict:
    p = EXPORT / "manifest.json"
    if not p.exists():
        print(f"No manifest at {p}. Run the pack stage first.", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(p.read_text())


def load_stats() -> dict:
    p = EXPORT / "norm_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def summary(manifest: dict) -> None:
    """What is in the export folder overall."""
    print(f"\n{'=' * 70}")
    print("  EXPORT FOLDER")
    print(f"{'=' * 70}")
    print(f"  inputs        {', '.join(manifest['input_variables'])}")
    print(f"  depths        {manifest['depths']}")
    print(f"  grid          {manifest['grid']['lat']} x {manifest['grid']['lon']}")
    print(f"  normalised    {manifest['normalized']}  ({manifest['normalization']})")
    print(f"  total         {len(manifest['shards'])} files, {manifest['total_size_mb']} MB")
    print(f"\n  {'file':<26} {'days':>5} {'MB':>7}  split")
    print("  " + "-" * 62)
    for s in manifest["shards"]:
        breakdown = ", ".join(f"{k} {v}" for k, v in s["split_counts"].items())
        print(f"  {s['file']:<26} {s['days']:>5} {s['size_mb']:>7}  {breakdown}")
    print()


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Undo the standardisation, turning the stored numbers back into units."""
    return values * std + mean


def describe_file(path: Path, manifest: dict, stats: dict) -> np.lib.npyio.NpzFile:
    z = np.load(path)
    print(f"\n{'=' * 70}")
    print(f"  {path.name}")
    print(f"{'=' * 70}")
    print(f"  {'array':<10} {'shape':<24} {'type':<9} what it is")
    print("  " + "-" * 66)
    meaning = {
        "X": "the 7 surface inputs",
        "Y": "temperature at 15 depths",
        "Y_valid": "where the answer exists",
        "mask": "ocean (True) vs land (False)",
        "dates": "days since 1970-01-01",
        "split": "0=train 1=val 2=test",
    }
    for k in z.files:
        a = z[k]
        print(f"  {k:<10} {str(a.shape):<24} {str(a.dtype):<9} {meaning.get(k, '')}")

    days = z["dates"].astype("datetime64[D]")
    print(f"\n  days in this file:")
    for i, (d, s) in enumerate(zip(days, z["split"])):
        print(f"     [{i}]  {d}   {SPLIT_NAME.get(int(s), '?')}")
    return z


def show_point(z, manifest: dict, stats: dict, day: int, lat: float, lon: float) -> None:
    """Every value at one grid cell, converted back to real units."""
    i = int(round((lat - LAT0) / STEP))
    j = int(round((lon - LON0) / STEP))
    i = max(0, min(z["mask"].shape[0] - 1, i))
    j = max(0, min(z["mask"].shape[1] - 1, j))

    real_lat = LAT0 + i * STEP
    real_lon = LON0 + j * STEP
    date = z["dates"].astype("datetime64[D]")[day]

    print(f"\n{'=' * 70}")
    print(f"  ONE POINT   day {date}   {real_lat}N {real_lon}E   [row {i}, col {j}]")
    print(f"{'=' * 70}")

    if not z["mask"][i, j]:
        print("  This cell is LAND. Pick another point.")
        return

    units = {"sst": "degC", "sla": "m", "sss": "psu", "uo": "m/s",
             "vo": "m/s", "uwnd": "m/s", "vwnd": "m/s"}

    print(f"\n  X - INPUTS")
    print(f"  {'variable':<8} {'stored':>10} {'real value':>13}  units")
    print("  " + "-" * 50)
    for c, var in enumerate(manifest["input_variables"]):
        raw = float(z["X"][day, c, i, j])
        s = stats.get("surface", {}).get(var)
        real = denormalize(raw, s["mean"], s["std"]) if s else float("nan")
        print(f"  {var:<8} {raw:>10.4f} {real:>13.4f}  {units.get(var, '')}")

    print(f"\n  Y - ANSWER")
    print(f"  {'depth':>8} {'stored':>10} {'real value':>13}")
    print("  " + "-" * 50)
    levels = stats.get("target", {}).get("levels", [])
    for k, depth in enumerate(manifest["depths"]):
        raw = float(z["Y"][day, k, i, j])
        real = denormalize(raw, levels[k]["mean"], levels[k]["std"]) if k < len(levels) else float("nan")
        bar = "#" * max(0, int((real - 3) / 30 * 26))
        print(f"  {depth:>7.0f}m {raw:>10.4f} {real:>12.3f}C  {bar}")
    print()


def plot(z, manifest: dict, stats: dict, day: int, out: Path) -> Path:
    """Draw the inputs and a few depth levels to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = manifest["input_variables"]
    depths = manifest["depths"]
    show_depths = [0, 4, 6, 9, 12, 14]          # surface, 30m, 75m, 150m, 500m, 1000m
    mask = z["mask"]
    date = z["dates"].astype("datetime64[D]")[day]

    n = len(variables) + len(show_depths)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    extent = [LON0, LON0 + mask.shape[1] * STEP, LAT0, LAT0 + mask.shape[0] * STEP]

    for ax, (c, var) in zip(axes, enumerate(variables)):
        s = stats.get("surface", {}).get(var)
        field = z["X"][day, c].astype("float64")
        if s:
            field = denormalize(field, s["mean"], s["std"])
        field = np.where(mask, field, np.nan)
        im = ax.imshow(field, origin="lower", extent=extent, cmap="viridis", aspect="auto")
        ax.set_title(f"{var}  (input)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

    levels = stats.get("target", {}).get("levels", [])
    for ax, k in zip(axes[len(variables):], show_depths):
        field = z["Y"][day, k].astype("float64")
        if k < len(levels):
            field = denormalize(field, levels[k]["mean"], levels[k]["std"])
        field = np.where(mask & z["Y_valid"][day, k], field, np.nan)
        im = ax.imshow(field, origin="lower", extent=extent, cmap="RdYlBu_r", aspect="auto")
        ax.set_title(f"temperature at {depths[k]:.0f} m  (answer)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Packed dataset, day {date}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=105)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="which .npz to open")
    ap.add_argument("--day", type=int, default=None, help="index of the day within the file")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--plot", action="store_true", help="write maps to reports/")
    args = ap.parse_args()

    manifest = load_manifest()
    stats = load_stats()

    if args.file is None and args.day is None:
        summary(manifest)
        print("  To look inside one file:")
        print("     .venv/bin/python scripts/view_npz.py --day 0")
        print("     .venv/bin/python scripts/view_npz.py --day 0 --plot")
        print("     .venv/bin/python scripts/view_npz.py --day 0 --lat 15 --lon 68\n")
        return 0

    path = Path(args.file) if args.file else Path(sorted(glob.glob(str(EXPORT / "*.npz")))[0])
    z = describe_file(path, manifest, stats)

    day = args.day if args.day is not None else 0
    if day >= len(z["dates"]):
        print(f"\nThis file has only {len(z['dates'])} days (0 to {len(z['dates'])-1}).",
              file=sys.stderr)
        return 1

    if args.lat is not None or args.plot is False:
        show_point(z, manifest, stats, day,
                   args.lat if args.lat is not None else 15.125,
                   args.lon if args.lon is not None else 68.125)

    if args.plot:
        out = ROOT / "reports" / f"npz_view_{path.stem}_day{day}.png"
        out.parent.mkdir(exist_ok=True)
        plot(z, manifest, stats, day, out)
        print(f"  picture -> {out}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
