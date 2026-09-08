"""Report the time span of every configured dataset, and their overlap.

The period this project can train on is not a free choice: it is the
intersection of every input's availability. One product ending early silently
caps the whole thing, so this prints each span and the window they share.

Run:  .venv/bin/python scripts/check_coverage.py
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oceanembed.config import load_config  # noqa: E402

# Extra candidates worth reporting even though only some are in the config,
# because when one product falls short the alternative is the next question.
EXTRA = {
    "winds (NRT)": "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
    "SSS (NRT)": "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
}


def epoch_to_date(value) -> dt.date | None:
    if value is None:
        return None
    # The catalogue reports milliseconds since the epoch.
    return dt.datetime.fromtimestamp(value / 1000, tz=dt.timezone.utc).date()


def time_span(dataset_id: str) -> tuple[dt.date, dt.date] | None:
    """Earliest and latest time step advertised for a dataset."""
    import copernicusmarine as cm

    try:
        cat = cm.describe(dataset_id=dataset_id, disable_progress_bar=True)
    except Exception as exc:
        print(f"    ! {type(exc).__name__}: {exc}")
        return None

    lo = hi = None
    for product in cat.products:
        for ds in product.datasets:
            for version in ds.versions:
                for part in version.parts:
                    for service in part.services:
                        for var in service.variables:
                            for coord in var.coordinates:
                                if coord.coordinate_id != "time":
                                    continue
                                a = epoch_to_date(coord.minimum_value)
                                b = epoch_to_date(coord.maximum_value)
                                if a and (lo is None or a < lo):
                                    lo = a
                                if b and (hi is None or b > hi):
                                    hi = b
    return (lo, hi) if lo and hi else None


def bar(lo: dt.date, hi: dt.date, axis_lo: int, axis_hi: int, width: int = 46) -> str:
    """A crude timeline bar, so the gaps are visible at a glance."""
    span = axis_hi - axis_lo
    start = max(0, min(width - 1, round((lo.year - axis_lo) / span * width)))
    end = max(start + 1, min(width, round((hi.year - axis_lo) / span * width)))
    return "." * start + "#" * (end - start) + "." * (width - end)


def main() -> int:
    cfg = load_config()

    targets: dict[str, str] = {}
    for key in ("sla", "sss", "currents", "winds", "glorys"):
        targets[key] = cfg["sources"][key]["dataset_id"]
    targets.update(EXTRA)

    print("\nQuerying the Copernicus catalogue for time coverage...\n")

    spans: dict[str, tuple[dt.date, dt.date]] = {}
    for label, dataset_id in targets.items():
        print(f"  {label:<16} {dataset_id}")
        span = time_span(dataset_id)
        if span:
            spans[label] = span
            print(f"    {span[0]}  ->  {span[1]}")
        print()

    # OISST is not a Copernicus product; its span is documented and stable.
    spans["SST (OISST)"] = (dt.date(1981, 9, 1), dt.date.today())

    axis_lo, axis_hi = 1981, 2027
    print("=" * 76)
    print(f"  {'dataset':<16} {axis_lo}{' ' * 38}{axis_hi}")
    print("=" * 76)
    for label, (lo, hi) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        print(f"  {label:<16} {bar(lo, hi, axis_lo, axis_hi)}  {lo.year}-{hi.year}")

    # ---------------------------------------------------------- overlap
    print("\n" + "=" * 76)

    required = ["sla", "sss", "currents", "winds", "glorys", "SST (OISST)"]
    have = {k: v for k, v in spans.items() if k in required}

    if len(have) == len(required):
        lo = max(v[0] for v in have.values())
        hi = min(v[1] for v in have.values())
        if lo < hi:
            days = (hi - lo).days
            print(f"  ALL SEVEN VARIABLES OVERLAP:  {lo}  ->  {hi}   ({days:,} days)")
        else:
            print("  NO WINDOW CONTAINS ALL SEVEN VARIABLES.")
            print("  The binding constraints are:")
            latest_start = max(have.items(), key=lambda kv: kv[1][0])
            earliest_end = min(have.items(), key=lambda kv: kv[1][1])
            print(f"    latest start : {latest_start[0]} begins {latest_start[1][0]}")
            print(f"    earliest end : {earliest_end[0]} ends   {earliest_end[1][1]}")

    # What is achievable if winds are dropped or sourced elsewhere.
    without_wind = {k: v for k, v in have.items() if k != "winds"}
    if without_wind:
        lo = max(v[0] for v in without_wind.values())
        hi = min(v[1] for v in without_wind.values())
        days = (hi - lo).days
        print(f"\n  WITHOUT CMEMS WINDS:          {lo}  ->  {hi}   ({days:,} days)")
        print("  (ERA5 covers 1940-present, so this becomes the real limit")
        print("   once winds come from ERA5 instead.)")

    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
