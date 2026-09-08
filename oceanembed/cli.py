"""Command line interface.

    python -m oceanembed <stage> [options]

Every stage is idempotent and restartable, because downloads and long builds
get interrupted. Running a stage twice never corrupts anything; it skips the
work already done.

Typical order:

    python -m oceanembed check                    # what is configured and present
    python -m oceanembed fetch                    # download raw data
    python -m oceanembed harmonize                # one clean file per day
    python -m oceanembed build-cube               # zarr cubes
    python -m oceanembed stats                    # normalisation statistics
    python -m oceanembed qc                       # quality report and figures
    python -m oceanembed pack                     # shards for Colab

Add --synthetic to harmonize to run the whole chain with no downloads and no
credentials, which is the fastest way to verify the plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime

from .config import Config, load_config, load_credentials


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--start", default=None, help="override start date, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="override end date, YYYY-MM-DD")
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="use every Nth day. --every 3 cuts download and training time "
             "roughly threefold; consecutive ocean days are highly similar.",
    )


def _resolve_days(cfg: Config, args) -> list[date]:
    from .config import date_range, _as_date

    start = _as_date(args.start) if args.start else cfg.start
    end = _as_date(args.end) if args.end else cfg.end
    days = date_range(start, end)
    if args.every > 1:
        days = days[:: args.every]
    return days


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oceanembed",
        description="Reconstruct subsurface ocean temperature from satellite surface observations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("check", help="report configuration, credentials and what is on disk")
    _add_common(p)

    p = sub.add_parser("fetch", help="download raw data")
    _add_common(p)
    p.add_argument(
        "--source",
        default="all",
        help="one of: all, sst, sla, sss, currents, winds, glorys",
    )
    p.add_argument("--force", action="store_true", help="re-download files already present")
    p.add_argument("--dry-run", action="store_true", help="report the download size without fetching")

    p = sub.add_parser("harmonize", help="regrid everything onto the canonical grid, one file per day")
    _add_common(p)
    p.add_argument("--synthetic", action="store_true", help="generate stand-in data instead of reading downloads")
    p.add_argument("--no-target", action="store_true", help="surface inputs only, skip GLORYS")
    p.add_argument("--force", action="store_true", help="rebuild days already written")

    p = sub.add_parser("build-cube", help="concatenate daily files into zarr cubes")
    _add_common(p)
    p.add_argument("--no-target", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild the store from scratch")

    p = sub.add_parser("stats", help="normalisation statistics and climatology")
    _add_common(p)
    p.add_argument("--split", default="train")
    p.add_argument("--no-climatology", action="store_true")

    p = sub.add_parser("qc", help="quality control report and figures")
    _add_common(p)
    p.add_argument("--no-figures", action="store_true")

    p = sub.add_parser("pack", help="export compressed shards for GPU training")
    _add_common(p)
    p.add_argument("--raw-values", action="store_true", help="do not standardise before packing")
    p.add_argument("--no-target", action="store_true")

    p = sub.add_parser("all", help="harmonize, build-cube, stats, qc and pack in sequence")
    _add_common(p)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-target", action="store_true")

    return parser


# ------------------------------------------------------------------ stages

def cmd_check(cfg: Config, args) -> int:
    from .build_cube import store_paths
    from .harmonize import daily_path

    days = _resolve_days(cfg, args)
    user, _ = load_credentials()

    print(f"config              {cfg.path}")
    print(f"domain              {cfg['domain']['lat_min']}-{cfg['domain']['lat_max']}N, "
          f"{cfg['domain']['lon_min']}-{cfg['domain']['lon_max']}E at {cfg['domain']['resolution']} deg")
    print(f"grid                100 x 240 = 24,000 cells")
    print(f"period              {days[0]} to {days[-1]}  ({len(days)} days"
          f"{', every %dth' % args.every if args.every > 1 else ''})")
    print(f"input variables     {', '.join(cfg.input_variables)}")
    print(f"depth levels        {len(cfg.depths)}: {cfg.depths}")
    print()
    print(f"CMEMS credentials   {'found (' + user + ')' if user else 'NOT SET - see .env.example'}")
    print()

    harmonized = sum(1 for d in days if daily_path(cfg, d).exists())
    print(f"harmonized days     {harmonized} / {len(days)}")

    surface_store, target_store = store_paths(cfg)
    print(f"surface cube        {'built' if surface_store.exists() else 'not built'}")
    print(f"target cube         {'built' if target_store.exists() else 'not built'}")

    raw_total = 0
    for source in ("oisst", "sla", "sss", "currents", "winds", "glorys"):
        d = cfg.raw_dir / source
        if d.exists():
            size = sum(f.stat().st_size for f in d.rglob("*.nc"))
            n = len(list(d.rglob("*.nc")))
            raw_total += size
            if n:
                print(f"raw/{source:<12}    {n} files, {size/1e9:.2f} GB")
    print(f"raw data total      {raw_total/1e9:.2f} GB")
    return 0


def cmd_fetch(cfg: Config, args) -> int:
    from .sources import cmems, oisst

    days = _resolve_days(cfg, args)
    which = args.source

    groups = ["sst", "sla", "sss", "currents", "winds", "glorys"] if which == "all" else [which]

    for group in groups:
        print(f"\n=== {group} ===")
        try:
            if group == "sst":
                if args.dry_run:
                    print(f"  would download {len(days)} daily OISST files (~{len(days)*1.7/1000:.1f} GB)")
                    continue
                paths = oisst.fetch(cfg, days, force=args.force)
                print(f"  {len(paths)} files ready")
            else:
                paths = cmems.fetch(cfg, days, group, force=args.force, dry_run=args.dry_run)
                print(f"  {len(paths)} yearly files ready")
        except cmems.CredentialsMissing as exc:
            print(f"  SKIPPED: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    return 0


def cmd_harmonize(cfg: Config, args) -> int:
    from .harmonize import harmonize_range

    days = _resolve_days(cfg, args)
    result = harmonize_range(
        cfg,
        days,
        use_synthetic=args.synthetic,
        include_target=not args.no_target,
        force=args.force,
    )
    print(f"\nwritten {result['written']}, skipped (already present) {result['skipped']}, "
          f"failed {len(result['failed'])}")

    for day, err in list(result["failed"].items())[:10]:
        print(f"  {day}: {err}", file=sys.stderr)
    for problem in result["problems"][:10]:
        print(f"  RANGE: {problem}", file=sys.stderr)
    return 0 if result["written"] or result["skipped"] else 1


def cmd_build_cube(cfg: Config, args) -> int:
    from .build_cube import build

    days = _resolve_days(cfg, args)
    manifest = build(cfg, days, include_target=not args.no_target, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_stats(cfg: Config, args) -> int:
    from .stats import climatology, compute

    result = compute(cfg, split=args.split)
    print(f"statistics from {result['n_days']} days of the '{result['split']}' split\n")
    print(f"{'variable':<10} {'mean':>10} {'std':>10} {'valid':>8}")
    for var, s in result["surface"].items():
        print(f"{var:<10} {s['mean']:>10.3f} {s['std']:>10.3f} {s['valid_fraction']:>7.1%}")

    if result.get("target"):
        print(f"\n{'depth':>8} {'mean':>10} {'std':>10}")
        for level in result["target"]["levels"]:
            print(f"{level['depth']:>7.0f}m {level['mean']:>10.3f} {level['std']:>10.3f}")

    if not args.no_climatology:
        path = climatology(cfg, split=args.split)
        print(f"\nclimatology -> {path}")
    return 0


def cmd_qc(cfg: Config, args) -> int:
    from .qc import run

    report = run(cfg, make_figures=not args.no_figures)
    cov = report["coverage"]
    print(f"coverage   {cov['present']} / {cov['requested']} days ({cov['completeness']:.1%})")
    if cov["gaps"]:
        print(f"  largest gaps: " + ", ".join(f"{g['start']} ({g['days']}d)" for g in cov["gaps"][:5]))

    print("\nsurface fields:")
    for var, s in report["fields"]["surface"].items():
        flag = "ok" if s["within_plausible_range"] else "OUT OF RANGE"
        print(f"  {var:<8} missing {s['missing_fraction']:>7.2%}  "
              f"[{s['min']}, {s['max']}]  {flag}")

    if report["fields"].get("target"):
        mono = report["fields"].get("monotonic_cooling_with_depth")
        print(f"\ntemperature cools monotonically with depth: {mono}")
        if not mono:
            print(f"  inversions at: {report['fields']['temperature_inversions']}")

    print(f"\nreport -> {report['_report_path']}")
    for fig in report.get("figures", []):
        print(f"figure -> {fig}")
    return 0


def cmd_pack(cfg: Config, args) -> int:
    from .pack import pack

    manifest = pack(cfg, normalize=not args.raw_values, include_target=not args.no_target)
    print(f"{len(manifest['shards'])} shards, {manifest['total_size_mb']} MB total\n")
    print(f"{'year':>6} {'days':>6} {'MB':>8}  split breakdown")
    for s in manifest["shards"]:
        breakdown = ", ".join(f"{k} {v}" for k, v in s["split_counts"].items())
        print(f"{s['year']:>6} {s['days']:>6} {s['size_mb']:>8}  {breakdown}")
    print(f"\n-> {cfg.export_dir}")
    return 0


def cmd_all(cfg: Config, args) -> int:
    """Run every post-download stage in order, each in its own process.

    Subprocesses rather than direct function calls, deliberately. Each stage
    holds a lot of memory at its peak - open zarr stores, dask graphs,
    matplotlib figures - and Python hands very little of that back to the
    operating system when the stage finishes. Chaining them inside one process
    makes peak memory the SUM of every stage, which is enough to get the run
    killed on a machine that has other work on it. A subprocess releases
    everything on exit, so peak memory becomes the MAXIMUM of the stages
    instead of their total.
    """
    import subprocess

    base: list[str] = ["--every", str(args.every)]
    if args.config:
        base += ["--config", args.config]
    if args.start:
        base += ["--start", args.start]
    if args.end:
        base += ["--end", args.end]

    steps: list[tuple[str, list[str]]] = [
        ("harmonize", ["--synthetic"] if args.synthetic else []),
        ("build-cube", ["--force"]),
        ("stats", []),
        ("qc", []),
        ("pack", []),
    ]
    if args.no_target:
        for name, extra in steps:
            if name in ("harmonize", "build-cube", "pack"):
                extra.append("--no-target")

    for name, extra in steps:
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}", flush=True)
        result = subprocess.run([sys.executable, "-m", "oceanembed", name, *base, *extra])
        if result.returncode != 0:
            print(f"\nstopped at '{name}' (exit {result.returncode})", file=sys.stderr)
            if result.returncode in (137, -9):
                print(
                    "Exit 137 means the process was killed, nearly always by the\n"
                    "out-of-memory killer. Build fewer days at a time with --every,\n"
                    "or one year at a time with --start and --end.",
                    file=sys.stderr,
                )
            return result.returncode
    return 0


STAGES = {
    "check": cmd_check,
    "fetch": cmd_fetch,
    "harmonize": cmd_harmonize,
    "build-cube": cmd_build_cube,
    "stats": cmd_stats,
    "qc": cmd_qc,
    "pack": cmd_pack,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    return STAGES[args.stage](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
