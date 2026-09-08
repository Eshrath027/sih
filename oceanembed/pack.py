"""Stage 6: pack the cube into shards for GPU training elsewhere.

Training happens on a free Colab or Kaggle GPU, not on this machine, so the
data has to travel. A zarr store is thousands of small files, which uploads
appallingly slowly to Google Drive. This stage rewrites it as a handful of
compressed .npz files - one per year - that upload as single objects.

What comes out, per shard:

  X     float32 (days, n_vars, 100, 240)   the surface inputs
  Y     float32 (days, 15, 100, 240)       temperature at the standard depths
  mask  bool    (100, 240)                 ocean cells
  dates int64   (days,)                    days since 1970-01-01

Values are already standardised using the training-split statistics, and land
is filled with 0 (which after standardisation is the mean, the least harmful
value to feed a convolution). The mask travels with the data so the loss can
exclude land - a model must never be rewarded or penalised for what it
predicts over Rajasthan.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .build_cube import load_mask, open_cube
from .stats import load_stats

# Integer codes for the per-day split array stored in every shard. Integers
# rather than strings so the array stays a compact numeric type that loads
# straight into a training script.
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def pack(
    cfg: Config,
    normalize: bool = True,
    include_target: bool = True,
    progress: bool = True,
) -> dict:
    """Write yearly .npz shards plus a manifest describing them."""
    mask = load_mask(cfg)
    stats = load_stats(cfg) if normalize else None
    out_dir = cfg.export_dir

    surface = open_cube(cfg, "surface")
    variables = [v for v in cfg.input_variables if v in surface.data_vars]

    target = None
    if include_target:
        try:
            target = open_cube(cfg, "target")
        except FileNotFoundError:
            include_target = False

    times = pd.DatetimeIndex(surface.time.values)
    years = sorted(set(times.year))

    shards = []
    iterator = years
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(years, desc="packing", unit="year")
        except ImportError:
            pass

    for year in iterator:
        idx = np.where(times.year == year)[0]
        if idx.size == 0:
            continue

        # --- inputs ----------------------------------------------------
        planes = []
        for var in variables:
            values = surface[var].isel(time=idx).values.astype("float32")
            if normalize:
                s = stats["surface"][var]
                values = (values - s["mean"]) / s["std"]
            planes.append(values)
        X = np.stack(planes, axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # A per-day split label, not one label for the whole shard. A calendar
        # year can straddle a split boundary - 2019 is training up to 30 June
        # and validation afterwards - so labelling a shard by its majority
        # would quietly put validation days into training and inflate every
        # score that gets reported.
        split_codes = np.array(
            [SPLIT_CODES.get(cfg.split_of(t.date()), -1) for t in times[idx]],
            dtype="int8",
        )

        payload = {
            "X": X,
            "mask": mask,
            "dates": times[idx].values.astype("datetime64[D]").astype("int64"),
            "split": split_codes,
        }

        # --- target ----------------------------------------------------
        if include_target:
            tvar = cfg["target"]["variable"]
            Y = target[tvar].isel(time=idx).values.astype("float32")
            if normalize:
                for i, level in enumerate(stats["target"]["levels"]):
                    Y[:, i] = (Y[:, i] - level["mean"]) / level["std"]
            # Where the target is missing (below the sea floor) the loss must
            # ignore the cell, so a per-cell validity mask travels with it.
            payload["Y_valid"] = np.isfinite(Y)
            payload["Y"] = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        dest = out_dir / f"oceanembed_{year}.npz"
        np.savez_compressed(dest, **payload)

        shards.append(
            {
                "year": year,
                "file": dest.name,
                "days": int(idx.size),
                "size_mb": round(dest.stat().st_size / 1e6, 1),
                # How this shard's days divide between the splits. A shard
                # spanning a boundary reports both, so the split is never
                # silently rounded to whichever happens to dominate.
                "split_counts": {
                    name: int((split_codes == code).sum())
                    for name, code in SPLIT_CODES.items()
                    if (split_codes == code).any()
                },
            }
        )

    surface.close()
    if target is not None:
        target.close()

    manifest = {
        "input_variables": variables,
        "n_input_channels": len(variables),
        "depths": cfg.depths,
        "grid": {"lat": 100, "lon": 240},
        "split_codes": SPLIT_CODES,
        "split_note": "each shard carries a per-day 'split' array; filter on it "
                      "rather than assuming a whole shard belongs to one split",
        "normalized": normalize,
        "normalization": "per-variable for inputs, per-depth-level for the target",
        "stats_file": "norm_stats.json",
        "land_fill_value": 0.0,
        "splits": {k: list(v) for k, v in cfg["time"]["splits"].items()},
        "shards": shards,
        "total_size_mb": round(sum(s["size_mb"] for s in shards), 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Ship the statistics alongside, so predictions can be converted back
    # into degrees Celsius on the training machine.
    if stats is not None:
        (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))

    return manifest


def load_shard(path: str | Path) -> dict:
    """Read one shard back. Mirrors what the training code on Colab will do."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
