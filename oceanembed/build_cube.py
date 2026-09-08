"""Stage 3: fold thousands of daily files into two analysis-ready cubes.

Output:
  data/processed/nio_surface.zarr   (time, lat, lon)          the model's inputs
  data/processed/nio_temp3d.zarr    (time, depth, lat, lon)   the training target

Zarr rather than one giant NetCDF, because a training loop reads random days
in random order. Zarr stores each chunk as a separate compressed block, so
reading day 1,700 costs one small read instead of seeking through a monolithic
file. It also appends cleanly, which lets this build incrementally and survive
interruption.

Chunking is chosen for how the data is actually consumed: whole spatial maps
for a small group of days. A training step wants every latitude and longitude
for one date, never a single pixel across all dates.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from .config import Config
from .harmonize import daily_path

# Days per chunk. Small enough that a random read is cheap, large enough that
# compression works well and the store does not become thousands of tiny files.
TIME_CHUNK = 32

SURFACE_STORE = "nio_surface.zarr"
TARGET_STORE = "nio_temp3d.zarr"


def store_paths(cfg: Config) -> tuple[Path, Path]:
    return (
        cfg.processed_dir / SURFACE_STORE,
        cfg.processed_dir / TARGET_STORE,
    )


def available_days(cfg: Config, days: list[date] | None = None) -> list[date]:
    """Days that have actually been harmonized to disk."""
    candidates = days if days is not None else cfg.dates()
    return [d for d in candidates if daily_path(cfg, d).exists()]


def _encoding(ds: xr.Dataset, chunks: dict[str, int]) -> dict:
    """Per-variable compression and chunk layout for the zarr store."""
    enc = {}
    for name, da in ds.data_vars.items():
        shape = tuple(chunks.get(d, ds.sizes[d]) for d in da.dims)
        enc[name] = {"chunks": shape}
    return enc


def build(
    cfg: Config,
    days: list[date] | None = None,
    include_target: bool = True,
    force: bool = False,
    batch_size: int = TIME_CHUNK,
    progress: bool = True,
) -> dict[str, object]:
    """Concatenate harmonized daily files into the two zarr cubes.

    Written in batches and appended along time, so peak memory stays at a few
    hundred megabytes no matter how many years are being built.
    """
    surface_path, target_path = store_paths(cfg)

    if force:
        for p in (surface_path, target_path):
            if p.exists():
                shutil.rmtree(p)

    present = available_days(cfg, days)
    if not present:
        raise RuntimeError(
            "No harmonized daily files found.\n"
            "Run `oceanembed harmonize` first (add --synthetic to test without downloads)."
        )

    surface_vars = cfg.input_variables
    target_var = cfg["target"]["variable"]

    batches = [present[i : i + batch_size] for i in range(0, len(present), batch_size)]
    iterator = batches
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(batches, desc="building cube", unit="batch")
        except ImportError:
            pass

    first_write = not surface_path.exists()
    n_written = 0
    mask_accum: np.ndarray | None = None

    for batch in iterator:
        chunk = xr.open_mfdataset(
            [daily_path(cfg, d) for d in batch],
            combine="by_coords",
            engine="netcdf4",
        ).load()

        # The land mask is static, so it is accumulated here and stored once
        # as a cube attribute rather than repeated for every single day.
        if "ocean_mask" in chunk:
            m = chunk["ocean_mask"]
            m = m.isel(time=0, drop=True) if "time" in m.dims else m
            m = m.values.astype(bool)
            mask_accum = m if mask_accum is None else (mask_accum & m)
            chunk = chunk.drop_vars("ocean_mask")

        surface = chunk[[v for v in surface_vars if v in chunk]]
        surface = surface.chunk({"time": min(TIME_CHUNK, surface.sizes["time"])})

        _append(
            surface,
            surface_path,
            first_write,
            _encoding(surface, {"time": TIME_CHUNK}),
        )

        if include_target and target_var in chunk:
            target = chunk[[target_var]]
            target = target.chunk({"time": min(TIME_CHUNK, target.sizes["time"])})
            _append(
                target,
                target_path,
                first_write or not target_path.exists(),
                _encoding(target, {"time": TIME_CHUNK}),
            )

        first_write = False
        n_written += len(batch)
        chunk.close()

    # Persist the shared land mask alongside the cubes. Every downstream
    # stage needs it, and recomputing it means rereading the whole store.
    if mask_accum is not None:
        mask_file = cfg.processed_dir / "ocean_mask.npy"
        np.save(mask_file, mask_accum)

    manifest = {
        "days": n_written,
        "first_day": present[0].isoformat(),
        "last_day": present[-1].isoformat(),
        "surface_variables": surface_vars,
        "target_variable": target_var if include_target else None,
        "depths": cfg.depths,
        "grid": {"lat": 100, "lon": 240, "resolution": cfg["domain"]["resolution"]},
        "ocean_cells": int(mask_accum.sum()) if mask_accum is not None else None,
        "surface_store": str(surface_path),
        "target_store": str(target_path) if include_target else None,
    }
    (cfg.processed_dir / "cube_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _append(ds: xr.Dataset, path: Path, first: bool, encoding: dict) -> None:
    """Write the first batch, append every batch after it."""
    if first or not path.exists():
        ds.to_zarr(path, mode="w", encoding=encoding, consolidated=True)
    else:
        ds.to_zarr(path, mode="a", append_dim="time", consolidated=True)


def open_cube(cfg: Config, which: str = "surface") -> xr.Dataset:
    """Open a built cube for reading."""
    surface_path, target_path = store_paths(cfg)
    path = surface_path if which == "surface" else target_path
    if not path.exists():
        raise FileNotFoundError(f"cube not built yet: {path}\nRun: oceanembed build-cube")
    return xr.open_zarr(path, consolidated=True)


def load_mask(cfg: Config) -> np.ndarray:
    """The shared ocean mask produced during the cube build."""
    p = cfg.processed_dir / "ocean_mask.npy"
    if not p.exists():
        raise FileNotFoundError(f"ocean mask missing: {p}\nRun: oceanembed build-cube")
    return np.load(p)
