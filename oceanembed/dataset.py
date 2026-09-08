"""Turning the packed shards into training batches.

Two things happen here that the packing stage deliberately left undone.

Auxiliary channels are added. The network cannot infer the time of year or
where it is on the globe from seven surface fields alone, but both matter
enormously: the same surface temperature means something different in January
than in July, and different at 5N than at 25N. Four extra channels supply that
context.

Patches are cut. The full domain is 100x240, and training on whole domains
would give only a few hundred gradient steps per epoch. Cutting random 64x64
windows instead turns each day into many different examples, which is what
makes a few hundred days trainable at all.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}

# The canonical grid, needed to build the latitude channel.
LAT0, STEP = 5.125, 0.25


class OceanEmbedDataset(Dataset):
    """Random patches from the packed shards.

    Args:
        export_dir: folder holding the .npz shards and manifest.json.
        split:      'train', 'val' or 'test'.
        patch:      side length of the square window, or None for whole domains.
        samples_per_day: how many random patches to draw from each day per epoch.
        min_ocean:  reject patches with less ocean than this fraction, so the
                    network is not fed windows that are almost entirely land.
    """

    def __init__(
        self,
        export_dir: str | Path,
        split: str = "train",
        patch: int | None = 64,
        samples_per_day: int = 16,
        min_ocean: float = 0.25,
        seed: int = 0,
    ):
        self.dir = Path(export_dir)
        self.patch = patch
        self.samples_per_day = samples_per_day if patch else 1
        self.min_ocean = min_ocean
        self.seed = seed

        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest at {manifest_path}. Run: oceanembed pack"
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.variables = self.manifest["input_variables"]
        self.depths = self.manifest["depths"]

        stats_path = self.dir / "norm_stats.json"
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

        # Load every shard's matching days into memory. The whole dataset is a
        # few hundred megabytes at most, so this is far simpler and faster than
        # reading from disk on every batch.
        code = SPLIT_CODES[split]
        X, Y, V, dates = [], [], [], []
        self.mask = None

        for path in sorted(glob.glob(str(self.dir / "*.npz"))):
            with np.load(path) as z:
                keep = np.where(z["split"] == code)[0]
                if keep.size == 0:
                    continue
                X.append(z["X"][keep])
                Y.append(z["Y"][keep])
                V.append(z["Y_valid"][keep] if "Y_valid" in z.files
                         else np.ones_like(z["Y"][keep], dtype=bool))
                dates.append(z["dates"][keep])
                if self.mask is None:
                    self.mask = z["mask"]

        if not X:
            raise ValueError(
                f"No days found for split '{split}'. "
                f"Check time.splits in the config covers the days you built."
            )

        self.X = np.concatenate(X).astype("float32")
        self.Y = np.concatenate(Y).astype("float32")
        self.valid = np.concatenate(V)
        self.dates = np.concatenate(dates)

        # Only score cells that are both ocean and have a target value.
        self.valid = self.valid & self.mask[None, None, :, :]

        self.aux = self._build_aux()
        self.n_days = self.X.shape[0]

    # ------------------------------------------------------------------
    def _build_aux(self) -> np.ndarray:
        """Four context channels per day: season (2), latitude, ocean mask.

        The day of year is encoded as a sine/cosine pair rather than a number.
        As a plain number, 31 December (365) and 1 January (1) sit at opposite
        ends of the range despite being a day apart; as a point on a circle
        they are adjacent, which is the truth the network needs.
        """
        n_lat, n_lon = self.mask.shape
        days = self.dates.astype("datetime64[D]")
        doy = (days - days.astype("datetime64[Y]")).astype(int) + 1

        angle = 2.0 * math.pi * doy / 365.25
        sin = np.sin(angle).astype("float32")
        cos = np.cos(angle).astype("float32")

        lat = (LAT0 + np.arange(n_lat) * STEP).astype("float32")
        lat = (lat - lat.mean()) / lat.std()
        lat_plane = np.broadcast_to(lat[:, None], (n_lat, n_lon))

        aux = np.empty((len(doy), 4, n_lat, n_lon), dtype="float32")
        aux[:, 0] = sin[:, None, None]
        aux[:, 1] = cos[:, None, None]
        aux[:, 2] = lat_plane[None]
        aux[:, 3] = self.mask.astype("float32")[None]
        return aux

    @property
    def in_channels(self) -> int:
        return self.X.shape[1] + self.aux.shape[1]

    def __len__(self) -> int:
        return self.n_days * self.samples_per_day

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        day = index // self.samples_per_day

        x = np.concatenate([self.X[day], self.aux[day]], axis=0)
        y = self.Y[day]
        v = self.valid[day]

        if self.patch:
            i, j = self._pick_window(index, day)
            p = self.patch
            x = x[:, i:i + p, j:j + p]
            y = y[:, i:i + p, j:j + p]
            v = v[:, i:i + p, j:j + p]

        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(np.ascontiguousarray(y)), \
               torch.from_numpy(np.ascontiguousarray(v))

    def _pick_window(self, index: int, day: int) -> tuple[int, int]:
        """A random window with enough ocean in it.

        Seeded from the sample index so an epoch is reproducible, while still
        giving a different crop every time that day comes round.
        """
        n_lat, n_lon = self.mask.shape
        p = self.patch
        rng = np.random.default_rng(self.seed * 1_000_003 + index)

        for _ in range(20):
            i = int(rng.integers(0, n_lat - p + 1))
            j = int(rng.integers(0, n_lon - p + 1))
            if self.mask[i:i + p, j:j + p].mean() >= self.min_ocean:
                return i, j

        # Nothing suitable found; fall back to the wettest window we can get
        # cheaply rather than returning an all-land patch.
        best, best_frac = (0, 0), -1.0
        for _ in range(20):
            i = int(rng.integers(0, n_lat - p + 1))
            j = int(rng.integers(0, n_lon - p + 1))
            frac = float(self.mask[i:i + p, j:j + p].mean())
            if frac > best_frac:
                best, best_frac = (i, j), frac
        return best

    # ------------------------------------------------------------------
    def depth_std(self) -> np.ndarray:
        """Standard deviation of each depth level, for converting to Celsius.

        The network works in standardised units; multiplying an error by these
        turns it back into degrees.
        """
        levels = self.stats.get("target", {}).get("levels", [])
        if not levels:
            return np.ones(len(self.depths), dtype="float32")
        return np.array([lv["std"] for lv in levels], dtype="float32")

    def summary(self) -> str:
        ocean = float(self.mask.mean())
        return (
            f"{self.n_days} days, {len(self)} patches/epoch, "
            f"{self.in_channels} input channels, {len(self.depths)} depths, "
            f"{ocean:.0%} ocean"
        )
