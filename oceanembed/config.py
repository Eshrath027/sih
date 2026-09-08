"""Configuration loading and path resolution.

One YAML file drives the whole pipeline. Everything else imports from here so
there is exactly one place where a setting can be defined.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# Project root is the directory containing 'configs/', found by walking up
# from this file. Keeps the pipeline runnable from any working directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    """Parsed pipeline configuration with convenience accessors."""

    raw: dict[str, Any]
    path: Path

    # -------------------------------------------------------------- access
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -------------------------------------------------------------- time
    @property
    def start(self) -> date:
        return _as_date(self.raw["time"]["start"])

    @property
    def end(self) -> date:
        return _as_date(self.raw["time"]["end"])

    def dates(self) -> list[date]:
        """Every day in the configured range, inclusive of both endpoints."""
        return date_range(self.start, self.end)

    def split_dates(self, split: str) -> list[date]:
        """Days belonging to one of the train/val/test splits."""
        lo, hi = self.raw["time"]["splits"][split]
        return date_range(_as_date(lo), _as_date(hi))

    def split_of(self, day: date) -> str | None:
        """Which split a given day falls into, or None if outside all of them."""
        for name, (lo, hi) in self.raw["time"]["splits"].items():
            if _as_date(lo) <= day <= _as_date(hi):
                return name
        return None

    # -------------------------------------------------------------- vars
    @property
    def input_variables(self) -> list[str]:
        return list(self.raw["inputs"]["variables"])

    @property
    def depths(self) -> list[float]:
        return [float(d) for d in self.raw["depths"]]

    # -------------------------------------------------------------- paths
    def path_for(self, key: str) -> Path:
        """Resolve a configured directory to an absolute path, creating it."""
        p = Path(self.raw["paths"][key])
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_dir(self) -> Path:
        return self.path_for("raw")

    @property
    def interim_dir(self) -> Path:
        return self.path_for("interim")

    @property
    def processed_dir(self) -> Path:
        return self.path_for("processed")

    @property
    def export_dir(self) -> Path:
        return self.path_for("export")

    @property
    def reports_dir(self) -> Path:
        return self.path_for("reports")

    def raw_source_dir(self, source: str) -> Path:
        """Per-source download directory, e.g. data/raw/oisst/."""
        p = self.raw_dir / source
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    """Read the YAML config. Falls back to configs/default.yaml."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}\n"
            f"Expected the pipeline config at {DEFAULT_CONFIG}"
        )
    with open(p) as fh:
        data = yaml.safe_load(fh)
    return Config(raw=data, path=p)


# ------------------------------------------------------------------ helpers

def _as_date(value: Any) -> date:
    """Accept a date, datetime, or 'YYYY-MM-DD' string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def date_range(start: date, end: date) -> list[date]:
    """Inclusive list of days from start to end."""
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def load_credentials() -> tuple[str | None, str | None]:
    """Copernicus Marine username and password.

    Read from the environment, optionally populated from a .env file at the
    project root. Credentials are never written into configs or committed.
    """
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except ImportError:
            # Minimal fallback so a missing python-dotenv is not fatal.
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

    return (
        os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"),
        os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"),
    )
