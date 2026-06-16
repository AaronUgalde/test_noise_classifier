"""Dataset loading and validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, project_root


REQUIRED_COLUMNS = {"id", "text", "manual_classification"}


def get_datasets_dir(config: dict | None = None) -> Path:
    cfg = config or load_config()
    return project_root() / cfg["data"]["datasets_dir"]


def list_datasets(config: dict | None = None) -> list[str]:
    cfg = config or load_config()
    return list(cfg["data"]["datasets"])


def load_dataset(name: str, config: dict | None = None) -> pd.DataFrame:
    """Load a single dataset CSV by name (without extension)."""
    cfg = config or load_config()
    datasets_dir = get_datasets_dir(cfg)
    path = datasets_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {name} missing columns: {missing}")

    df = df.sort_values(cfg["data"]["id_column"]).reset_index(drop=True)
    label_col = cfg["data"]["label_column"]
    if not set(df[label_col].unique()).issubset({0, 1}):
        raise ValueError(f"Dataset {name} has invalid labels in {label_col}")

    return df


def load_reference_labels(config: dict | None = None) -> pd.Series:
    """Load labels from the original dataset for shared stratified splits."""
    cfg = config or load_config()
    df = load_dataset("original", cfg)
    return df[cfg["data"]["label_column"]].astype(int)
