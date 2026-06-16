"""Aggregate fold-level results into summary statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, project_root

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
    "mcc",
    "balanced_accuracy",
]


def load_fold_results(results_dir: Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    root = results_dir or (project_root() / cfg["project"]["results_dir"])
    path = root / "folds" / "results_by_fold.csv"
    if not path.exists():
        raise FileNotFoundError(f"No fold results found at {path}")
    return pd.read_csv(path)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std per (dataset, model)."""
    grouped = df.groupby(["dataset", "model"])[METRIC_COLUMNS]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def save_summary(df: pd.DataFrame, results_dir: Path | None = None) -> tuple[Path, Path]:
    cfg = load_config()
    root = results_dir or (project_root() / cfg["project"]["results_dir"])
    root.mkdir(parents=True, exist_ok=True)

    summary = aggregate_results(df)

    csv_path = root / "results_summary.csv"
    json_path = root / "results_summary.json"

    summary.to_csv(csv_path, index=False)

    records = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        record = {"dataset": dataset, "model": model, "metrics": {}}
        for metric in METRIC_COLUMNS:
            record["metrics"][metric] = {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        records.append(record)

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return csv_path, json_path
