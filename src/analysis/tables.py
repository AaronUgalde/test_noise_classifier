"""Publication-ready tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.aggregate import METRIC_COLUMNS, aggregate_results
from src.utils.config import load_config, project_root


def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(std) or std == 0.0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def generate_table1(summary: pd.DataFrame) -> pd.DataFrame:
    """Table 1: full results per dataset and model."""
    rows = []
    for _, row in summary.iterrows():
        entry = {"Dataset": row["dataset"], "Modelo": row["model"]}
        for metric in METRIC_COLUMNS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            entry[metric.replace("_", " ").title()] = _format_mean_std(
                row[mean_col], row[std_col]
            )
            entry[f"{metric}_mean"] = row[mean_col]
        rows.append(entry)

    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["Dataset", "macro_f1_mean"], ascending=[True, False]
    ).drop(columns=["macro_f1_mean"])
    return table


def generate_table2(summary: pd.DataFrame) -> pd.DataFrame:
    """Table 2: global ranking averaged over all datasets."""
    global_rows = []
    for model, group in summary.groupby("model"):
        entry = {"Modelo": model}
        for metric in METRIC_COLUMNS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            entry[metric.replace("_", " ").title()] = _format_mean_std(
                group[mean_col].mean(), group[std_col].mean()
            )
            entry[f"{metric}_mean"] = group[mean_col].mean()
        global_rows.append(entry)

    table = pd.DataFrame(global_rows)
    table = table.sort_values("macro_f1_mean", ascending=False).drop(
        columns=["macro_f1_mean"]
    )
    return table


def save_tables(df: pd.DataFrame, results_dir: Path | None = None) -> tuple[Path, Path]:
    cfg = load_config()
    root = results_dir or (project_root() / cfg["project"]["results_dir"])
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_results(df)
    table1 = generate_table1(summary)
    table2 = generate_table2(summary)

    path1 = tables_dir / "table1_full_results.csv"
    path2 = tables_dir / "table2_global_ranking.csv"
    table1.to_csv(path1, index=False)
    table2.to_csv(path2, index=False)
    return path1, path2
