"""Statistical tests and critical difference analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

from src.utils.config import load_config, project_root


def build_model_dataset_matrix(
    df: pd.DataFrame, metric: str = "macro_f1"
) -> pd.DataFrame:
    """Average metric per (dataset, model) then pivot to models x datasets."""
    avg = df.groupby(["dataset", "model"])[metric].mean().reset_index()
    matrix = avg.pivot(index="model", columns="dataset", values=metric)
    return matrix.sort_index()


def compute_average_ranks(matrix: pd.DataFrame) -> pd.Series:
    """Rank models per dataset (1=best), then average ranks."""
    ranks = matrix.rank(axis=0, ascending=False)
    return ranks.mean(axis=1).sort_values()


def friedman_test(matrix: pd.DataFrame) -> dict:
    """Run Friedman test across datasets for all models."""
    # One array per model, values across datasets
    arrays = [matrix.loc[model].values for model in matrix.index]
    stat, p_value = friedmanchisquare(*arrays)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_datasets": int(matrix.shape[1]),
        "n_models": int(matrix.shape[0]),
    }


def nemenyi_posthoc(matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Nemenyi post-hoc test on transposed matrix (datasets as samples)."""
    # scikit-posthocs expects samples x groups
    data = matrix.T
    return sp.posthoc_nemenyi_friedman(data.values)


def critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute critical difference for Nemenyi test."""
    # q_alpha for alpha=0.05 (two-tailed), approximate Studentized range
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164, 15: 3.313,
        20: 3.419, 24: 3.475, 30: 3.544,
    }
    q = q_alpha_table.get(n_models, 3.475)
    cd = q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))
    return float(cd)


def run_statistical_analysis(
    df: pd.DataFrame,
    alpha: float = 0.05,
    results_dir: Path | None = None,
) -> dict:
    cfg = load_config()
    root = results_dir or (project_root() / cfg["project"]["results_dir"])
    stats_dir = root / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_model_dataset_matrix(df, metric="macro_f1")
    avg_ranks = compute_average_ranks(matrix)
    n_datasets = matrix.shape[1]
    n_models = matrix.shape[0]

    if n_datasets >= 3 and n_models >= 2:
        friedman = friedman_test(matrix)
        nemenyi = nemenyi_posthoc(matrix, alpha=alpha)
        nemenyi.index = matrix.index
        nemenyi.columns = matrix.index
        cd = critical_difference(n_models, n_datasets, alpha=alpha)
    else:
        friedman = {
            "statistic": None,
            "p_value": None,
            "n_datasets": n_datasets,
            "n_models": n_models,
            "skipped": "Friedman requires at least 3 datasets",
        }
        nemenyi = pd.DataFrame()
        cd = None

    output = {
        "metric": "macro_f1",
        "alpha": alpha,
        "friedman": friedman,
        "average_ranks": avg_ranks.to_dict(),
        "critical_difference": cd,
        "nemenyi_pvalues": nemenyi.to_dict() if not nemenyi.empty else {},
    }

    out_path = stats_dir / "friedman_nemenyi.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    avg_ranks.to_csv(stats_dir / "average_ranks.csv")
    if not nemenyi.empty:
        nemenyi.to_csv(stats_dir / "nemenyi_pvalues.csv")

    return output
