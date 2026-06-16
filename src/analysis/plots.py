"""Publication-ready visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.aggregate import METRIC_COLUMNS
from src.analysis.statistics import (
    build_model_dataset_matrix,
    compute_average_ranks,
    critical_difference,
)
from src.utils.config import load_config, project_root

NOISE_ORDER = [
    "original",
    "sin_emojis",
    "sin_numeros",
    "solo_minusculas",
    "solo_mayusculas",
    "abreviaciones",
    "sin_acentos",
    "sin_puntuacion",
    "sin_acentos_sin_puntuacion",
    "insertar_letras",
    "swap_letras",
    "error_teclado",
    "eliminar_letras",
    "slang",
    "stemming",
    "ruido_combinado",
    "lematizado",
    "sin_stopwords",
]


def _ordered_datasets(columns: list[str]) -> list[str]:
    ordered = [d for d in NOISE_ORDER if d in columns]
    remaining = [d for d in columns if d not in ordered]
    return ordered + remaining


def _plot_heatmap(matrix: pd.DataFrame, metric: str, out_path: Path, config: dict) -> None:
    datasets = _ordered_datasets(list(matrix.columns))
    matrix = matrix[datasets]
    matrix = matrix.sort_index()

    figsize = tuple(config["plots"]["figsize_heatmap"])
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=False,
        cmap="YlGnBu",
        linewidths=0.3,
        cbar_kws={"label": metric},
    )
    plt.title(f"Heatmap: {metric}")
    plt.xlabel("Dataset")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=config["plots"]["dpi"])
    plt.close()


def plot_global_ranking(df: pd.DataFrame, out_path: Path, config: dict) -> None:
    metrics = ["macro_f1", "mcc", "roc_auc"]
    global_means = df.groupby("model")[metrics].mean().sort_values("macro_f1", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=tuple(config["plots"]["figsize_bar"]))
    for ax, metric in zip(axes, metrics):
        data = global_means[metric].sort_values(ascending=True)
        ax.barh(data.index, data.values, color=sns.color_palette("colorblind")[0])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlim(0, 1)
    plt.suptitle("Global Model Ranking (averaged over datasets)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=config["plots"]["dpi"])
    plt.close()


def plot_metric_boxplots(df: pd.DataFrame, out_path: Path, config: dict) -> None:
    metrics = ["macro_f1", "mcc", "roc_auc"]
    fig, axes = plt.subplots(1, 3, figsize=tuple(config["plots"]["figsize_box"]))
    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x=metric, y="model", ax=ax, orient="h", color=sns.color_palette("colorblind")[1])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlim(0, 1)
    plt.suptitle("Metric Distribution Across All Runs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=config["plots"]["dpi"])
    plt.close()


def plot_critical_difference(df: pd.DataFrame, out_path: Path, config: dict) -> None:
    matrix = build_model_dataset_matrix(df, "macro_f1")
    avg_ranks = compute_average_ranks(matrix)
    cd = critical_difference(matrix.shape[0], matrix.shape[1])

    sorted_models = avg_ranks.sort_values().index.tolist()
    sorted_ranks = avg_ranks.sort_values().values
    n = len(sorted_models)

    fig, ax = plt.subplots(figsize=tuple(config["plots"]["figsize_cd"]))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(f"Critical Difference Diagram (CD = {cd:.3f})")

    y_pos = 0.5
    ax.hlines(y_pos, 1, n, color="black", linewidth=2)

    for i, (model, rank) in enumerate(zip(sorted_models, sorted_ranks), start=1):
        ax.plot(i, y_pos, "ko", markersize=8)
        ax.text(i, y_pos + 0.08, model, rotation=45, ha="left", va="bottom", fontsize=8)

    # Connect models not significantly different (within CD)
    groups: list[list[int]] = []
    current_group = [1]
    for i in range(2, n + 1):
        if sorted_ranks[i - 1] - sorted_ranks[current_group[0] - 1] <= cd:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [i]
    if len(current_group) > 1:
        groups.append(current_group)

    bar_y = 0.35
    for group in groups:
        ax.plot([group[0], group[-1]], [bar_y, bar_y], color="gray", linewidth=6, solid_capstyle="butt")
        bar_y -= 0.05

    plt.tight_layout()
    plt.savefig(out_path, dpi=config["plots"]["dpi"])
    plt.close()


def generate_all_figures(df: pd.DataFrame, results_dir: Path | None = None) -> list[Path]:
    cfg = load_config()
    root = results_dir or (project_root() / cfg["project"]["results_dir"])
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for metric, filename in [
        ("macro_f1", "fig1_macro_f1_heatmap.png"),
        ("mcc", "fig2_mcc_heatmap.png"),
        ("roc_auc", "fig3_roc_auc_heatmap.png"),
    ]:
        matrix = build_model_dataset_matrix(df, metric)
        out = fig_dir / filename
        _plot_heatmap(matrix, metric, out, cfg)
        paths.append(out)

    p4 = fig_dir / "fig4_global_ranking.png"
    plot_global_ranking(df, p4, cfg)
    paths.append(p4)

    p5 = fig_dir / "fig5_metric_boxplots.png"
    plot_metric_boxplots(df, p5, cfg)
    paths.append(p5)

    p6 = fig_dir / "fig6_critical_difference.png"
    plot_critical_difference(df, p6, cfg)
    paths.append(p6)

    return paths
