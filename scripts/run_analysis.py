#!/usr/bin/env python3
"""Generate aggregated results, tables, figures, and discussion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.aggregate import load_fold_results, save_summary  # noqa: E402
from src.analysis.discussion import generate_discussion  # noqa: E402
from src.analysis.plots import generate_all_figures  # noqa: E402
from src.analysis.statistics import run_statistical_analysis  # noqa: E402
from src.analysis.tables import save_tables  # noqa: E402
from src.utils.config import load_config, project_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-experiment analysis pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "experiment.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results_dir = project_root() / config["project"]["results_dir"]

    print("Loading fold results...")
    df = load_fold_results(results_dir)

    print("Aggregating results...")
    csv_path, json_path = save_summary(df, results_dir)
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")

    print("Generating tables...")
    t1, t2 = save_tables(df, results_dir)
    print(f"  Saved: {t1}")
    print(f"  Saved: {t2}")

    print("Running statistical analysis...")
    stats = run_statistical_analysis(
        df, alpha=config["statistics"]["alpha"], results_dir=results_dir
    )
    p_value = stats["friedman"]["p_value"]
    if p_value is not None:
        print(f"  Friedman p-value: {p_value:.6f}")
    else:
        print(f"  Friedman: skipped ({stats['friedman'].get('skipped', 'insufficient data')})")
    if stats["critical_difference"] is not None:
        print(f"  Critical difference: {stats['critical_difference']:.4f}")

    print("Generating figures...")
    figures = generate_all_figures(df, results_dir)
    for fig in figures:
        print(f"  Saved: {fig}")

    print("Generating discussion...")
    discussion_path = generate_discussion(df, results_dir)
    print(f"  Saved: {discussion_path}")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
