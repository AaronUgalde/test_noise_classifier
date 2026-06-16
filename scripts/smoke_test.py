#!/usr/bin/env python3
"""Quick smoke test on a subset of models and one fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.aggregate import load_fold_results, save_summary  # noqa: E402
from src.evaluation.runner import run_experiments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the experiment pipeline.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tfidf_lr", "mpnet_lr", "bert_fe_lr", "bert_ft"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Running smoke test: dataset=original, fold=0, seed=42")
    print(f"Models: {args.models}")
    print("Using smoke_test.yaml overrides (2 epochs for fine-tuning)")
    root = Path(__file__).resolve().parents[1]
    run_experiments(
        datasets=["original"],
        models=args.models,
        folds=[0],
        seeds=[42],
        resume=False,
        device=args.device,
        config_path=str(root / "config" / "experiment.yaml"),
        override_path=str(root / "config" / "smoke_test.yaml"),
    )

    df = load_fold_results(ROOT / "results")
    print("\nResults:")
    print(df.to_string(index=False))

    save_summary(df, ROOT / "results")
    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()
