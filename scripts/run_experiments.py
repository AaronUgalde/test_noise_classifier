#!/usr/bin/env python3
"""Main CLI for running noise robustness experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.runner import run_experiments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run depression classification noise robustness experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "experiment.yaml"),
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to YAML override (e.g. config/experiment_day.yaml).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to run (default: all from config).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs to run (default: all from config).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Fold indices to run (default: 0..k-1).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Training seeds (default: from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        datasets=args.datasets,
        models=args.models,
        folds=args.folds,
        seeds=args.seeds,
        resume=not args.no_resume,
        device=args.device,
        config_path=args.config,
        override_path=args.override,
    )


if __name__ == "__main__":
    main()
