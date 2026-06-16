"""Shared stratified splits (k-fold or holdout)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.loader import load_reference_labels
from src.utils.config import load_config


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def get_stratified_folds(config: dict | None = None) -> list[FoldSplit]:
    """Return fixed stratified splits shared across all datasets."""
    cfg = config or load_config()
    labels = load_reference_labels(cfg).values
    cv_cfg = cfg["cv"]
    split_type = cv_cfg.get("split", "kfold")

    if split_type == "holdout":
        test_size = cv_cfg.get("test_size", 0.2)
        split_seed = cv_cfg.get("split_seed", cv_cfg.get("fold_seed", 42))
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=split_seed,
            stratify=labels,
        )
        return [
            FoldSplit(
                fold=0,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        ]

    n_folds = cv_cfg["n_folds"]
    fold_seed = cv_cfg.get("fold_seed", cv_cfg.get("split_seed", 42))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    folds: list[FoldSplit] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        folds.append(
            FoldSplit(
                fold=fold_idx,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )
    return folds
