"""Checkpoint management for resumable experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Track completed experiment runs and persist fold-level results."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.folds_dir = self.results_dir / "folds"
        self.folds_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.folds_dir / "results_by_fold.csv"
        self.checkpoint_file = self.results_dir / "checkpoints.json"
        self._completed = self._load_completed()

    def _load_completed(self) -> set[tuple[str, str, int, int]]:
        if not self.checkpoint_file.exists():
            return set()
        data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        return {tuple(item) for item in data.get("completed", [])}

    def is_completed(self, dataset: str, model: str, fold: int, seed: int) -> bool:
        return (dataset, model, fold, seed) in self._completed

    def mark_completed(self, dataset: str, model: str, fold: int, seed: int) -> None:
        self._completed.add((dataset, model, fold, seed))
        payload = {"completed": [list(item) for item in sorted(self._completed)]}
        self.checkpoint_file.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def append_result(self, row: dict[str, Any]) -> None:
        import pandas as pd

        df = pd.DataFrame([row])
        header = not self.results_file.exists()
        df.to_csv(self.results_file, mode="a", index=False, header=header)

    def load_results(self):
        import pandas as pd

        if not self.results_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.results_file)
