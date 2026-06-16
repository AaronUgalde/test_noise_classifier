"""Experiment orchestration."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from src.data.loader import load_dataset
from src.data.splits import get_stratified_folds
from src.evaluation.metrics import compute_metrics
from src.features.embeddings import EmbeddingExtractor
from src.features.fasttext import FastTextEmbedder
from src.features.tfidf import build_tfidf_vectorizer, fit_transform_tfidf
from src.models.finetune import finetune_transformer
from src.models.sklearn_clf import build_classifier, train_predict_sklearn
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config, project_root
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_model_id(model_id: str) -> dict:
    """Parse model identifier into components."""
    if model_id.endswith("_ft"):
        arch = model_id.replace("_ft", "")
        return {"family": "transformer_ft", "arch": arch, "clf": None}

    fe_match = re.match(r"^(bert|roberta|deberta|distilbert|mbert|xlmr)_fe_(lr|svm)$", model_id)
    if fe_match:
        return {
            "family": "transformer_fe",
            "arch": fe_match.group(1),
            "clf": fe_match.group(2),
        }

    for prefix in ("tfidf", "fasttext", "mpnet"):
        if model_id.startswith(f"{prefix}_"):
            return {"family": prefix, "arch": None, "clf": model_id.split("_", 1)[1]}

    raise ValueError(f"Unknown model id: {model_id}")


class ExperimentRunner:
    """Run a single (dataset, model, fold, seed) experiment."""

    def __init__(self, config: dict | None = None, device: str = "auto") -> None:
        self.config = config or load_config()
        self.device = get_device(device)
        self.fasttext: FastTextEmbedder | None = None
        self.embedder = EmbeddingExtractor(self.config, self.device)

    def _get_texts_labels(self, dataset_name: str, indices: np.ndarray):
        df = load_dataset(dataset_name, self.config)
        text_col = self.config["data"]["text_column"]
        label_col = self.config["data"]["label_column"]
        subset = df.iloc[indices]
        texts = subset[text_col].astype(str).tolist()
        labels = subset[label_col].astype(int).tolist()
        return texts, labels

    def run_single(
        self,
        dataset: str,
        model_id: str,
        fold: int,
        seed: int,
    ) -> dict:
        set_seed(seed)
        parsed = parse_model_id(model_id)
        folds = get_stratified_folds(self.config)
        split = folds[fold]
        train_texts, train_labels = self._get_texts_labels(dataset, split.train_idx)
        test_texts, test_labels = self._get_texts_labels(dataset, split.test_idx)

        family = parsed["family"]

        if family == "tfidf":
            vectorizer = build_tfidf_vectorizer(self.config)
            x_train, x_test = fit_transform_tfidf(vectorizer, train_texts, test_texts)
            clf = build_classifier(parsed["clf"], self.config, seed)
            y_pred, y_prob = train_predict_sklearn(clf, x_train, train_labels, x_test)

        elif family == "fasttext":
            if self.fasttext is None:
                self.fasttext = FastTextEmbedder(self.config)
            x_train, x_test = self.fasttext.fit_transform(train_texts, test_texts)
            clf = build_classifier(parsed["clf"], self.config, seed)
            y_pred, y_prob = train_predict_sklearn(clf, x_train, train_labels, x_test)

        elif family == "mpnet":
            x_train, x_test = self.embedder.extract_mpnet(
                train_texts, test_texts, dataset, fold
            )
            clf = build_classifier(parsed["clf"], self.config, seed)
            y_pred, y_prob = train_predict_sklearn(clf, x_train, train_labels, x_test)

        elif family == "transformer_fe":
            x_train, x_test = self.embedder.extract_transformer_fe(
                parsed["arch"], train_texts, test_texts, dataset, fold
            )
            clf = build_classifier(parsed["clf"], self.config, seed)
            y_pred, y_prob = train_predict_sklearn(clf, x_train, train_labels, x_test)

        elif family == "transformer_ft":
            model_name = self.config["transformers"][parsed["arch"]]
            result = finetune_transformer(
                model_name,
                train_texts,
                train_labels,
                test_texts,
                self.config,
                seed,
                self.device,
            )
            y_pred, y_prob = result.y_pred, result.y_prob

        else:
            raise ValueError(f"Unsupported family: {family}")

        metrics = compute_metrics(test_labels, y_pred, y_prob)
        return {
            "dataset": dataset,
            "model": model_id,
            "fold": fold,
            "seed": seed,
            **metrics,
        }


def run_experiments(
    datasets: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    folds: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    resume: bool = True,
    device: str = "auto",
    config_path: str | None = None,
    override_path: str | None = None,
) -> None:
    from src.utils.config import load_config_with_overrides

    config = load_config_with_overrides(config_path, override_path)
    results_dir = project_root() / config["project"]["results_dir"]
    checkpoint = CheckpointManager(results_dir)
    runner = ExperimentRunner(config, device=device)

    datasets = list(datasets or config["data"]["datasets"])
    models = list(models or config["models"])
    available_folds = get_stratified_folds(config)
    if folds is None:
        folds = [split.fold for split in available_folds]
    else:
        folds = list(folds)
    seeds = list(seeds or config["cv"]["train_seeds"])

    total = len(datasets) * len(models) * len(folds) * len(seeds)
    completed = 0

    from tqdm import tqdm

    for dataset in datasets:
        for model_id in models:
            for fold in folds:
                for seed in seeds:
                    if resume and checkpoint.is_completed(dataset, model_id, fold, seed):
                        completed += 1
                        continue
                    row = runner.run_single(dataset, model_id, fold, seed)
                    checkpoint.append_result(row)
                    checkpoint.mark_completed(dataset, model_id, fold, seed)
                    completed += 1
                    tqdm.write(
                        f"[{completed}/{total}] {dataset} | {model_id} | fold={fold} | seed={seed} | "
                        f"macro_f1={row['macro_f1']:.4f}"
                    )
