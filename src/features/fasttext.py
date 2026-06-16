"""FastText document embeddings with TF-IDF weighting."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import project_root


class FastTextEmbedder:
    """Spanish FastText embeddings with TF-IDF-weighted document averaging."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.dim = 300
        self._model = None
        self._vectorizer: TfidfVectorizer | None = None

    @property
    def model_path(self) -> Path:
        cache_dir = project_root() / self.config["fasttext"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / self.config["fasttext"]["model_filename"]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        path = self.model_path
        if not path.exists():
            self._download_model()
        import fasttext

        # Suppress FastText warning about deprecated API
        fasttext.FastText.eprint = lambda x: None  # type: ignore[attr-defined]
        self._model = fasttext.load_model(str(path))

    def _download_model(self) -> None:
        url = self.config["fasttext"]["model_url"]
        gz_path = self.model_path.with_suffix(".bin.gz")
        print(f"Downloading FastText model from {url} ...")
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(self.model_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)

    def _word_vector(self, word: str) -> np.ndarray:
        self._ensure_model()
        vec = self._model.get_word_vector(word)
        if np.allclose(vec, 0):
            return np.zeros(self.dim, dtype=np.float32)
        return vec.astype(np.float32)

    def _document_vector(self, text: str, tfidf_row, feature_names) -> np.ndarray:
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        indices = tfidf_row.indices
        data = tfidf_row.data
        weights = {feature_names[i]: w for i, w in zip(indices, data)}

        weighted_sum = np.zeros(self.dim, dtype=np.float64)
        weight_total = 0.0
        for token in tokens:
            w = weights.get(token, 0.0)
            if w <= 0:
                continue
            weighted_sum += w * self._word_vector(token)
            weight_total += w

        if weight_total == 0:
            vecs = [self._word_vector(t) for t in tokens]
            return np.mean(vecs, axis=0).astype(np.float32)
        return (weighted_sum / weight_total).astype(np.float32)

    def fit_transform(self, train_texts, test_texts):
        self._vectorizer = TfidfVectorizer(
            max_features=self.config["tfidf"]["max_features"],
            ngram_range=(1, 1),
            min_df=self.config["tfidf"]["min_df"],
            sublinear_tf=self.config["tfidf"]["sublinear_tf"],
        )
        tfidf_train = self._vectorizer.fit_transform(train_texts)
        tfidf_test = self._vectorizer.transform(test_texts)
        feature_names = self._vectorizer.get_feature_names_out()

        x_train = np.vstack(
            [
                self._document_vector(text, tfidf_train[i], feature_names)
                for i, text in enumerate(train_texts)
            ]
        )
        x_test = np.vstack(
            [
                self._document_vector(text, tfidf_test[i], feature_names)
                for i, text in enumerate(test_texts)
            ]
        )
        return x_train, x_test
