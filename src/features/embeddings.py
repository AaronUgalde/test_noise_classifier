"""MPNet and Transformer feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from src.utils.config import project_root


def _encode_mpnet(
    model_name: str,
    texts: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=str(device))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    # SentenceTransformer handles truncation internally; enforce max tokens via model max seq
    _ = max_length
    return embeddings.astype(np.float32)


def _get_hf_tokenizer(model_name: str):
    if "deberta" in model_name.lower():
        from transformers import DebertaV2Tokenizer

        return DebertaV2Tokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def _encode_transformer_cls(
    model_name: str,
    texts: list[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    tokenizer = _get_hf_tokenizer(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings).astype(np.float32)


class EmbeddingExtractor:
    """Extract MPNet or frozen Transformer [CLS] embeddings."""

    def __init__(self, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.cache_dir = project_root() / "results" / "cache" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self, dataset: str, fold: int, model_key: str, split: str
    ) -> Path:
        return self.cache_dir / f"{dataset}_{model_key}_fold{fold}_{split}.npy"

    def extract_mpnet(
        self,
        train_texts: list[str],
        test_texts: list[str],
        dataset: str,
        fold: int,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        model_name = self.config["sentence_transformer"]["mpnet"]
        train_path = self._cache_path(dataset, fold, "mpnet", "train")
        test_path = self._cache_path(dataset, fold, "mpnet", "test")

        if use_cache and train_path.exists() and test_path.exists():
            return np.load(train_path), np.load(test_path)

        max_length = self.config["text"]["max_length"]
        x_train = _encode_mpnet(model_name, train_texts, max_length, self.device)
        x_test = _encode_mpnet(model_name, test_texts, max_length, self.device)

        if use_cache:
            np.save(train_path, x_train)
            np.save(test_path, x_test)
        return x_train, x_test

    def extract_transformer_fe(
        self,
        arch: str,
        train_texts: list[str],
        test_texts: list[str],
        dataset: str,
        fold: int,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        model_name = self.config["transformers"][arch]
        train_path = self._cache_path(dataset, fold, f"{arch}_fe", "train")
        test_path = self._cache_path(dataset, fold, f"{arch}_fe", "test")

        if use_cache and train_path.exists() and test_path.exists():
            return np.load(train_path), np.load(test_path)

        max_length = self.config["text"]["max_length"]
        batch_size = self.config["finetune"]["batch_size"]
        x_train = _encode_transformer_cls(
            model_name, train_texts, max_length, self.device, batch_size
        )
        x_test = _encode_transformer_cls(
            model_name, test_texts, max_length, self.device, batch_size
        )

        if use_cache:
            np.save(train_path, x_train)
            np.save(test_path, x_test)
        return x_train, x_test
