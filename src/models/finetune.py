"""HuggingFace fine-tuning for Transformer classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


@dataclass
class FinetuneResult:
    y_pred: np.ndarray
    y_prob: np.ndarray


def _get_tokenizer(model_name: str):
    if "deberta" in model_name.lower():
        from transformers import DebertaV2Tokenizer

        return DebertaV2Tokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def _compute_class_weights(labels: list[int]) -> torch.Tensor:
    labels_arr = np.array(labels)
    n_samples = len(labels_arr)
    classes, counts = np.unique(labels_arr, return_counts=True)
    weights = np.zeros(len(classes), dtype=np.float32)
    for cls, count in zip(classes, counts):
        weights[int(cls)] = n_samples / (len(classes) * count)
    return torch.tensor(weights, dtype=torch.float32)


def _tokenize_dataset(tokenizer, texts, max_length: int):
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _subsample_training_data(
    train_texts: list[str],
    train_labels: list[int],
    fraction: float,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Stratified subsample of training data (fine-tuning only)."""
    if fraction >= 1.0:
        return train_texts, train_labels

    indices = np.arange(len(train_labels))
    selected, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=seed,
        stratify=train_labels,
    )
    selected = np.sort(selected)
    texts = [train_texts[i] for i in selected]
    labels = [train_labels[i] for i in selected]
    return texts, labels


def finetune_transformer(
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    config: dict,
    seed: int,
    device: torch.device,
) -> FinetuneResult:
    ft_cfg = config["finetune"]
    max_length = config["text"]["max_length"]

    tokenizer = _get_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "0", 1: "1"}, label2id={"0": 0, "1": 1}
    )

    train_fraction = ft_cfg.get("train_fraction", 1.0)
    train_texts, train_labels = _subsample_training_data(
        train_texts, train_labels, train_fraction, seed
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=ft_cfg["val_split"],
        random_state=ft_cfg["val_seed"],
        stratify=train_labels,
    )

    train_enc = _tokenize_dataset(tokenizer, train_texts, max_length)
    val_enc = _tokenize_dataset(tokenizer, val_texts, max_length)
    test_enc = _tokenize_dataset(tokenizer, test_texts, max_length)

    train_ds = Dataset.from_dict({**train_enc, "labels": train_labels})
    val_ds = Dataset.from_dict({**val_enc, "labels": val_labels})
    test_ds = Dataset.from_dict(test_enc)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = _compute_class_weights(train_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    use_fp16 = device.type == "cuda"
    total_steps = max(
        1,
        (len(train_ds) // ft_cfg["batch_size"]) * ft_cfg["num_epochs"],
    )
    warmup_steps = int(total_steps * ft_cfg["warmup_ratio"])
    training_args = TrainingArguments(
        output_dir=f"results/cache/finetune/{model_name.replace('/', '_')}_{seed}",
        learning_rate=ft_cfg["learning_rate"],
        per_device_train_batch_size=ft_cfg["batch_size"],
        per_device_eval_batch_size=ft_cfg["batch_size"],
        num_train_epochs=ft_cfg["num_epochs"],
        weight_decay=ft_cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=ft_cfg["metric_for_best"],
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        logging_steps=50,
        report_to=[],
        fp16=use_fp16,
        dataloader_pin_memory=device.type == "cuda",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=ft_cfg["early_stopping_patience"])
        ],
    )

    trainer.train()

    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = np.argmax(probs, axis=1)
    y_prob = probs[:, 1]
    return FinetuneResult(y_pred=y_pred.astype(int), y_prob=y_prob.astype(float))
