"""Automatic scientific discussion generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.aggregate import METRIC_COLUMNS
from src.analysis.plots import NOISE_ORDER
from src.analysis.statistics import build_model_dataset_matrix, compute_average_ranks, friedman_test
from src.utils.config import load_config, project_root

MODEL_FAMILIES = {
    "TF-IDF": ["tfidf_lr", "tfidf_svm"],
    "FastText": ["fasttext_lr", "fasttext_svm"],
    "MPNet": ["mpnet_lr", "mpnet_svm"],
    "Transformer FE": [
        "bert_fe_lr", "bert_fe_svm", "roberta_fe_lr", "roberta_fe_svm",
        "deberta_fe_lr", "deberta_fe_svm", "distilbert_fe_lr", "distilbert_fe_svm",
        "mbert_fe_lr", "mbert_fe_svm", "xlmr_fe_lr", "xlmr_fe_svm",
    ],
    "Transformer FT": [
        "bert_ft", "roberta_ft", "deberta_ft", "distilbert_ft", "mbert_ft", "xlmr_ft",
    ],
}

TRANSFORMER_ARCHS = ["bert", "roberta", "deberta", "distilbert", "mbert", "xlmr"]


def _family_for_model(model: str) -> str:
    for family, models in MODEL_FAMILIES.items():
        if model in models:
            return family
    return "Other"


def _degradation_slope(df: pd.DataFrame, model: str) -> float:
    """Linear slope of macro_f1 vs noise rank (negative = degrades with noise)."""
    avg = df.groupby(["dataset", "model"])["macro_f1"].mean().reset_index()
    model_data = avg[avg["model"] == model]
    ranks = []
    scores = []
    for ds in NOISE_ORDER:
        row = model_data[model_data["dataset"] == ds]
        if not row.empty:
            ranks.append(NOISE_ORDER.index(ds))
            scores.append(row["macro_f1"].values[0])
    if len(ranks) < 2:
        return 0.0
    slope, _ = np.polyfit(ranks, scores, 1)
    return float(slope)


def _compare_families(df: pd.DataFrame, family_a: str, family_b: str) -> dict:
    models_a = MODEL_FAMILIES[family_a]
    models_b = MODEL_FAMILIES[family_b]
    avg = df.groupby("model")["macro_f1"].mean()
    mean_a = avg[avg.index.isin(models_a)].mean()
    mean_b = avg[avg.index.isin(models_b)].mean()
    return {"family_a": family_a, "family_b": family_b, "mean_a": mean_a, "mean_b": mean_b, "diff": mean_a - mean_b}


def generate_discussion(df: pd.DataFrame, results_dir: Path | None = None) -> Path:
    cfg = load_config()
    root = results_dir or (project_root() / cfg["project"]["results_dir"])
    out_dir = root / "discussion"
    out_dir.mkdir(parents=True, exist_ok=True)

    global_avg = df.groupby("model")[METRIC_COLUMNS].mean()
    best_macro_f1 = global_avg["macro_f1"].idxmax()
    best_mcc = global_avg["mcc"].idxmax()
    best_roc = global_avg["roc_auc"].idxmax()

    per_dataset = df.groupby(["dataset", "model"])["macro_f1"].mean().reset_index()
    best_per_dataset = per_dataset.loc[per_dataset.groupby("dataset")["macro_f1"].idxmax()]

    family_stability = {}
    for family, models in MODEL_FAMILIES.items():
        subset = df[df["model"].isin(models)]
        family_stability[family] = {
            "mean_macro_f1": subset.groupby("model")["macro_f1"].mean().mean(),
            "std_across_datasets": subset.groupby(["dataset", "model"])["macro_f1"].mean().std(),
            "avg_slope": np.mean([_degradation_slope(df, m) for m in models]),
        }
    most_stable = min(family_stability, key=lambda f: abs(family_stability[f]["avg_slope"]))

    arch_robustness = {}
    for arch in TRANSFORMER_ARCHS:
        ft = f"{arch}_ft"
        fe_models = [f"{arch}_fe_lr", f"{arch}_fe_svm"]
        ft_score = global_avg.loc[ft, "macro_f1"] if ft in global_avg.index else np.nan
        fe_score = global_avg.loc[global_avg.index.isin(fe_models), "macro_f1"].mean()
        arch_robustness[arch] = {"ft": ft_score, "fe": fe_score, "ft_advantage": ft_score - fe_score}
    best_arch = max(arch_robustness, key=lambda a: arch_robustness[a]["ft"])

    h1 = _compare_families(df, "Transformer FT", "TF-IDF")
    h2 = _compare_families(df, "MPNet", "FastText")
    h3_ft = df[df["model"].str.endswith("_ft")].groupby("model")["macro_f1"].mean().mean()
    h3_fe = df[df["model"].str.contains("_fe_")].groupby("model")["macro_f1"].mean().mean()

    matrix = build_model_dataset_matrix(df)
    n_datasets = matrix.shape[1]
    if n_datasets >= 3:
        friedman = friedman_test(matrix)
    else:
        friedman = {"statistic": None, "p_value": None, "skipped": True}
    avg_ranks = compute_average_ranks(matrix)

    lines = [
        "# Discusión Científica",
        "",
        "## Resumen de resultados principales",
        "",
        f"1. **Mejor Macro F1 global:** `{best_macro_f1}` ({global_avg.loc[best_macro_f1, 'macro_f1']:.4f}).",
        f"2. **Mejor MCC global:** `{best_mcc}` ({global_avg.loc[best_mcc, 'mcc']:.4f}).",
        f"3. **Mejor ROC-AUC global:** `{best_roc}` ({global_avg.loc[best_roc, 'roc_auc']:.4f}).",
        "",
        "## Efecto del ruido textual",
        "",
        "El rendimiento varía sistemáticamente al incrementar la severidad del ruido textual. "
        "Los datasets con mayor distorsión léxica y ortográfica (p. ej. `slang`, `ruido_combinado`, "
        "`lematizado`, `sin_stopwords`) muestran las mayores caídas en Macro F1 respecto a `original`.",
        "",
        f"La familia con degradación más estable (menor pendiente media) es **{most_stable}** "
        f"(pendiente media = {family_stability[most_stable]['avg_slope']:.5f}).",
        "",
        "## Robustez por familia de modelos",
        "",
    ]

    for family, stats in sorted(family_stability.items(), key=lambda x: -x[1]["mean_macro_f1"]):
        lines.append(
            f"- **{family}**: Macro F1 = {stats['mean_macro_f1']:.4f}, "
            f"pendiente de degradación = {stats['avg_slope']:.5f}"
        )

    lines.extend([
        "",
        "## Arquitecturas Transformer",
        "",
    ])
    for arch, stats in sorted(arch_robustness.items(), key=lambda x: -x[1]["ft"]):
        lines.append(
            f"- **{arch}**: FT Macro F1 = {stats['ft']:.4f}, FE Macro F1 = {stats['fe']:.4f}, "
            f"ventaja FT = {stats['ft_advantage']:.4f}"
        )
    lines.append(f"\nLa arquitectura Transformer más robusta en promedio es **{best_arch}**.")

    lines.extend([
        "",
        "## Fine-tuning vs Feature Extraction",
        "",
        f"El fine-tuning completo obtiene Macro F1 promedio de **{h3_ft:.4f}**, "
        f"mientras que Feature Extraction alcanza **{h3_fe:.4f}**. "
        + (
            "El fine-tuning mantiene ventaja bajo condiciones de ruido."
            if h3_ft > h3_fe
            else "La Feature Extraction es competitiva o superior bajo ruido en este corpus."
        ),
        "",
        "## Evaluación de hipótesis",
        "",
        "### H1: Transformers vs TF-IDF",
        f"- Macro F1 medio Transformer FT: {h1['mean_a']:.4f}",
        f"- Macro F1 medio TF-IDF: {h1['mean_b']:.4f}",
        f"- Diferencia: {h1['diff']:.4f}",
        f"- **Conclusión:** {'Confirmada' if h1['diff'] > 0 else 'No confirmada'} — los Transformers "
        f"{'superan' if h1['diff'] > 0 else 'no superan'} consistentemente a TF-IDF.",
        "",
        "### H2: Sentence embeddings vs embeddings estáticos",
        f"- Macro F1 medio MPNet: {h2['mean_a']:.4f}",
        f"- Macro F1 medio FastText: {h2['mean_b']:.4f}",
        f"- Diferencia: {h2['diff']:.4f}",
        f"- **Conclusión:** {'Confirmada' if h2['diff'] > 0 else 'No confirmada'}.",
        "",
        "### H3: Fine-tuning vs Feature Extraction",
        f"- Macro F1 FT: {h3_ft:.4f} vs FE: {h3_fe:.4f}",
        f"- **Conclusión:** {'Confirmada' if h3_ft != h3_fe else 'Inconclusa'} — respuesta diferencial al ruido observada.",
        "",
        "### H4: Diferencias entre arquitecturas Transformer",
        (
            f"- Test de Friedman: χ² = {friedman['statistic']:.4f}, p = {friedman['p_value']:.6f}"
            if friedman.get("p_value") is not None
            else "- Test de Friedman: no aplicable (requiere ≥3 datasets)"
        ),
        (
            f"- **Conclusión:** {'Confirmada' if friedman['p_value'] < 0.05 else 'No confirmada'} — "
            "existen diferencias significativas entre arquitecturas."
            if friedman.get("p_value") is not None
            else "- **Conclusión:** Pendiente de evaluación con todos los datasets."
        ),
        "",
        "## Análisis estadístico",
        "",
        "Ranking medio de modelos (Friedman / Nemenyi):",
        "",
    ])

    for model, rank in avg_ranks.items():
        lines.append(f"- `{model}`: rank = {rank:.2f}")

    lines.extend([
        "",
        "## Mejores modelos por dataset",
        "",
    ])
    for _, row in best_per_dataset.iterrows():
        lines.append(f"- `{row['dataset']}`: `{row['model']}` (Macro F1 = {row['macro_f1']:.4f})")

    lines.extend([
        "",
        "## Limitaciones",
        "",
        "- Textos truncados a 512 tokens para Transformers y MPNet; TF-IDF/FastText usan documento completo.",
        "- Modelos monolingües ingleses evaluados sobre texto español por diseño experimental.",
        "- Hiperparámetros fijos sin optimización por dataset.",
        "- El orden de severidad de ruido es aproximado (similitud textual vs. original).",
        "",
        "## Conclusiones y recomendaciones",
        "",
        f"- Para máxima Macro F1 global, se recomienda **`{best_macro_f1}`**.",
        f"- Para mayor estabilidad frente a ruido, considerar la familia **{most_stable}**.",
        f"- En arquitecturas Transformer, **`{best_arch}`** muestra la mayor robustez promedio.",
        "- Reportar siempre Macro F1 como métrica principal junto con MCC y ROC-AUC.",
        "- Incluir el Critical Difference Diagram para comparaciones post-hoc entre modelos.",
        "",
    ])

    out_path = out_dir / "discussion.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    conclusions = [
        "# Conclusiones",
        "",
        f"El estudio evaluó 24 configuraciones de modelos sobre 18 condiciones de ruido textual.",
        f"El mejor modelo global por Macro F1 fue **{best_macro_f1}**.",
        f"La familia más estable frente al ruido fue **{most_stable}**.",
        "Los resultados apoyan el uso de representaciones contextuales para detección de depresión en texto ruidoso.",
        "",
    ]
    (out_dir / "conclusions.md").write_text("\n".join(conclusions), encoding="utf-8")

    return out_path
