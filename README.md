# Robustez al Ruido Textual en Detección de Depresión

Pipeline experimental reproducible que mide cómo **18 tipos de ruido textual** (errores de teclado, abreviaciones, slang, etc.) afectan el rendimiento de **24 modelos** de clasificación en la **detección binaria de depresión** a partir de textos en español. Produce métricas, tablas, figuras y análisis estadístico listos para publicación.

**Documentación técnica detallada:** [docs/DOCUMENTACION_TECNICA.md](docs/DOCUMENTACION_TECNICA.md)

---

## Tabla de contenidos

- [Inicio rápido](#inicio-rápido)
- [Qué evalúa el proyecto](#qué-evalúa-el-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Salidas generadas](#salidas-generadas)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Solución de problemas](#solución-de-problemas)
- [Reproducibilidad y citación](#reproducibilidad-y-citación)

---

## Inicio rápido

Tras [instalar](#instalación) dependencias y activar el entorno virtual:

```bash
# 1. Verificar que el pipeline funciona (~5 min)
python scripts/smoke_test.py --device cuda    # GPU NVIDIA
# python scripts/smoke_test.py --device auto  # Mac Apple Silicon (MPS)

# 2. Experimento completo — perfil día (~5–10 h en GPU 16 GB)
python scripts/run_experiments.py \
  --override config/experiment_day.yaml \
  --device cuda

# 3. Análisis, tablas y figuras (~5 min)
python scripts/run_analysis.py
```

En Mac, sustituye `--device cuda` por `--device auto`. Los experimentos **reanudan automáticamente** si se interrumpen (estado en `results/checkpoints.json`).

---

## Qué evalúa el proyecto

| Dimensión | Detalle |
|-----------|---------|
| **Tarea** | Clasificación binaria: depresión (1) vs. sin depresión (0) |
| **Datos** | 18 variantes del mismo corpus con distinto ruido textual |
| **Modelos** | 24 modelos en 5 familias (TF-IDF, FastText, MPNet, Transformer FE, Transformer FT) |
| **Métrica principal** | Macro F1 |
| **Runs (perfil día)** | 432 = 18 datasets × 24 modelos × 1 split × 1 semilla |

**Diseño clave:** las particiones train/test son idénticas para los 18 datasets (calculadas desde `original`), de modo que las diferencias de rendimiento reflejan solo el efecto del ruido textual.

---

## Requisitos

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| Python | 3.10+ | 3.11+ |
| GPU | 8 GB VRAM | NVIDIA 16 GB (CUDA 12.x) |
| RAM | 16 GB | 32 GB |
| Disco | ~25 GB libres | SSD |
| SO | Linux, Windows 11, macOS | Linux o Windows con drivers NVIDIA |

> La primera ejecución descarga FastText (~4 GB) y modelos HuggingFace. Ver [documentación técnica](docs/DOCUMENTACION_TECNICA.md#8-dependencias-y-recursos-externos).

---

## Instalación

### 1. Entorno virtual

**Linux / macOS:**

```bash
cd test_noise_classifier
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. PyTorch

Instala PyTorch **antes** que el resto. Consulta [pytorch.org](https://pytorch.org/get-started/locally/) si tu versión de CUDA difiere:

```bash
# GPU NVIDIA (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Mac Apple Silicon — instalar desde pytorch.org sin índice CUDA
```

### 3. Dependencias del proyecto

```bash
pip install -r requirements.txt
```

### 4. Verificar GPU (opcional)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 5. Variable de entorno para gráficos (recomendada)

```bash
export MPLCONFIGDIR=./results/.matplotlib   # Linux / macOS
# $env:MPLCONFIGDIR = ".\results\.matplotlib"  # Windows PowerShell
```

---

## Ejecución

### Perfiles de configuración

| Perfil | Archivo | Splits | Semillas | Runs | Tiempo estimado |
|--------|---------|--------|----------|------|-----------------|
| **Día** (recomendado) | `config/experiment_day.yaml` | Holdout 80/20 | 1 | 432 | ~5–10 h (GPU 16 GB) |
| **Publicación** | `config/experiment.yaml` | 5-fold CV | 3 | 6.480 | ~2–3 días |

```bash
# Perfil día (una máquina, un día)
python scripts/run_experiments.py \
  --override config/experiment_day.yaml \
  --device cuda

# Protocolo publicación riguroso (5-fold × 3 semillas)
python scripts/run_experiments.py --device cuda
```

### Ejecución por fases (opcional)

```bash
# Fase 1 — clásicos y MPNet (~1–2 h)
python scripts/run_experiments.py --override config/experiment_day.yaml --device cuda \
  --models tfidf_lr tfidf_svm fasttext_lr fasttext_svm mpnet_lr mpnet_svm

# Fase 2 — Transformer Feature Extraction (~2–3 h)
python scripts/run_experiments.py --override config/experiment_day.yaml --device cuda \
  --models bert_fe_lr bert_fe_svm roberta_fe_lr roberta_fe_svm deberta_fe_lr deberta_fe_svm \
           distilbert_fe_lr distilbert_fe_svm mbert_fe_lr mbert_fe_svm xlmr_fe_lr xlmr_fe_svm

# Fase 3 — Fine-tuning (~2.5–4 h)
python scripts/run_experiments.py --override config/experiment_day.yaml --device cuda \
  --models bert_ft roberta_ft deberta_ft distilbert_ft mbert_ft xlmr_ft

python scripts/run_analysis.py
```

### Flags CLI útiles

```bash
python scripts/run_experiments.py \
  --override config/experiment_day.yaml \
  --datasets original abreviaciones slang \
  --models tfidf_lr bert_ft \
  --device cuda \
  --no-resume
```

| Flag | Descripción |
|------|-------------|
| `--override` | YAML parcial que sobrescribe `experiment.yaml` |
| `--config` | Ruta al config base (default: `config/experiment.yaml`) |
| `--datasets` | Subconjunto de datasets (default: los 18) |
| `--models` | Subconjunto de modelos (default: los 24) |
| `--device` | `auto`, `cuda`, `mps` o `cpu` |
| `--no-resume` | Reejecutar todo desde cero (por defecto reanuda) |

### Reanudar tras una interrupción

Usa el mismo comando; la reanudación es automática. Para empezar de cero:

```bash
rm -rf results/folds results/checkpoints.json results/cache
python scripts/run_experiments.py --override config/experiment_day.yaml --device cuda --no-resume
```

### Modelos evaluados (24)

| Familia | Modelos |
|---------|---------|
| TF-IDF | `tfidf_lr`, `tfidf_svm` |
| FastText | `fasttext_lr`, `fasttext_svm` |
| MPNet | `mpnet_lr`, `mpnet_svm` |
| Transformer FE | `{bert,roberta,deberta,distilbert,mbert,xlmr}_fe_{lr,svm}` |
| Transformer FT | `{bert,roberta,deberta,distilbert,mbert,xlmr}_ft` |

---

## Salidas generadas

| Artefacto | Ruta |
|-----------|------|
| Resultados por run | `results/folds/results_by_fold.csv` |
| Resumen agregado | `results/results_summary.csv`, `results_summary.json` |
| Tablas publicación | `results/tables/table1_full_results.csv`, `table2_global_ranking.csv` |
| Figuras 1–6 | `results/figures/` |
| Tests estadísticos | `results/statistics/friedman_nemenyi.json` |
| Discusión automática | `results/discussion/discussion.md` |

---

## Estructura del proyecto

```
test_noise_classifier/
├── config/
│   ├── experiment.yaml         # Protocolo publicación (5-fold × 3 semillas)
│   ├── experiment_day.yaml     # Perfil ≤1 día (holdout, 1 semilla)
│   └── smoke_test.yaml         # Smoke test rápido
├── datasets_textos_depresivos/ # 18 CSVs de ruido textual
├── docs/
│   └── DOCUMENTACION_TECNICA.md
├── scripts/
│   ├── run_experiments.py      # CLI principal
│   ├── run_analysis.py         # Tablas, figuras, Friedman, discusión
│   └── smoke_test.py           # Validación rápida (~5 min)
├── src/                        # Código modular (data, features, models, evaluation, analysis)
└── results/                    # Salidas (generado al ejecutar)
```

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `CUDA: False` | Reinstala PyTorch con build CUDA; actualiza drivers NVIDIA |
| Out of memory en fine-tuning | Baja `finetune.batch_size` en `experiment_day.yaml` (p. ej. 8) |
| Descarga lenta de HuggingFace | Opcional: `export HF_TOKEN=tu_token` |
| FastText tarda mucho la primera vez | Normal (~4 GB); runs siguientes reutilizan caché |
| Cambié `train_fraction` o config | Usa `--no-resume` o borra `results/checkpoints.json` |
| Mac muy lento | Esperado: perfil día ~20–35 h con `--device auto` (MPS) |

---

## Reproducibilidad y citación

- Hiperparámetros fijos en YAML; sin búsqueda de hiperparámetros.
- Misma partición holdout (seed=42) para los 18 datasets.
- Especifica en publicaciones qué perfil usaste (`experiment.yaml` vs `experiment_day.yaml`).
- Cita el repositorio e indica la configuración YAML utilizada.

Para detalles de arquitectura, diseño experimental, catálogo de modelos y pipeline de análisis, consulta [docs/DOCUMENTACION_TECNICA.md](docs/DOCUMENTACION_TECNICA.md).
