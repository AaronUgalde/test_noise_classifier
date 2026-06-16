# Robustez al Ruido Textual en Detección de Depresión

Pipeline experimental reproducible para evaluar cómo el ruido textual afecta diferentes representaciones lingüísticas y arquitecturas Transformer en clasificación binaria de depresión (español, 18 condiciones de ruido, 24 modelos).

---

## Requisitos de hardware (RTX 5080 + i9)

| Recurso | Mínimo recomendado |
|---------|-------------------|
| GPU | NVIDIA RTX 5080 (16 GB VRAM) o equivalente |
| CPU | Intel i9 (8+ cores) |
| RAM | 16 GB (32 GB recomendado) |
| Disco | ~25 GB libres (datasets + FastText ~4 GB + modelos HuggingFace + resultados) |
| SO | Linux o Windows 11 con drivers NVIDIA actualizados |
| CUDA | 12.x compatible con tu versión de PyTorch |

---

## Instalación (Linux / Windows con CUDA)

### 1. Clonar o copiar el proyecto

```bash
cd test_noise_classifier
```

### 2. Entorno virtual

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. PyTorch con soporte CUDA

Instala PyTorch **antes** que el resto, con la build CUDA adecuada para tu sistema. Consulta [pytorch.org](https://pytorch.org/get-started/locally/) si la siguiente línea no coincide con tu CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Resto de dependencias

```bash
pip install -r requirements.txt
```

### 5. Verificar que CUDA funciona

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Debe imprimir `CUDA: True` y el nombre de tu GPU (p. ej. `NVIDIA GeForce RTX 5080`).

### 6. Variable de entorno para gráficos (opcional pero recomendada)

**Linux / macOS:**

```bash
export MPLCONFIGDIR=./results/.matplotlib
```

**Windows (PowerShell):**

```powershell
$env:MPLCONFIGDIR = ".\results\.matplotlib"
```

> La primera ejecución con modelos `fasttext_*` descargará `cc.es.300.bin` (~4 GB). Los modelos Transformer se descargan de HuggingFace en la primera corrida.

---

## Guía rápida: ejecutar en RTX 5080 + i9

### Paso 1 — Smoke test (~5 min)

Comprueba que el pipeline funciona antes de lanzar las 432 evaluaciones:

```bash
python scripts/smoke_test.py --device cuda
```

Ejecuta 4 modelos en el dataset `original`. Si termina sin errores, puedes continuar.

### Paso 2 — Experimento completo (perfil día, ~5–10 h)

Perfil recomendado para una sola máquina con 5080:

- **18 datasets** (todos)
- **24 modelos** (todos)
- Holdout estratificado **80/20** (sin k-fold)
- **1 semilla** (42)
- Fine-tuning con **50% del train** (`train_fraction: 0.5`); evaluación en test completo
- **432 runs** en total

```bash
python scripts/run_experiments.py \
  --override config/experiment_day.yaml \
  --device cuda \
  --resume
```

`--resume` salta runs ya completados si interrumpes la ejecución (estado en `results/checkpoints.json`).

### Paso 3 — Análisis, tablas y figuras (~5 min)

```bash
python scripts/run_analysis.py
```

### Salidas generadas

| Artefacto | Ruta |
|-----------|------|
| Resultados por run | `results/folds/results_by_fold.csv` |
| Resumen agregado | `results/results_summary.csv`, `.json` |
| Tablas publicación | `results/tables/table1_full_results.csv`, `table2_global_ranking.csv` |
| Figuras 1–6 | `results/figures/` |
| Tests estadísticos | `results/statistics/friedman_nemenyi.json` |
| Discusión automática | `results/discussion/discussion.md` |

---

## Ejecución por fases (opcional)

Si prefieres lanzar el experimento en bloques:

```bash
# Fase 1 — modelos clásicos y MPNet (~1–2 h)
python scripts/run_experiments.py \
  --override config/experiment_day.yaml --device cuda --resume \
  --models tfidf_lr tfidf_svm fasttext_lr fasttext_svm mpnet_lr mpnet_svm

# Fase 2 — Transformer Feature Extraction (~2–3 h)
python scripts/run_experiments.py \
  --override config/experiment_day.yaml --device cuda --resume \
  --models bert_fe_lr bert_fe_svm roberta_fe_lr roberta_fe_svm deberta_fe_lr deberta_fe_svm \
           distilbert_fe_lr distilbert_fe_svm mbert_fe_lr mbert_fe_svm xlmr_fe_lr xlmr_fe_svm

# Fase 3 — Fine-tuning (~2.5–4 h)
python scripts/run_experiments.py \
  --override config/experiment_day.yaml --device cuda --resume \
  --models bert_ft roberta_ft deberta_ft distilbert_ft mbert_ft xlmr_ft

# Análisis
python scripts/run_analysis.py
```

---

## Perfiles de configuración

| Archivo | Validación | Semillas | FT train | Runs | Tiempo (5080 + i9) |
|---------|------------|----------|----------|------|---------------------|
| `experiment_day.yaml` | Holdout 80/20 | 1 | 50% del train | 432 | **~5–10 h** |
| `experiment.yaml` | 5-fold CV | 3 | 100% | 6.480 | ~2–3 días |

Configuración base: [`config/experiment.yaml`](config/experiment.yaml).  
Override de perfil día: [`config/experiment_day.yaml`](config/experiment_day.yaml).

### Publicación rigurosa (5-fold × 3 semillas)

Sin `--override`, usando el protocolo completo:

```bash
python scripts/run_experiments.py --device cuda --resume
python scripts/run_analysis.py
```

---

## Modelos evaluados (24)

| Familia | Modelos |
|---------|---------|
| TF-IDF | `tfidf_lr`, `tfidf_svm` |
| FastText | `fasttext_lr`, `fasttext_svm` |
| MPNet | `mpnet_lr`, `mpnet_svm` |
| Transformer FE | `{bert,roberta,deberta,distilbert,mbert,xlmr}_fe_{lr,svm}` |
| Transformer FT | `{bert,roberta,deberta,distilbert,mbert,xlmr}_ft` |

---

## Diseño experimental

- **Métrica principal:** Macro F1
- **Anti-leakage:** vectorizadores y embeddings se ajustan solo en train de cada split
- **Fine-tuning:** early stopping (patience=2) con 10% del train reservado para validación interna
- **Class weights:** balanceados en sklearn y en la loss de fine-tuning
- **Textos Transformer/MPNet:** truncados a 512 tokens; TF-IDF/FastText usan documento completo

---

## Flags CLI

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
| `--device cuda` | Fuerza GPU NVIDIA |
| `--resume` | Continúa desde `results/checkpoints.json` |
| `--no-resume` | Reejecuta todo desde cero |

---

## Reanudar tras una interrupción

Usa el **mismo** `--override` y `--device` que al iniciar:

```bash
python scripts/run_experiments.py \
  --override config/experiment_day.yaml \
  --device cuda \
  --resume
```

Para empezar de cero, borra el estado previo:

```bash
rm -rf results/folds results/checkpoints.json results/cache
```

---

## Solución de problemas (5080 / CUDA)

| Problema | Solución |
|----------|----------|
| `CUDA: False` | Reinstala PyTorch con build CUDA; actualiza drivers NVIDIA |
| Out of memory en FT | Baja `finetune.batch_size` en `experiment_day.yaml` (p. ej. 8) |
| Descarga lenta de HuggingFace | Opcional: `export HF_TOKEN=tu_token` para mayor rate limit |
| FastText tarda mucho | Normal la primera vez (~4 GB); runs siguientes reutilizan caché |
| Cambié `train_fraction` o config | Usa `--no-resume` o borra `results/checkpoints.json` |

---

## Ejecución en Mac (Apple Silicon)

En Mac usa `--device auto` (MPS) en lugar de `cuda`. El perfil día completo tarda **~20–35 h** en M4, no ~1 día en GPU NVIDIA.

```bash
source .venv/bin/activate
export MPLCONFIGDIR=./results/.matplotlib

python scripts/smoke_test.py --device auto

python scripts/run_experiments.py \
  --override config/experiment_day.yaml \
  --device auto \
  --resume

python scripts/run_analysis.py
```

Instala PyTorch para Mac desde [pytorch.org](https://pytorch.org/get-started/locally/) (sin índice CUDA).

---

## Estructura del proyecto

```
test_noise_classifier/
├── config/
│   ├── experiment.yaml       # Protocolo publicación (5-fold × 3 semillas)
│   ├── experiment_day.yaml   # Perfil ≤1 día (holdout, 1 semilla)
│   └── smoke_test.yaml       # Smoke test rápido
├── datasets_textos_depresivos/   # 18 CSVs de ruido textual
├── scripts/
│   ├── run_experiments.py    # CLI principal
│   ├── run_analysis.py       # Tablas, figuras, Friedman, discusión
│   └── smoke_test.py
├── src/                      # Código modular
└── results/                  # Salidas (generado al ejecutar)
```

---

## Reproducibilidad

- Hiperparámetros fijos en YAML; sin búsqueda de hiperparámetros
- Misma partición holdout (seed=42) para los 18 datasets
- Checkpoints en `results/checkpoints.json`
- Especifica en publicaciones qué perfil usaste (`experiment.yaml` vs `experiment_day.yaml`)

## Cómo citar

Cita el repositorio e indica la configuración YAML utilizada (`config/experiment.yaml` o `config/experiment_day.yaml`).
