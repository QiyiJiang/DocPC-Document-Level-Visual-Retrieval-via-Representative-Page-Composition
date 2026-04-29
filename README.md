# DocPC: Document-Level Visual Retrieval via Representative Page Composition

This project focuses on **document-level visual retrieval**. It converts PDF documents into representative page images (for example `first4`, `first9`, `last4`, `uniform4`, `random4`, `first2_last2`, and `clip4`), then trains and evaluates ColPali/ColQwen-based models with metrics such as P@K, R@K, MRR@K, and NDCG@K.

The repository includes three core capabilities:
- Data construction: generate multiple page-composition images from PDFs and build query-positive mappings.
- Model training: fine-tune ColPali / ColQwen2.5 models.
- Evaluation: run standard evaluation, page-strategy comparison, and scale experiments.

## Directory Overview

- `dataset_generate/`: data generation and enrichment scripts (PDF-to-image, query generation, positive target augmentation, etc.).
- `src/train/`: training scripts (single-GPU/multi-GPU and different data organizations).
- `src/eval/`: evaluation scripts (Milvus/Weaviate retrieval, strategy aggregation, scale experiments).
- `src/colpali_engine/`: model, trainer, loss, and utility code.
- `old_results/`: placeholder for historical results.

## Environment Setup

> This repo is expected to run with `uv`. Prefer `uv run` for Python commands.

1) Create a virtual environment (optional):

```bash
uv venv
source .venv/bin/activate
```

2) Install dependencies (there is no pinned dependency file in this repo, so install based on imports):

```bash
uv pip install torch torchvision transformers pillow tqdm numpy pymupdf pymilvus weaviate-client openai
```

If you train/evaluate ColPali or ColQwen, ensure CUDA, PyTorch, and GPU driver versions are compatible.

## Data Preparation Pipeline

Many scripts use hardcoded paths (for example `/data/docpc_project/...`). Update constants at the top of each script before running.

### 1) Convert PDFs to Representative Images

Common strategy scripts in `dataset_generate/`:
- `pdf_first4_pages_to_image.py`
- `pdf_first9_pages_to_image.py`
- `pdf_last4_pages_to_image.py`
- `pdf_uniform4_pages_to_image.py`
- `pdf_random4_pages_to_image.py`
- `pdf_first2_last2_pages_to_image.py`
- `pdf_clip_select4_pages_to_image.py`

Example:

```bash
uv run python dataset_generate/pdf_first4_pages_to_image.py
uv run python dataset_generate/pdf_first9_pages_to_image.py
```

### 2) Build Similar Groups and Queries

Typical pipeline:

```bash
uv run python dataset_generate/similar_groups_text.py
uv run python dataset_generate/query_create_text.py
uv run python dataset_generate/add_pos_target_for_deepseek_to_query_list.py
```

Notes:
- `similar_groups_text.py`: builds similar text groups.
- `query_create_text.py`: generates queries from keywords/text excerpts.
- `add_pos_target_for_deepseek_to_query_list.py`: writes reverse-mapped positives into `pos_target_for_deepseek`.

## Training

Training scripts are under `src/train/`. Common entry points:
- `train_pdfa.sh` / `train_pdfa_colpali.py`
- `train_pdfa_page.sh` / `train_pdfa_page.py`
- `train_colqwen25_model.py`

Example:

```bash
cd src/train
uv run python train_pdfa_colpali.py \
  --dataset-root /path/to/pdfa \
  --output-dir /path/to/output_model \
  --pretrained-model-name-or-path /path/to/base_model \
  --image-subdir pos_target_for_deepseek_images_first4 \
  --pos-target-column pos_target_for_deepseek
```

To run the batch shell script directly:

```bash
cd src/train
bash train_pdfa.sh
```

## Evaluation

Evaluation scripts are under `src/eval/`:
- `eval_model.py`: unified evaluation (`colpali` / `colqwen`, optional Milvus).
- `eval_page_strategy.py`: page-level retrieval + strategy aggregation.
- `eval_scale_experiment.py`: scale-gradient experiment.
- Batch wrappers: `eval_model.sh`, `eval_page_strategy.sh`, `eval_scale_experiment.sh`.

### 1) General Evaluation (`eval_model.py`)

```bash
cd src/eval
uv run python eval_model.py \
  --model-path /path/to/model \
  --model-type colqwen \
  --image-dir /path/to/images \
  --eval-dataset-path /path/to/query_list_with_pos_target.json \
  --results-dir /path/to/results \
  --use-milvus \
  --pos-target-column pos_target_for_deepseek \
  --full-pool \
  --run-tag demo_run
```

### 2) Page Strategy Evaluation (`eval_page_strategy.py`)

```bash
cd src/eval
uv run python eval_page_strategy.py \
  --model-path /path/to/model \
  --model-type colqwen \
  --eval-dataset-path /path/to/query_list_with_pos_target.json \
  --metadata-path /path/to/image_page_metadata.json \
  --image-dir /path/to/page_images \
  --results-dir /path/to/results \
  --strategy first4 uniform4 last4 \
  --score-agg max \
  --pos-target-column pos_target_for_deepseek \
  --full-pool
```

### 3) Scale Experiment (`eval_scale_experiment.py`)

```bash
cd src/eval
uv run python eval_scale_experiment.py \
  --model-path /path/to/model \
  --image-dir /path/to/images \
  --eval-dataset-path /path/to/query_list_with_pos_target.json \
  --results-dir /path/to/results \
  --pos-target-column pos_target_for_deepseek \
  --sample-ratios 0.2 0.4 0.6 0.8 1.0 \
  --seed 42 \
  --run-tag biology
```

## Outputs and Metrics

Evaluation outputs are written to your `results-dir`. Common files:
- `eval_*.json`: retrieval results per query.
- `metrics_*.json`: metric summaries for each run.
- `page_strategy_summary*.json`: multi-strategy comparison summary.
- `scale_summary_*.json`: scale experiment summary.

Core metrics:
- `P@K`, `R@K`
- `MRR@K`
- `NDCG@K`

## Notes

- Most scripts have hardcoded paths at the top; update them when migrating environments.
- Some `.sh` scripts call `python xxx.py` directly; if needed, replace with `uv run python xxx.py`.
- Page-level evaluation depends on `metadata` mapping (`page_name` ↔ `document_name`).
- When using Milvus, local temporary DB files are created and cleaned up at the end of evaluation.