# Textual Scene Experiments

This repository keeps legacy notebooks as experiment records while moving shared training, evaluation, metric, and data code into Python modules.

## Repository Layout

```text
src/textual_scene/          Shared Python package
  data.py                   CSV/JSON loading, gold_order parsing, split helpers
  permutations.py           Canonical 24-class order mapping
  metrics.py                Shared order metrics
  models.py                 Model factory interfaces
  trainer.py                Training orchestration and output writing
  evaluator.py              Evaluation orchestration and prediction writing
scripts/                    Notebook-free entry points
configs/                    YAML experiment configs
notebooks/                  Thin Colab runners
experiments/legacy_notebooks/
                            Preserved historical notebooks
outputs/                    Generated runs, metrics, predictions, checkpoints
```

## Install

Local:

```bash
pip install -e .
pip install -r requirements.txt
```

Colab:

```bash
git clone <repo-url>
cd textual_scene
pip install -r requirements.txt
pip install -e .
```

The notebook at `notebooks/run_experiment_colab.ipynb` mounts Google Drive, clones or pulls the repo, installs dependencies, runs a script, and inspects metrics and predictions. It should not redefine datasets, model classes, or training functions.

## Train

Smoke-test the pipeline with 32 synthetic samples:

```bash
python scripts/train_order_head.py --config configs/smoke32.yaml
```

Run the frozen backbone order-head config after filling in `data.path` and model options:

```bash
python scripts/train_order_head.py --config configs/frozen_order_head.yaml
```

Run the LoRA order-head config:

```bash
python scripts/train_order_head.py --config configs/lora_order_head.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/frozen_order_head.yaml \
  --checkpoint outputs/frozen_order_head/checkpoints/last
```

For the local smoke test:

```bash
python scripts/evaluate.py \
  --config configs/smoke32.yaml \
  --checkpoint outputs/smoke32/checkpoints/last
```

## Config Files

Experiment differences should live in YAML, not copied Python code. Each config uses these top-level sections:

```yaml
experiment_name:
model:
data:
training:
evaluation:
output_dir:
seed:
```

Important data settings:

```yaml
data:
  path: /path/to/train.csv
  id_column: sample_id
  order_column: gold_order
  image_columns: [image_1, image_2, image_3, image_4]
  valid_ratio: 0.1
  split_path: outputs/splits/experiment_seed42.json
```

`gold_order` may be formatted like `1 2 3 4`, `1,2,3,4`, or `[1, 2, 3, 4]`.

## Outputs

Each run writes:

```text
outputs/<experiment_name>/
  config.yaml
  train_log.csv
  metrics.json
  predictions.csv
  run_summary.txt
  checkpoints/
```

`predictions.csv` contains:

```text
sample_id
gold_order
predicted_order
gold_class
predicted_class
exact_correct
first_correct
last_correct
position_accuracy
```

## Legacy Notebooks

Historical notebooks are preserved under `experiments/legacy_notebooks/`. They remain useful as implementation records and result references, but new experiments should call the Python scripts and change only YAML configuration values.
