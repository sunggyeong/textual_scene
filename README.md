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

For structure tests without installing the heavy experiment stack:

```bash
pip install -e ".[dev]"
pytest -q
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

The frozen backbone and LoRA order-head files are example configs for the next experiment PR. They document the intended YAML shape, but the order-head model implementations are intentionally blocked in this structure PR:

```bash
configs/frozen_order_head.example.yaml
configs/lora_order_head.example.yaml
```

## Evaluate

For the local smoke test, evaluation uses `evaluation.split: validation` and reloads the saved split file so the evaluation target matches the training validation target:

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
  train/
    train_log.csv
    metrics.json
    predictions.csv
    run_summary.txt
  eval/
    metrics.json
    predictions.csv
  checkpoints/
```

`train/metrics.json` and `eval/metrics.json` are intentionally separated so an independent evaluation command cannot overwrite the metrics produced during training. `predictions.csv` contains:

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

## Current Scope

This PR is a repository-structure PR. The smoke baseline is present only to verify that data loading, split persistence, metric computation, checkpoint writing, and CLI entry points work end to end.

The `training` YAML values such as `epochs`, `batch_size`, `gradient_accumulation_steps`, `learning_rate`, and `save_steps` are retained as the intended config contract. They will be wired to the real order-head trainer in the next experiment PR.

Transformer-backed frozen/LoRA order-head checkpoint restoration is not implemented yet. Passing those model types raises `NotImplementedError` rather than silently loading a base model.
