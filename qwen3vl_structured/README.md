# Qwen3-VL 8B 4-Task Structured Pipeline

Local NVIDIA GPU pipeline for Qwen3-VL-8B 4-bit QLoRA training and
`pairwise + first + last` structured decoding.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch wheel for your local driver before running
training.

Qwen3-VL currently requires recent Transformers support. The Qwen model card and
official repo recommend `transformers>=4.57.0`; if your install cannot load the
model, install the latest Transformers from source.

## Configure

Edit `configs/pilot.json` or `configs/full_train.json`:

```json
{
  "data_root": "datasets/snuaichallenge_data",
  "output_dir": "outputs/qwen3vl_8b_4task_pilot"
}
```

Expected dataset:

```text
datasets/snuaichallenge_data/
  train.csv
  test.csv
  sample_submission.csv
  train/<Id>/<Input_n>
  test/<Id>/<Input_n>
```

## Run

Run from this directory so relative paths like `outputs/...` land under
`qwen3vl_structured/`:

```bash
cd qwen3vl_structured
python run_qwen3vl_structured.py --mode check --config configs/pilot.json
python run_qwen3vl_structured.py --mode pilot --config configs/pilot.json
python run_qwen3vl_structured.py --mode train --config configs/full_train.json
python run_qwen3vl_structured.py --mode eval --config configs/full_train.json --checkpoint outputs/qwen3vl_8b_4task_full/checkpoint-1000
python run_qwen3vl_structured.py --mode infer --config configs/full_train.json --checkpoint outputs/qwen3vl_8b_4task_full/checkpoint-1000
python run_qwen3vl_structured.py --mode all --config configs/pilot.json
```

Modes:

- `check`: CUDA, dataset, processor, 4-bit model loading, and LoRA target checks.
- `pilot`: short train run, then evaluates every saved `checkpoint-*` plus `final_adapter`.
- `train`: train LoRA adapter.
- `eval`: evaluate a checkpoint on quick/tuning/holdout splits.
- `infer`: create `submission.csv`.
- `all`: train and evaluate only. Run `infer` explicitly after choosing a checkpoint.

## Task Setup

Tasks:

```text
order
pairwise
first
last
```

Ratios:

```python
{"order": 0.30, "pairwise": 0.40, "first": 0.15, "last": 0.15}
```

Loss weights:

```python
{"order": 1.0, "pairwise": 0.5, "first": 1.5, "last": 1.5}
```

The final decoder does not use direct order generation. It extracts:

```text
P(i before j)
P_first(i)
P_last(i)
```

Then scores all 24 permutations:

```text
score(order) =
    alpha * pairwise_score(order)
  + beta  * log P_first(order[0])
  + gamma * log P_last(order[-1])
```

## Outputs

```text
outputs/<run_name>/
  environment.json
  train_config.json
  dataset_check.json
  lora_modules.txt
  all_module_names.txt
  checkpoint-*
  probability_cache/
  splits/
  quick_metrics.csv
  holdout_metrics.csv
  <checkpoint>_quick_metrics.csv
  <checkpoint>_holdout_metrics.csv
  <checkpoint>_predictions.csv
  decoding_grid.csv
  predictions.csv
  pilot_checkpoint_results.json
  best_config.json
  submission.csv
```

The full config intentionally caps training at `max_train_steps = 4000`; one
full epoch over the balanced 4-task record pool can be much longer on Qwen3-VL.
