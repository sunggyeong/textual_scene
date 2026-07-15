# Endpoint Contrastive v1

This branch adds endpoint contrastive auxiliary training on top of the best
weighted 4-task order refine adapter.

Use these notebooks in Colab:

```text
endpoint_contrastive_train.ipynb
endpoint_contrastive_eval_infer.ipynb
```

## Initial Adapter

Training starts from the existing best 4-task run:

```python
INITIAL_ADAPTER_ROOT = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_lgt_order_refine_v1/runs/"
    "20260714_003635/lgt_order_refine"
)
```

The notebook resolves the actual adapter automatically in this order:

```text
best_adapter/
checkpoint-900/
INITIAL_ADAPTER_ROOT itself
latest checkpoint-* under INITIAL_ADAPTER_ROOT
```

## Training

The original `order / pairwise / first / last` tasks are kept, and two
training-only auxiliary tasks are added:

```text
first_contrast
last_contrast
```

Ratios:

```python
TASK_RATIOS = {
    "order": 0.27,
    "pairwise": 0.35,
    "first": 0.12,
    "last": 0.12,
    "first_contrast": 0.07,
    "last_contrast": 0.07,
}
```

Loss weights:

```python
TASK_LOSS_WEIGHTS = {
    "order": 1.0,
    "pairwise": 0.5,
    "first": 1.5,
    "last": 1.5,
    "first_contrast": 0.75,
    "last_contrast": 0.75,
}
```

Endpoint contrastive negatives use:

```python
ENDPOINT_NEGATIVE_SAMPLING = {
    "hard_adjacent": 0.60,
    "random_non_endpoint": 0.40,
}
```

Training settings:

```python
LEARNING_RATE = 3e-6
NUM_TRAIN_EPOCHS = 1
MAX_TRAIN_STEPS = -1
SAVE_STEPS = 200
LOGGING_STEPS = 20
```

`MAX_TRAIN_STEPS = -1` disables step-based early stopping, so training runs for `NUM_TRAIN_EPOCHS` while saving every 200 steps.

Outputs are written under:

```text
/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_endpoint_contrastive_v1/runs/{RUN_ID}/endpoint_contrastive/
```

## Evaluation And Inference

Final decoding intentionally keeps the previous best structured decoder. The
contrastive task is evaluated as an auxiliary metric, but it is not added to the
submission scoring formula.

For each sample the evaluator computes:

```text
P(i before j)
P_first(i)
P_last(i)
```

Then all 24 possible chronological orders are scored:

```python
score(order) =
    alpha * pairwise_score(order)
  + beta  * log P_first(order[0])
  + gamma * log P_last(order[-1])
```

Grid search includes boundary ablations:

```python
ALPHAS = [0.5, 1.0, 1.5, 2.0]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
```

The baseline test score recorded for comparison is:

```text
0.56544
```
