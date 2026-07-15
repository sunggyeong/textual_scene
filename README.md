# Hard Endpoint Contrastive v1

This branch keeps the same baseline adapter and learning rate used in branch 9,
then adds hard endpoint auxiliary tasks.

Use these notebooks in Colab:

```text
hard_endpoint_contrastive_train.ipynb
hard_endpoint_contrastive_eval_infer.ipynb
```

## Baseline Adapter

Training starts from the same weighted 4-task run used in branch 9:

```python
INITIAL_ADAPTER_ROOT = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_lgt_order_refine_v1/runs/"
    "20260714_003635/lgt_order_refine"
)
```

The notebook resolves the actual adapter in this order:

```text
best_adapter/
checkpoint-900/
INITIAL_ADAPTER_ROOT itself
latest checkpoint-* under INITIAL_ADAPTER_ROOT
```

The eval notebook also keeps the same baseline comparison:

```text
qwen2vl_lgt_multitask_v1/runs/20260712_234828/lgt_multitask/checkpoint-3500
```

## Training

The original `order / pairwise / first / last` tasks are kept, and two
training-only hard endpoint tasks are added:

```text
first_vs_second
third_vs_last
```

Ratios:

```python
TASK_RATIOS = {
    "order": 0.27,
    "pairwise": 0.33,
    "first": 0.13,
    "last": 0.13,
    "first_vs_second": 0.07,
    "third_vs_last": 0.07,
}
```

Loss weights:

```python
TASK_LOSS_WEIGHTS = {
    "order": 1.0,
    "pairwise": 0.5,
    "first": 1.5,
    "last": 1.5,
    "first_vs_second": 0.75,
    "third_vs_last": 0.75,
}
```

Training settings match branch 9:

```python
LEARNING_RATE = 3e-6
NUM_TRAIN_EPOCHS = 1
MAX_TRAIN_STEPS = -1
SAVE_STEPS = 200
LOGGING_STEPS = 20
```

`MAX_TRAIN_STEPS = -1` disables step-based early stopping, so training runs for
`NUM_TRAIN_EPOCHS` while saving every 200 steps.

Outputs are written under:

```text
/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_hard_endpoint_contrastive_v1/runs/{RUN_ID}/hard_endpoint_contrastive/
```

## Evaluation And Inference

Final decoding intentionally keeps the previous best structured decoder. The
hard endpoint tasks are evaluated as auxiliary metrics, but they are not added
to the submission scoring formula.

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
