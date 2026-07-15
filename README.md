# Qwen3-VL 8B Local Structured Pipeline

This branch adds a local NVIDIA GPU Python pipeline for Qwen3-VL-8B 4-task
training and structured decoding.

Project directory:

```text
qwen3vl_structured/
```

Put the dataset under the repository root:

```text
datasets/snuaichallenge_data/
  train.csv
  test.csv
  sample_submission.csv
  train/
  test/
```

Primary entrypoint:

```bash
cd qwen3vl_structured
python run_qwen3vl_structured.py --mode check --config configs/pilot.json
python run_qwen3vl_structured.py --mode pilot --config configs/pilot.json
```

The pipeline keeps the same 4-task supervision and decoder shape used by the
best Qwen2-VL run:

```text
order
pairwise
first
last
```

Final inference uses `pairwise + first + last` probabilities and exhaustive
24-permutation structured decoding. Direct order generation is trained but not
used in the final submission decoder.
