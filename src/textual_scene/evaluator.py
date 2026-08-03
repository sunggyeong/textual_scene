"""Evaluation orchestration and prediction persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from textual_scene.data import SceneSample, load_samples, select_split_samples
from textual_scene.metrics import compute_order_metrics, sample_metrics
from textual_scene.models import MajorityOrderModel, create_model
from textual_scene.permutations import order_to_class


def evaluate_model(model: Any, samples: list[SceneSample]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gold_orders = [sample.gold_order for sample in samples]
    predicted_orders = [tuple(model.predict_order(sample)) for sample in samples]
    metrics = compute_order_metrics(gold_orders, predicted_orders)
    rows: list[dict[str, Any]] = []
    for sample, predicted_order in zip(samples, predicted_orders):
        per_sample = sample_metrics(sample.gold_order, predicted_order)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "gold_order": " ".join(map(str, sample.gold_order)),
                "predicted_order": " ".join(map(str, predicted_order)),
                "gold_class": sample.gold_class,
                "predicted_class": order_to_class(predicted_order),
                **per_sample,
            }
        )
    return metrics, rows


def run_evaluation(config: dict[str, Any], checkpoint: str | Path | None = None) -> dict[str, Any]:
    experiment_name = config.get("experiment_name", "evaluation")
    output_dir = Path(config.get("output_dir", "outputs")) / experiment_name
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(config.get("data", {}))
    evaluation_config = config.get("evaluation", {})
    split = evaluation_config.get("split")
    if split:
        split_path = config.get("data", {}).get("split_path")
        if not split_path:
            raise ValueError("evaluation.split requires data.split_path")
        if not Path(split_path).exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        samples = select_split_samples(samples, split_path, split)
    model = _load_model(config.get("model", {}), checkpoint)
    metrics, prediction_rows = evaluate_model(model, samples)
    write_predictions(prediction_rows, eval_dir / "predictions.csv")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "eval_dir": str(eval_dir), "metrics": metrics}


def write_predictions(rows: list[dict[str, Any]], path: str | Path) -> None:
    fieldnames = [
        "sample_id",
        "gold_order",
        "predicted_order",
        "gold_class",
        "predicted_class",
        "exact_correct",
        "first_correct",
        "last_correct",
        "position_accuracy",
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_model(model_config: dict[str, Any], checkpoint: str | Path | None) -> Any:
    if checkpoint:
        if (Path(checkpoint) / "model.json").exists():
            return MajorityOrderModel.from_pretrained(checkpoint)
        raise NotImplementedError("Only smoke baseline checkpoints are restorable in this structure PR.")
    return create_model(model_config)
