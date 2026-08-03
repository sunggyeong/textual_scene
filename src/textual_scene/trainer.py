"""Training orchestration and output persistence."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from textual_scene.config import save_config
from textual_scene.data import load_samples, split_samples
from textual_scene.evaluator import evaluate_model, write_predictions
from textual_scene.models import create_model


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    experiment_name = config.get("experiment_name") or datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "outputs")) / experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    samples = load_samples(config.get("data", {}))
    train_samples, valid_samples = split_samples(
        samples,
        valid_ratio=float(config.get("data", {}).get("valid_ratio", 0.1)),
        seed=seed,
        split_path=config.get("data", {}).get("split_path"),
    )

    model = create_model(config.get("model", {}))
    if hasattr(model, "fit"):
        model.fit(train_samples)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_dir / "last")

    metrics, prediction_rows = evaluate_model(model, valid_samples or train_samples)
    _write_train_log(output_dir / "train_log.csv", train_samples, valid_samples)
    write_predictions(prediction_rows, output_dir / "predictions.csv")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "run_summary.txt").write_text(
        "\n".join(
            [
                f"experiment_name: {experiment_name}",
                f"train_samples: {len(train_samples)}",
                f"validation_samples: {len(valid_samples)}",
                f"checkpoint: {checkpoint_dir / 'last'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), "metrics": metrics}


def _write_train_log(path: Path, train_samples: list[Any], valid_samples: list[Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_samples", "validation_samples"])
        writer.writeheader()
        writer.writerow({"step": 0, "train_samples": len(train_samples), "validation_samples": len(valid_samples)})
