"""Model factory interfaces for order-head experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from textual_scene.data import SceneSample
from textual_scene.permutations import class_to_order, order_to_class


class MajorityOrderModel:
    """Tiny deterministic baseline used for smoke tests and local plumbing checks."""

    def __init__(self, class_index: int = 0) -> None:
        self.class_index = int(class_index)

    def fit(self, samples: Iterable[SceneSample]) -> None:
        counts = Counter(sample.gold_class for sample in samples)
        if counts:
            self.class_index = counts.most_common(1)[0][0]

    def predict_class(self, sample: SceneSample) -> int:
        return self.class_index

    def predict_order(self, sample: SceneSample) -> tuple[int, int, int, int]:
        return class_to_order(self.predict_class(sample))

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.json").write_text(json.dumps({"class_index": self.class_index}, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "MajorityOrderModel":
        data = json.loads((Path(path) / "model.json").read_text(encoding="utf-8"))
        return cls(class_index=int(data["class_index"]))


def create_model(config: dict[str, Any]) -> Any:
    model_type = config.get("type", "majority_baseline")
    if model_type == "majority_baseline":
        return MajorityOrderModel()
    if model_type in {"frozen_order_head", "lora_order_head", "lora_generation"}:
        return create_transformers_model(config)
    raise ValueError(f"Unknown model type: {model_type}")


def create_transformers_model(config: dict[str, Any]) -> Any:
    """Create heavyweight experiment models behind a stable factory boundary.

    The repository structure is now ready for Qwen2-VL, LoRA, and frozen-head
    implementations without redefining them inside notebooks.
    """
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install transformers to create Qwen/LoRA experiment models") from exc

    model_name = config.get("name_or_path") or config.get("model_name_or_path")
    if not model_name:
        raise ValueError("model.name_or_path is required for transformer-backed models")
    model = AutoModelForVision2Seq.from_pretrained(model_name, **config.get("from_pretrained_kwargs", {}))

    if config.get("freeze_backbone", False):
        for parameter in model.parameters():
            parameter.requires_grad = False
    if config.get("lora"):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install peft to enable LoRA models") from exc
        model = get_peft_model(model, LoraConfig(**config["lora"]))
    return model


def class_logits_to_order(logits: Any) -> tuple[int, int, int, int]:
    class_index = int(logits.argmax().item() if hasattr(logits, "argmax") else max(range(len(logits)), key=logits.__getitem__))
    return class_to_order(class_index)


def order_to_label(order: Iterable[int]) -> int:
    return order_to_class(order)
