"""Data loading and split helpers for order experiments."""

from __future__ import annotations

import ast
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from textual_scene.permutations import PERMUTATIONS, order_to_class, validate_order


@dataclass(frozen=True)
class SceneSample:
    sample_id: str
    gold_order: tuple[int, int, int, int]
    gold_class: int
    image_paths: tuple[str, ...] = ()
    captions: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def load_samples(config: dict[str, Any]) -> list[SceneSample]:
    path = config.get("path")
    if path:
        samples = _load_file(Path(path), config)
    elif "synthetic_samples" in config:
        samples = make_synthetic_samples(int(config.get("synthetic_samples", 32)))
    else:
        raise ValueError("data.path is required unless data.synthetic_samples is set")
    _validate_unique_ids(samples)
    limit = config.get("sample_limit")
    if limit:
        samples = samples[: int(limit)]
    if config.get("validate_images", False):
        validate_image_paths(samples, config.get("image_root"))
    return samples


def split_samples(
    samples: list[SceneSample],
    valid_ratio: float = 0.1,
    seed: int = 42,
    split_path: str | Path | None = None,
) -> tuple[list[SceneSample], list[SceneSample]]:
    if split_path and Path(split_path).exists():
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        train_ids = split["train_ids"]
        valid_ids = split["valid_ids"]
        by_id = {sample.sample_id: sample for sample in samples}
        return [by_id[item] for item in train_ids if item in by_id], [by_id[item] for item in valid_ids if item in by_id]

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, round(len(shuffled) * valid_ratio)) if len(shuffled) > 1 else 0
    valid = shuffled[:valid_size]
    train = shuffled[valid_size:]
    if split_path:
        path = Path(split_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"train_ids": [sample.sample_id for sample in train], "valid_ids": [sample.sample_id for sample in valid]},
                indent=2,
            ),
            encoding="utf-8",
        )
    return train, valid


def select_split_samples(samples: list[SceneSample], split_path: str | Path, split: str) -> list[SceneSample]:
    split_data = json.loads(Path(split_path).read_text(encoding="utf-8"))
    split_keys = {"train": "train_ids", "validation": "valid_ids", "valid": "valid_ids"}
    if split not in split_keys:
        raise ValueError(f"Unknown evaluation split: {split}")
    key = split_keys[split]
    ids = split_data[key]
    by_id = {sample.sample_id: sample for sample in samples}
    return [by_id[item] for item in ids if item in by_id]


def validate_image_paths(samples: Iterable[SceneSample], image_root: str | Path | None = None) -> None:
    root = Path(image_root) if image_root else None
    missing: list[str] = []
    for sample in samples:
        for image_path in sample.image_paths:
            candidate = (root / image_path) if root and not Path(image_path).is_absolute() else Path(image_path)
            if not candidate.exists():
                missing.append(str(candidate))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing image paths: {preview}")


def make_synthetic_samples(count: int = 32) -> list[SceneSample]:
    samples: list[SceneSample] = []
    for idx in range(count):
        order = PERMUTATIONS[idx % len(PERMUTATIONS)]
        captions = tuple(f"synthetic caption {idx:04d}-{frame}" for frame in range(1, 5))
        samples.append(
            SceneSample(
                sample_id=f"synthetic_{idx:04d}",
                gold_order=order,
                gold_class=order_to_class(order),
                captions=captions,
            )
        )
    return samples


def _load_file(path: Path, config: dict[str, Any]) -> list[SceneSample]:
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    elif path.suffix.lower() in {".json", ".jsonl"}:
        rows = _read_json_rows(path)
    else:
        raise ValueError(f"Unsupported data file: {path}")
    return [_row_to_sample(row, config, path.parent) for row in rows]


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("samples", [])


def _row_to_sample(row: dict[str, Any], config: dict[str, Any], base_dir: Path) -> SceneSample:
    id_column = config.get("id_column", "sample_id")
    order_column = config.get("order_column", "gold_order")
    image_columns = config.get("image_columns", [])
    caption_columns = config.get("caption_columns", [])
    sample_id = str(row.get(id_column) or row.get("id") or row.get("ID") or len(str(row)))
    order = parse_order(row[order_column])
    image_paths = tuple(str((base_dir / row[col]).resolve()) if row.get(col) else "" for col in image_columns)
    image_paths = tuple(path for path in image_paths if path)
    captions = tuple(str(row[col]) for col in caption_columns if row.get(col))
    if image_columns and len(image_paths) != 4:
        raise ValueError(f"Expected exactly four image paths for sample {sample_id}, got {len(image_paths)}")
    if caption_columns and len(captions) != 4:
        raise ValueError(f"Expected exactly four captions for sample {sample_id}, got {len(captions)}")
    return SceneSample(
        sample_id=sample_id,
        gold_order=order,
        gold_class=order_to_class(order),
        image_paths=image_paths,
        captions=captions,
        metadata=row,
    )


def parse_order(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("("):
            value = ast.literal_eval(stripped)
        else:
            value = stripped.replace(",", " ").split()
    return validate_order(value)


def _validate_unique_ids(samples: list[SceneSample]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample in samples:
        if sample.sample_id in seen:
            duplicates.append(sample.sample_id)
        seen.add(sample.sample_id)
    if duplicates:
        raise ValueError(f"Duplicate sample_id values found: {', '.join(duplicates[:5])}")
