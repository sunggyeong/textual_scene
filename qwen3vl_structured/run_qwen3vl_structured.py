#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import gc
import itertools
import json
import math
import os
import platform
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_IMPORT_ERROR = None
try:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from tqdm.auto import tqdm

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc
    np = None
    pd = None
    torch = None
    Image = None
    Dataset = object
    tqdm = None
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None
    AutoProcessor = None
    BitsAndBytesConfig = None
    Trainer = object
    TrainingArguments = None
    set_seed = None


PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PERMUTATIONS = list(itertools.permutations([1, 2, 3, 4]))


def require_runtime_deps() -> None:
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependency. Install requirements first:\n"
            "  python -m pip install -r requirements.txt"
        ) from RUNTIME_IMPORT_ERROR


def no_grad_decorator(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_position_labels(answer: Any) -> list[int]:
    positions = answer if isinstance(answer, list) else ast.literal_eval(str(answer))
    if not isinstance(positions, list):
        raise ValueError(f"Answer is not a list: {answer}")
    positions = [int(value) for value in positions]
    if len(positions) != 4 or sorted(positions) != [1, 2, 3, 4]:
        raise ValueError(f"Invalid Answer permutation: {answer}")
    return positions


def positions_to_chronological_order(positions: list[int]) -> list[int]:
    return [
        input_index
        for input_index, _ in sorted(
            enumerate(positions, start=1),
            key=lambda item: item[1],
        )
    ]


def sequence_to_answer(order: list[int]) -> list[int]:
    answer = [0] * 4
    for rank, image_number in enumerate(order, start=1):
        answer[int(image_number) - 1] = rank
    return answer


def parse_order_prediction(text: str) -> list[int] | None:
    match = re.fullmatch(r"\s*\[\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*\]\s*", str(text))
    if not match:
        return None
    values = [int(value) for value in match.groups()]
    return values if sorted(values) == [1, 2, 3, 4] else None


def format_order(order: list[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in order) + "]"


def dtype_from_config(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported compute_dtype: {name}")


@dataclass
class Paths:
    data_root: Path
    output_dir: Path
    train_csv: Path
    test_csv: Path
    sample_submission_csv: Path
    train_image_root: Path
    test_image_root: Path
    probability_cache_dir: Path


def build_paths(config: dict[str, Any]) -> Paths:
    data_root = Path(config["data_root"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    return Paths(
        data_root=data_root,
        output_dir=output_dir,
        train_csv=data_root / "train.csv",
        test_csv=data_root / "test.csv",
        sample_submission_csv=data_root / "sample_submission.csv",
        train_image_root=data_root / "train",
        test_image_root=data_root / "test",
        probability_cache_dir=output_dir / "probability_cache",
    )


def get_image_paths(row: pd.Series, image_root: Path) -> list[Path]:
    sample_dir = image_root / str(row["Id"])
    paths = [sample_dir / str(row[f"Input_{index}"]) for index in range(1, 5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image files for Id={row['Id']}: {missing}")
    return paths


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def read_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)
    sample_df = pd.read_csv(paths.sample_submission_csv) if paths.sample_submission_csv.exists() else None
    train_df["Id"] = train_df["Id"].astype(str)
    test_df["Id"] = test_df["Id"].astype(str)
    if sample_df is not None:
        sample_df["Id"] = sample_df["Id"].astype(str)
    return train_df, test_df, sample_df


def add_answer_columns(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    orders = []
    positions_list = []
    for _, row in train_df.iterrows():
        positions = parse_position_labels(row["Answer"])
        order = positions_to_chronological_order(positions)
        if "No_ordering" in row and parse_bool(row["No_ordering"]) and order != [1, 2, 3, 4]:
            raise ValueError(f"No_ordering conflict for Id={row['Id']}: {row['Answer']}")
        positions_list.append(positions)
        orders.append(order)
    train_df["Answer_positions"] = positions_list
    train_df["Chronological_order"] = orders
    return train_df


def split_train_valid(train_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_ids = train_df["Id"].unique().copy()
    rng = np.random.default_rng(int(config.get("seed", 42)))
    rng.shuffle(unique_ids)
    valid_size = max(1, int(len(unique_ids) * float(config.get("valid_ratio", 0.1))))
    valid_ids = set(unique_ids[:valid_size])
    training_ids = set(unique_ids[valid_size:])
    training_df = train_df[train_df["Id"].isin(training_ids)].reset_index(drop=True)
    validation_df = train_df[train_df["Id"].isin(valid_ids)].reset_index(drop=True)
    limit = config.get("train_original_rows")
    if limit is not None:
        training_df = training_df.sample(n=min(int(limit), len(training_df)), random_state=int(config.get("seed", 42))).reset_index(drop=True)
    return training_df, validation_df


def pairwise_target(positions: list[int], first_index: int, second_index: int) -> str:
    return "1" if int(positions[first_index]) < int(positions[second_index]) else "2"


def base_record(row_index: int, row: pd.Series, image_root: Path) -> dict[str, Any]:
    positions = list(row["Answer_positions"])
    order = list(row["Chronological_order"])
    return {
        "row_index": int(row_index),
        "sample_id": str(row["Id"]),
        "sentence": "" if pd.isna(row["Sentence"]) else str(row["Sentence"]),
        "positions": positions,
        "order": order,
        "image_paths": [str(path) for path in get_image_paths(row, image_root)],
    }


def build_record_pools(dataframe: pd.DataFrame, image_root: Path, task_ratios: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    pools = {task: [] for task in task_ratios}
    for row_index, row in dataframe.iterrows():
        base = base_record(row_index, row, image_root)
        if "order" in pools:
            item = dict(base)
            item.update({"task_type": "order", "target": format_order(base["order"])})
            pools["order"].append(item)
        if "first" in pools:
            item = dict(base)
            item.update({"task_type": "first", "target": str(base["order"][0])})
            pools["first"].append(item)
        if "last" in pools:
            item = dict(base)
            item.update({"task_type": "last", "target": str(base["order"][-1])})
            pools["last"].append(item)
        if "pairwise" in pools:
            for first_index, second_index in PAIR_INDICES:
                item = dict(base)
                item.update({
                    "task_type": "pairwise",
                    "first_index": first_index,
                    "second_index": second_index,
                    "image_paths": [base["image_paths"][first_index], base["image_paths"][second_index]],
                    "target": pairwise_target(base["positions"], first_index, second_index),
                })
                pools["pairwise"].append(item)
    return pools


def sample_records(records: list[dict[str, Any]], count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    indices = rng.integers(0, len(records), size=count)
    return [records[int(index)] for index in indices]


def build_balanced_records(dataframe: pd.DataFrame, image_root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    ratios = {key: float(value) for key, value in config["task_ratios"].items()}
    pools = build_record_pools(dataframe, image_root, ratios)
    anchor_task = "pairwise" if "pairwise" in pools else next(iter(pools))
    base_total = int(math.ceil(len(pools[anchor_task]) / ratios[anchor_task]))
    rng = np.random.default_rng(int(config.get("seed", 42)))
    merged = []
    for task, ratio in ratios.items():
        count = max(1, int(round(base_total * ratio)))
        records = sample_records(pools[task], count, rng)
        merged.extend(records)
        print(task, "pool:", len(pools[task]), "sampled:", len(records))
    rng.shuffle(merged)
    return merged, pools


def task_instruction(example: dict[str, Any]) -> str:
    sentence = example["sentence"]
    task_type = example["task_type"]
    if task_type == "pairwise":
        return (
            f"Caption:\n{sentence}\n\n"
            "Question: Which image occurs first?\n"
            "If the first image occurs earlier, answer 1.\n"
            "If the second image occurs earlier, answer 2.\n"
            "Answer only 1 or 2."
        )
    if task_type == "first":
        return (
            f"Caption:\n{sentence}\n\n"
            "Question: Which image represents the beginning of the story?\n"
            "Answer only the image number from 1 to 4."
        )
    if task_type == "last":
        return (
            f"Caption:\n{sentence}\n\n"
            "Question: Which image represents the end of the story?\n"
            "Answer only the image number from 1 to 4."
        )
    if task_type == "order":
        return (
            f"Caption:\n{sentence}\n\n"
            "Question: Compare the temporal relation between scenes and identify the likely first and last scenes.\n"
            "Using these cues, determine the complete chronological order.\n"
            "Output only the final ordered list, such as [1, 2, 3, 4]."
        )
    raise ValueError(task_type)


def make_messages(example: dict[str, Any], include_answer: bool = False) -> list[dict[str, Any]]:
    content = []
    for idx, _ in enumerate(example["image_paths"], start=1):
        content.append({"type": "text", "text": f"\nImage {idx}:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": "\n\n" + task_instruction(example)})
    messages = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": str(example["target"])})
    return messages


class TaskDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class MultiTaskCollator:
    def __init__(self, processor: Any):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.assistant_prefix_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def _find_subsequence(self, ids: list[int], pattern: list[int]) -> int | None:
        if not pattern:
            return None
        for i in range(0, max(0, len(ids) - len(pattern) + 1)):
            if ids[i:i + len(pattern)] == pattern:
                return i
        return None

    def _mask_prompt(self, input_ids: torch.Tensor, target: str) -> torch.Tensor:
        ids = input_ids.tolist()
        labels = input_ids.clone()
        start = None
        prefix = self.assistant_prefix_ids
        for i in range(0, max(0, len(ids) - len(prefix) + 1)):
            if ids[i:i + len(prefix)] == prefix:
                start = i + len(prefix)
        if start is None:
            target_ids = self.tokenizer.encode(str(target), add_special_tokens=False)
            target_start = self._find_subsequence(ids, target_ids)
            if target_start is not None:
                start = target_start
        if start is None:
            labels[:] = -100
        else:
            labels[:start] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        task_types = []
        targets = []
        for example in batch:
            text = self.processor.apply_chat_template(make_messages(example, include_answer=True), tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images.append([load_rgb(Path(path)) for path in example["image_paths"]])
            task_types.append(example["task_type"])
            targets.append(str(example["target"]))
        encoded = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.stack([self._mask_prompt(row, target) for row, target in zip(encoded["input_ids"], targets)])
        valid_target_counts = labels.ne(-100).sum(dim=1)
        if (valid_target_counts == 0).any():
            raise ValueError(f"Assistant target tokens were not found: {valid_target_counts.tolist()}")
        encoded["labels"] = labels
        encoded["task_type"] = task_types
        return encoded


class TaskLossTrainer(Trainer):
    def __init__(self, *args: Any, task_loss_weights: dict[str, float] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.task_loss_weights = task_loss_weights or {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_types = inputs.pop("task_type")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        valid_mask = shift_labels.ne(-100)
        sample_losses = (token_losses * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
        weights = torch.tensor(
            [self.task_loss_weights.get(task, 1.0) for task in task_types],
            dtype=sample_losses.dtype,
            device=sample_losses.device,
        )
        loss = (sample_losses * weights).mean()
        logs = {}
        for task in sorted(set(task_types)):
            task_mask = torch.tensor([value == task for value in task_types], device=sample_losses.device)
            if task_mask.any():
                logs[f"train_{task}_loss"] = sample_losses[task_mask].mean().detach().float().item()
        if logs:
            self.log(logs)
        return (loss, outputs) if return_outputs else loss


def load_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except Exception:
        pass
    try:
        return AutoModelForImageTextToText
    except Exception:
        return AutoModelForVision2Seq


def load_processor(config: dict[str, Any]) -> Any:
    return AutoProcessor.from_pretrained(
        config["model_id"],
        min_pixels=int(config.get("min_pixels", 200704)),
        max_pixels=int(config.get("max_pixels", 602112)),
        trust_remote_code=True,
    )


def load_base_model(config: dict[str, Any], for_training: bool) -> Any:
    compute_dtype = dtype_from_config(config.get("compute_dtype", "bfloat16"))
    quant_config = None
    if bool(config.get("load_in_4bit", True)):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model_cls = load_model_class()
    model = model_cls.from_pretrained(
        config["model_id"],
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if for_training:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )
    return model


def discover_linear_modules(model: Any, output_path: Path) -> list[str]:
    module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            module_names.append(leaf)
    unique = sorted(set(module_names))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(unique) + "\n", encoding="utf-8")
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    selected = [name for name in preferred if name in unique]
    return selected or unique


def build_train_model(config: dict[str, Any], paths: Paths) -> Any:
    model = load_base_model(config, for_training=True)
    target_modules = config.get("lora_target_modules")
    if not target_modules:
        target_modules = discover_linear_modules(model, paths.output_dir / "lora_modules.txt")
    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    print("LoRA target modules:", target_modules)
    model.print_trainable_parameters()
    return model


def model_device(active_model: Any) -> torch.device:
    return next(active_model.parameters()).device


def load_eval_model(config: dict[str, Any], checkpoint: Path) -> Any:
    base = load_base_model(config, for_training=False)
    if (checkpoint / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=False)
    else:
        model = base
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.num_beams = 1
    return model


def make_eval_example(row: pd.Series, image_root: Path, task_type: str, pair: tuple[int, int] | None = None) -> dict[str, Any]:
    image_paths = [str(path) for path in get_image_paths(row, image_root)]
    if "Answer_positions" in row:
        positions = list(row["Answer_positions"])
        order = list(row["Chronological_order"])
    else:
        positions = [1, 2, 3, 4]
        order = [1, 2, 3, 4]
    example = {
        "sample_id": str(row["Id"]),
        "sentence": "" if pd.isna(row["Sentence"]) else str(row["Sentence"]),
        "positions": positions,
        "order": order,
        "image_paths": image_paths,
        "task_type": task_type,
        "target": "1",
    }
    if task_type == "pairwise":
        assert pair is not None
        a, b = pair
        example["first_index"] = a - 1
        example["second_index"] = b - 1
        example["image_paths"] = [image_paths[a - 1], image_paths[b - 1]]
    return example


@no_grad_decorator
def score_candidate_strings(model: Any, processor: Any, example: dict[str, Any], candidates: list[str]) -> dict[str, float]:
    old_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"
    prompt_text = processor.apply_chat_template(make_messages(example, include_answer=False), tokenize=False, add_generation_prompt=True)
    images = [load_rgb(Path(path)) for path in example["image_paths"]]
    scores = []
    for candidate in candidates:
        full_text = prompt_text + str(candidate)
        encoded = processor(text=[full_text], images=[images], return_tensors="pt")
        encoded = {key: value.to(model_device(model)) if torch.is_tensor(value) else value for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        prompt_encoded = processor(text=[prompt_text], images=[images], return_tensors="pt")
        prompt_len = int(prompt_encoded["input_ids"].shape[1])
        labels[:, :prompt_len] = -100
        outputs = model(**encoded)
        logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        token_mask = shift_labels.ne(-100)
        logprob = -(loss * token_mask).sum().float().item()
        scores.append(logprob)
    probs = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).numpy()
    processor.tokenizer.padding_side = old_padding_side
    return {candidate: float(prob) for candidate, prob in zip(candidates, probs)}


def extract_probability_cache(config: dict[str, Any], paths: Paths, processor: Any, model: Any, rows: pd.DataFrame, image_root: Path, checkpoint_name: str, tag: str) -> list[dict[str, Any]]:
    cache_path = paths.probability_cache_dir / f"{checkpoint_name}_{tag}_{config.get('prompt_version', 'prompt')}_probability_cache.json"
    if cache_path.exists():
        print("[SKIP]", cache_path)
        return load_json(cache_path)
    records = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"{checkpoint_name} {tag} probs"):
        if "Answer_positions" in row:
            positions = list(row["Answer_positions"])
            gold_order = list(row["Chronological_order"])
        else:
            positions = [1, 2, 3, 4]
            gold_order = [1, 2, 3, 4]
        first_probs = {int(k): v for k, v in score_candidate_strings(model, processor, make_eval_example(row, image_root, "first"), ["1", "2", "3", "4"]).items()}
        last_probs = {int(k): v for k, v in score_candidate_strings(model, processor, make_eval_example(row, image_root, "last"), ["1", "2", "3", "4"]).items()}
        pair_probs = {}
        pair_correct = []
        for first_index, second_index in PAIR_INDICES:
            a, b = first_index + 1, second_index + 1
            probs = score_candidate_strings(model, processor, make_eval_example(row, image_root, "pairwise", pair=(a, b)), ["1", "2"])
            p_a_before_b = float(probs["1"])
            pair_probs[f"{a}>{b}"] = p_a_before_b
            pair_probs[f"{b}>{a}"] = float(1.0 - p_a_before_b)
            pred_first = a if p_a_before_b >= 0.5 else b
            gold_first = a if positions[first_index] < positions[second_index] else b
            pair_correct.append(int(pred_first == gold_first))
        records.append({
            "sample_id": str(row["Id"]),
            "gold_order": gold_order,
            "first_probs": {str(k): v for k, v in first_probs.items()},
            "last_probs": {str(k): v for k, v in last_probs.items()},
            "pair_probs": pair_probs,
            "pairwise_accuracy": float(np.mean(pair_correct)),
            "task_first_accuracy": float(max(first_probs, key=first_probs.get) == gold_order[0]),
            "task_last_accuracy": float(max(last_probs, key=last_probs.get) == gold_order[-1]),
        })
    save_json(records, cache_path)
    return records


def order_ranks(order: list[int]) -> dict[int, int]:
    return {int(image_number): position for position, image_number in enumerate(order)}


def pair_accuracy_from_orders(pred_order: list[int], gold_order: list[int]) -> float:
    pred_ranks = order_ranks(pred_order)
    gold_ranks = order_ranks(gold_order)
    return float(np.mean([
        (pred_ranks[a] < pred_ranks[b]) == (gold_ranks[a] < gold_ranks[b])
        for a, b in itertools.combinations([1, 2, 3, 4], 2)
    ]))


def order_metric_row(pred_order: list[int], gold_order: list[int]) -> dict[str, float]:
    return {
        "exact_match": float(pred_order == gold_order),
        "pair_accuracy": pair_accuracy_from_orders(pred_order, gold_order),
        "position_accuracy": float(np.mean([p == g for p, g in zip(pred_order, gold_order)])),
        "valid_output": 1.0,
    }


def structured_score(sample: dict[str, Any], order: tuple[int, int, int, int], alpha: float, beta: float, gamma: float) -> float:
    eps = 1e-12
    pair_score = np.mean([
        math.log(float(sample["pair_probs"][f"{order[i]}>{order[j]}"]) + eps)
        for i in range(4)
        for j in range(i + 1, 4)
    ])
    first_score = math.log(float(sample["first_probs"][str(order[0])]) + eps)
    last_score = math.log(float(sample["last_probs"][str(order[-1])]) + eps)
    return alpha * pair_score + beta * first_score + gamma * last_score


def decode_structured(sample: dict[str, Any], alpha: float, beta: float, gamma: float) -> list[int]:
    return list(max(PERMUTATIONS, key=lambda order: structured_score(sample, order, alpha, beta, gamma)))


def evaluate_decoding(samples: list[dict[str, Any]], alpha: float, beta: float, gamma: float) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for sample in samples:
        gold = [int(value) for value in sample["gold_order"]]
        pred = decode_structured(sample, alpha, beta, gamma)
        metric = order_metric_row(pred, gold)
        metric.update({
            "sample_id": sample["sample_id"],
            "pred_order": pred,
            "gold_order": gold,
            "pairwise_accuracy": sample["pairwise_accuracy"],
            "task_first_accuracy": sample["task_first_accuracy"],
            "task_last_accuracy": sample["task_last_accuracy"],
            "task_first_last_both_correct": float(sample["task_first_accuracy"] == 1.0 and sample["task_last_accuracy"] == 1.0),
            "decoded_first_accuracy": float(pred[0] == gold[0]),
            "decoded_last_accuracy": float(pred[-1] == gold[-1]),
            "decoded_first_last_both_correct": float(pred[0] == gold[0] and pred[-1] == gold[-1]),
        })
        rows.append(metric)
    df = pd.DataFrame(rows)
    summary = {
        "exact_match": df["exact_match"].mean(),
        "pair_accuracy": df["pair_accuracy"].mean(),
        "position_accuracy": df["position_accuracy"].mean(),
        "valid_output_rate": df["valid_output"].mean(),
        "mean_pairwise_accuracy": df["pairwise_accuracy"].mean(),
        "task_first_accuracy": df["task_first_accuracy"].mean(),
        "task_last_accuracy": df["task_last_accuracy"].mean(),
        "task_first_last_both_correct_rate": df["task_first_last_both_correct"].mean(),
        "decoded_first_accuracy": df["decoded_first_accuracy"].mean(),
        "decoded_last_accuracy": df["decoded_last_accuracy"].mean(),
        "decoded_first_last_both_correct_rate": df["decoded_first_last_both_correct"].mean(),
    }
    return summary, df


def grid_search(config: dict[str, Any], paths: Paths, samples: list[dict[str, Any]], checkpoint_name: str, tag: str) -> pd.DataFrame:
    grid_path = paths.output_dir / f"{checkpoint_name}_{tag}_decoding_grid.csv"
    if grid_path.exists():
        return pd.read_csv(grid_path)
    rows = []
    for alpha, beta, gamma in itertools.product(config["alpha_grid"], config["beta_grid"], config["gamma_grid"]):
        summary, _ = evaluate_decoding(samples, alpha=float(alpha), beta=float(beta), gamma=float(gamma))
        summary.update({"checkpoint": checkpoint_name, "tag": tag, "alpha": alpha, "beta": beta, "gamma": gamma, "decoding": "pair_first_last"})
        rows.append(summary)
    df = pd.DataFrame(rows).sort_values(["exact_match", "pair_accuracy", "position_accuracy"], ascending=False).reset_index(drop=True)
    df.to_csv(grid_path, index=False)
    return df


def validate_dataset(config: dict[str, Any], paths: Paths, fast_check: bool = False) -> dict[str, Any]:
    train_df, test_df, sample_df = read_data(paths)
    train_df = add_answer_columns(train_df)
    row_iter = train_df.head(20).iterrows() if fast_check else train_df.iterrows()
    missing_train = 0
    for _, row in tqdm(list(row_iter), desc="check train images"):
        try:
            get_image_paths(row, paths.train_image_root)
        except FileNotFoundError:
            missing_train += 1
    test_iter = test_df.head(20).iterrows() if fast_check else test_df.iterrows()
    missing_test = 0
    for _, row in tqdm(list(test_iter), desc="check test images"):
        try:
            get_image_paths(row, paths.test_image_root)
        except FileNotFoundError:
            missing_test += 1
    report = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "sample_submission_rows": None if sample_df is None else len(sample_df),
        "expected_train_images": len(train_df) * 4,
        "expected_test_images": len(test_df) * 4,
        "checked_fast": bool(fast_check),
        "missing_train_rows": missing_train,
        "missing_test_rows": missing_test,
        "no_ordering_count": int(train_df["No_ordering"].apply(parse_bool).sum()) if "No_ordering" in train_df else None,
        "examples": train_df[["Id", "Answer", "Chronological_order"]].head(5).to_dict("records"),
    }
    save_json(report, paths.output_dir / "dataset_check.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def save_environment(config: dict[str, Any], paths: Paths) -> None:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
        "model_id": config["model_id"],
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    save_json(env, paths.output_dir / "environment.json")
    print(json.dumps(env, ensure_ascii=False, indent=2))


def run_check(config: dict[str, Any], paths: Paths, fast_check: bool = False) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(config, paths)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    validate_dataset(config, paths, fast_check=fast_check)
    processor = load_processor(config)
    token_info = {}
    for value in ["1", "2", "3", "4", " 1", "\n1"]:
        token_info[repr(value)] = processor.tokenizer.encode(value, add_special_tokens=False)
    save_json(token_info, paths.output_dir / "digit_tokenization.json")
    print("digit tokenization:", token_info)
    print("Processor load OK.")


def run_train(config: dict[str, Any], paths: Paths) -> Path:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.probability_cache_dir.mkdir(parents=True, exist_ok=True)
    save_environment(config, paths)
    save_json(config, paths.output_dir / "train_config.json")
    set_seed(int(config.get("seed", 42)))
    random.seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    train_df, _, _ = read_data(paths)
    train_df = add_answer_columns(train_df)
    training_df, validation_df = split_train_valid(train_df, config)
    train_records, train_pools = build_balanced_records(training_df, paths.train_image_root, config)
    save_json(
        {
            task: {"pool": len(train_pools[task]), "sampled": sum(record["task_type"] == task for record in train_records)}
            for task in config["task_ratios"]
        },
        paths.output_dir / "task_distribution.json",
    )
    processor = load_processor(config)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    model = build_train_model(config, paths)
    training_args = TrainingArguments(
        output_dir=str(paths.output_dir),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        max_steps=int(config.get("max_train_steps", -1)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        max_grad_norm=float(config.get("max_grad_norm", 0.3)),
        bf16=dtype_from_config(config.get("compute_dtype", "bfloat16")) == torch.bfloat16,
        fp16=dtype_from_config(config.get("compute_dtype", "bfloat16")) == torch.float16,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config.get("optim", "paged_adamw_8bit"),
        logging_steps=int(config.get("logging_steps", 20)),
        save_strategy="steps",
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=config.get("save_total_limit"),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
    )
    trainer = TaskLossTrainer(
        model=model,
        args=training_args,
        train_dataset=TaskDataset(train_records),
        data_collator=MultiTaskCollator(processor),
        task_loss_weights={key: float(value) for key, value in config["task_loss_weights"].items()},
    )
    trainer.train()
    final_dir = paths.output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print("saved:", final_dir)
    return final_dir


def checkpoint_label(checkpoint: Path) -> str:
    return checkpoint.name if checkpoint.name else "model"


def run_eval(config: dict[str, Any], paths: Paths, checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        checkpoint = paths.output_dir / "final_adapter"
    checkpoint = checkpoint.expanduser().resolve()
    processor = load_processor(config)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = load_eval_model(config, checkpoint)
    train_df, _, _ = read_data(paths)
    train_df = add_answer_columns(train_df)
    _, validation_df = split_train_valid(train_df, config)
    shuffled = validation_df.sample(frac=1.0, random_state=int(config.get("seed", 42))).reset_index(drop=True)
    quick_n = min(int(config.get("quick_eval_rows", 50)), len(shuffled))
    quick_rows = shuffled.iloc[:quick_n].reset_index(drop=True)
    tuning_rows = shuffled.iloc[quick_n:quick_n + min(150, max(0, len(shuffled) - quick_n))].reset_index(drop=True)
    holdout_rows = shuffled.iloc[quick_n + len(tuning_rows):quick_n + len(tuning_rows) + min(150, max(0, len(shuffled) - quick_n - len(tuning_rows)))].reset_index(drop=True)
    if len(tuning_rows) == 0:
        tuning_rows = quick_rows
    if len(holdout_rows) == 0:
        holdout_rows = quick_rows
    ckpt = checkpoint_label(checkpoint)
    quick_samples = extract_probability_cache(config, paths, processor, model, quick_rows, paths.train_image_root, ckpt, f"quick{len(quick_rows)}")
    quick_search = grid_search(config, paths, quick_samples, ckpt, f"quick{len(quick_rows)}")
    quick_search.to_csv(paths.output_dir / "quick_metrics.csv", index=False)
    tuning_samples = extract_probability_cache(config, paths, processor, model, tuning_rows, paths.train_image_root, ckpt, f"tuning{len(tuning_rows)}")
    holdout_samples = extract_probability_cache(config, paths, processor, model, holdout_rows, paths.train_image_root, ckpt, f"holdout{len(holdout_rows)}")
    tuning_search = grid_search(config, paths, tuning_samples, ckpt, f"tuning{len(tuning_rows)}")
    best_weights = tuning_search.iloc[0].to_dict()
    holdout_summary, predictions = evaluate_decoding(
        holdout_samples,
        alpha=float(best_weights["alpha"]),
        beta=float(best_weights["beta"]),
        gamma=float(best_weights["gamma"]),
    )
    holdout_summary.update({
        "checkpoint": ckpt,
        "checkpoint_dir": str(checkpoint),
        "alpha": float(best_weights["alpha"]),
        "beta": float(best_weights["beta"]),
        "gamma": float(best_weights["gamma"]),
        "tuning_exact_match": float(best_weights["exact_match"]),
    })
    pd.DataFrame([holdout_summary]).to_csv(paths.output_dir / "holdout_metrics.csv", index=False)
    predictions.to_csv(paths.output_dir / "predictions.csv", index=False)
    best_config = {
        "model_id": config["model_id"],
        "checkpoint": ckpt,
        "checkpoint_dir": str(checkpoint),
        "task_ratios": config["task_ratios"],
        "task_loss_weights": config["task_loss_weights"],
        "decoding": {
            "type": "pair_first_last_permutation_search",
            "alpha": float(best_weights["alpha"]),
            "beta": float(best_weights["beta"]),
            "gamma": float(best_weights["gamma"]),
        },
        "holdout": holdout_summary,
        "baseline_test_score": 0.56544,
    }
    save_json(best_config, paths.output_dir / "best_config.json")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_config


def validate_submission(submission_df: pd.DataFrame, sample_submission_df: pd.DataFrame | None) -> None:
    if sample_submission_df is not None:
        if len(submission_df) != len(sample_submission_df):
            raise ValueError("Submission row count mismatch.")
        if submission_df["Id"].astype(str).tolist() != sample_submission_df["Id"].astype(str).tolist():
            raise ValueError("Submission Id order mismatch.")
    if submission_df["Id"].duplicated().any():
        raise ValueError("Duplicate Id in submission.")
    for answer in submission_df["Answer"]:
        values = ast.literal_eval(str(answer))
        if sorted([int(value) for value in values]) != [1, 2, 3, 4]:
            raise ValueError(f"Invalid submission answer: {answer}")


def run_infer(config: dict[str, Any], paths: Paths, checkpoint: Path | None) -> Path:
    if checkpoint is None:
        best_path = paths.output_dir / "best_config.json"
        if best_path.exists():
            checkpoint = Path(load_json(best_path)["checkpoint_dir"])
        else:
            checkpoint = paths.output_dir / "final_adapter"
    checkpoint = checkpoint.expanduser().resolve()
    best_config_path = paths.output_dir / "best_config.json"
    if best_config_path.exists():
        decode_config = load_json(best_config_path)["decoding"]
    else:
        decode_config = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    processor = load_processor(config)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = load_eval_model(config, checkpoint)
    _, test_df, sample_df = read_data(paths)
    submission_rows = []
    test_cache = []
    cache_path = paths.probability_cache_dir / f"{checkpoint_label(checkpoint)}_test_probability_cache.json"
    existing_by_id = {}
    if cache_path.exists():
        existing = load_json(cache_path)
        existing_by_id = {record["sample_id"]: record for record in existing}
        test_cache = existing
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="test inference"):
        sample_id = str(row["Id"])
        if sample_id in existing_by_id:
            sample = existing_by_id[sample_id]
            pred_order = sample["pred_order"]
        else:
            first_probs = {int(k): v for k, v in score_candidate_strings(model, processor, make_eval_example(row, paths.test_image_root, "first"), ["1", "2", "3", "4"]).items()}
            last_probs = {int(k): v for k, v in score_candidate_strings(model, processor, make_eval_example(row, paths.test_image_root, "last"), ["1", "2", "3", "4"]).items()}
            pair_probs = {}
            for first_index, second_index in PAIR_INDICES:
                a, b = first_index + 1, second_index + 1
                probs = score_candidate_strings(model, processor, make_eval_example(row, paths.test_image_root, "pairwise", pair=(a, b)), ["1", "2"])
                pair_probs[f"{a}>{b}"] = float(probs["1"])
                pair_probs[f"{b}>{a}"] = float(1.0 - probs["1"])
            sample = {
                "sample_id": sample_id,
                "first_probs": {str(k): v for k, v in first_probs.items()},
                "last_probs": {str(k): v for k, v in last_probs.items()},
                "pair_probs": pair_probs,
            }
            pred_order = decode_structured(
                sample,
                alpha=float(decode_config["alpha"]),
                beta=float(decode_config["beta"]),
                gamma=float(decode_config["gamma"]),
            )
            sample["pred_order"] = pred_order
            test_cache.append(sample)
            if len(test_cache) % 100 == 0:
                save_json(test_cache, cache_path)
        submission_rows.append({"Id": sample_id, "Answer": str(sequence_to_answer(pred_order))})
    save_json(test_cache, cache_path)
    submission = pd.DataFrame(submission_rows)
    validate_submission(submission, sample_df)
    submission_path = paths.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print("submission saved:", submission_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return submission_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "pilot", "train", "eval", "infer", "all"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fast-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_deps()
    config = load_json(Path(args.config))
    paths = build_paths(config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.mode == "check":
        run_check(config, paths, fast_check=args.fast_check)
    elif args.mode == "train":
        run_train(config, paths)
    elif args.mode == "eval":
        run_eval(config, paths, checkpoint)
    elif args.mode == "infer":
        run_infer(config, paths, checkpoint)
    elif args.mode == "pilot":
        ckpt = run_train(config, paths)
        run_eval(config, paths, ckpt)
    elif args.mode == "all":
        ckpt = run_train(config, paths)
        best = run_eval(config, paths, ckpt)
        run_infer(config, paths, Path(best["checkpoint_dir"]))
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
