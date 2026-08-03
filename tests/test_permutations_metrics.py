import json

import pytest

from textual_scene.data import load_samples, parse_order, resolve_image_path, select_split_samples, split_samples
from textual_scene.evaluator import run_evaluation
from textual_scene.metrics import compute_order_metrics
from textual_scene.permutations import CLASS_TO_ORDER, ORDER_TO_CLASS, PERMUTATIONS, class_to_order, order_to_class
from textual_scene.trainer import run_training


def test_permutation_mapping_roundtrip():
    assert len(PERMUTATIONS) == 24
    assert len(ORDER_TO_CLASS) == 24
    assert len(CLASS_TO_ORDER) == 24
    for order in PERMUTATIONS:
        assert class_to_order(order_to_class(order)) == order


def test_order_metrics():
    gold = [(1, 2, 3, 4), (4, 3, 2, 1)]
    pred = [(1, 2, 3, 4), (4, 2, 3, 1)]
    metrics = compute_order_metrics(gold, pred)
    assert metrics["exact_order_accuracy"] == 0.5
    assert metrics["first_accuracy"] == 1.0
    assert metrics["last_accuracy"] == 1.0
    assert metrics["position_accuracy"] == 0.75
    assert 0.0 <= metrics["pairwise_relation_accuracy"] <= 1.0


def test_invalid_permutation_is_rejected():
    with pytest.raises(ValueError):
        order_to_class((1, 1, 2, 3))


def test_csv_and_json_order_parsing(tmp_path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("sample_id,gold_order\ncsv_1,\"1 2 3 4\"\n", encoding="utf-8")
    json_path = tmp_path / "samples.json"
    json_path.write_text(json.dumps([{"sample_id": "json_1", "gold_order": [4, 3, 2, 1]}]), encoding="utf-8")

    assert parse_order("1,2,3,4") == (1, 2, 3, 4)
    assert load_samples({"path": str(csv_path)})[0].gold_order == (1, 2, 3, 4)
    assert load_samples({"path": str(json_path)})[0].gold_order == (4, 3, 2, 1)


def test_split_reload_preserves_order(tmp_path):
    samples = load_samples({"synthetic_samples": 12})
    split_path = tmp_path / "split.json"
    train_a, valid_a = split_samples(samples, valid_ratio=0.25, seed=7, split_path=split_path)
    train_b, valid_b = split_samples(samples, valid_ratio=0.25, seed=999, split_path=split_path)

    assert [sample.sample_id for sample in train_a] == [sample.sample_id for sample in train_b]
    assert [sample.sample_id for sample in valid_a] == [sample.sample_id for sample in valid_b]


def test_split_reload_rejects_missing_ids(tmp_path):
    samples = load_samples({"synthetic_samples": 4})
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train_ids": ["missing"], "valid_ids": ["synthetic_0000"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Split contains IDs missing"):
        split_samples(samples, split_path=split_path)
    with pytest.raises(ValueError, match="Split contains IDs missing"):
        select_split_samples(samples, split_path, "train")


def test_duplicate_sample_ids_are_rejected(tmp_path):
    path = tmp_path / "dupes.json"
    path.write_text(
        json.dumps(
            [
                {"sample_id": "same", "gold_order": [1, 2, 3, 4]},
                {"sample_id": "same", "gold_order": [4, 3, 2, 1]},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_samples({"path": str(path)})


def test_empty_data_path_without_synthetic_samples_is_rejected():
    with pytest.raises(ValueError, match="data.path is required"):
        load_samples({})


def test_image_root_resolves_image_columns(tmp_path):
    metadata_dir = tmp_path / "metadata"
    image_dir = tmp_path / "images"
    metadata_dir.mkdir()
    image_dir.mkdir()
    expected = image_dir / "abc.jpg"
    expected.write_bytes(b"fake")
    csv_path = metadata_dir / "train.csv"
    csv_path.write_text(
        "sample_id,gold_order,image_1,image_2,image_3,image_4,caption_1,caption_2,caption_3,caption_4\n"
        "a,\"1 2 3 4\",abc.jpg,abc.jpg,abc.jpg,abc.jpg,c1,c2,c3,c4\n",
        encoding="utf-8",
    )

    samples = load_samples(
        {
            "path": str(csv_path),
            "image_root": str(image_dir),
            "image_columns": ["image_1", "image_2", "image_3", "image_4"],
            "caption_columns": ["caption_1", "caption_2", "caption_3", "caption_4"],
            "validate_images": True,
        }
    )

    assert samples[0].image_paths == tuple(str(expected.resolve()) for _ in range(4))
    assert samples[0].captions == ("c1", "c2", "c3", "c4")
    assert resolve_image_path(expected, image_dir) == str(expected)


def test_train_eval_outputs_do_not_overwrite(tmp_path):
    config = {
        "experiment_name": "smoke_test",
        "seed": 42,
        "output_dir": str(tmp_path / "outputs"),
        "model": {"type": "majority_baseline"},
        "data": {
            "synthetic_samples": 32,
            "valid_ratio": 0.25,
            "split_path": str(tmp_path / "outputs" / "splits" / "smoke_test_seed42.json"),
        },
        "training": {"epochs": 1},
        "evaluation": {"split": "validation"},
    }

    train_result = run_training(config)
    eval_result = run_evaluation(config, checkpoint=tmp_path / "outputs" / "smoke_test" / "checkpoints" / "last")

    train_metrics = tmp_path / "outputs" / "smoke_test" / "train" / "metrics.json"
    eval_metrics = tmp_path / "outputs" / "smoke_test" / "eval" / "metrics.json"
    assert train_metrics.exists()
    assert eval_metrics.exists()
    assert train_metrics.read_text(encoding="utf-8") == eval_metrics.read_text(encoding="utf-8")
    assert train_result["metrics"] == eval_result["metrics"]
