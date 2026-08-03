"""Shared evaluation metrics for four-frame order predictions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from textual_scene.permutations import validate_order


def exact_order_accuracy(gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]) -> float:
    pairs = _validated_pairs(gold_orders, predicted_orders)
    return _mean(gold == pred for gold, pred in pairs)


def position_accuracy(gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]) -> float:
    pairs = _validated_pairs(gold_orders, predicted_orders)
    return _mean(
        sum(int(g == p) for g, p in zip(gold, pred)) / 4.0
        for gold, pred in pairs
    )


def first_accuracy(gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]) -> float:
    pairs = _validated_pairs(gold_orders, predicted_orders)
    return _mean(gold[0] == pred[0] for gold, pred in pairs)


def last_accuracy(gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]) -> float:
    pairs = _validated_pairs(gold_orders, predicted_orders)
    return _mean(gold[-1] == pred[-1] for gold, pred in pairs)


def pairwise_relation_accuracy(gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]) -> float:
    pairs = _validated_pairs(gold_orders, predicted_orders)
    scores: list[float] = []
    for gold, pred in pairs:
        gold_rank = {frame: idx for idx, frame in enumerate(gold)}
        pred_rank = {frame: idx for idx, frame in enumerate(pred)}
        correct = 0
        total = 0
        for left in range(1, 5):
            for right in range(left + 1, 5):
                correct += int((gold_rank[left] < gold_rank[right]) == (pred_rank[left] < pred_rank[right]))
                total += 1
        scores.append(correct / total)
    return _mean(scores)


def compute_order_metrics(gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]) -> dict[str, float]:
    return {
        "exact_order_accuracy": exact_order_accuracy(gold_orders, predicted_orders),
        "position_accuracy": position_accuracy(gold_orders, predicted_orders),
        "first_accuracy": first_accuracy(gold_orders, predicted_orders),
        "last_accuracy": last_accuracy(gold_orders, predicted_orders),
        "pairwise_relation_accuracy": pairwise_relation_accuracy(gold_orders, predicted_orders),
    }


def sample_metrics(gold_order: Iterable[int], predicted_order: Iterable[int]) -> dict[str, float | bool]:
    gold = validate_order(gold_order)
    pred = validate_order(predicted_order)
    return {
        "exact_correct": gold == pred,
        "first_correct": gold[0] == pred[0],
        "last_correct": gold[-1] == pred[-1],
        "position_accuracy": sum(int(g == p) for g, p in zip(gold, pred)) / 4.0,
    }


def _validated_pairs(
    gold_orders: Sequence[Iterable[int]], predicted_orders: Sequence[Iterable[int]]
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    if len(gold_orders) != len(predicted_orders):
        raise ValueError("gold_orders and predicted_orders must have the same length")
    return [(validate_order(gold), validate_order(pred)) for gold, pred in zip(gold_orders, predicted_orders)]


def _mean(values: Iterable[float | bool]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
