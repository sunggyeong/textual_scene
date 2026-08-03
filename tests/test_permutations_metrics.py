from textual_scene.metrics import compute_order_metrics
from textual_scene.permutations import CLASS_TO_ORDER, ORDER_TO_CLASS, PERMUTATIONS, class_to_order, order_to_class


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
