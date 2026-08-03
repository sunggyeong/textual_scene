"""Canonical 24-class mapping for four-frame temporal orders."""

from __future__ import annotations

from itertools import permutations
from typing import Iterable

FrameOrder = tuple[int, int, int, int]

PERMUTATIONS: tuple[FrameOrder, ...] = tuple(permutations((1, 2, 3, 4)))
ORDER_TO_CLASS: dict[FrameOrder, int] = {order: idx for idx, order in enumerate(PERMUTATIONS)}
CLASS_TO_ORDER: dict[int, FrameOrder] = {idx: order for order, idx in ORDER_TO_CLASS.items()}


def validate_order(order: Iterable[int]) -> FrameOrder:
    """Return a normalized order tuple or raise ValueError."""
    normalized = tuple(int(item) for item in order)
    if len(normalized) != 4:
        raise ValueError(f"Expected four frame ids, got {normalized!r}")
    if set(normalized) != {1, 2, 3, 4}:
        raise ValueError(f"Expected a permutation of 1,2,3,4, got {normalized!r}")
    return normalized  # type: ignore[return-value]


def order_to_class(order: Iterable[int]) -> int:
    return ORDER_TO_CLASS[validate_order(order)]


def class_to_order(class_index: int) -> FrameOrder:
    idx = int(class_index)
    if idx not in CLASS_TO_ORDER:
        raise ValueError(f"Class index must be in [0, 23], got {class_index!r}")
    return CLASS_TO_ORDER[idx]
