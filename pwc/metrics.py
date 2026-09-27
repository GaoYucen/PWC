from collections import Counter
from typing import Mapping, Sequence

DEFAULT_OVERLAP_THRESHOLD = 0.95


def weighted_route_overlap_ratio(
    actual_edges: Sequence[int],
    predicted_edges: Sequence[int],
    edge_lengths: Mapping[int, float],
) -> float:
    """Fraction of actual physical route length covered by prediction."""
    if not actual_edges:
        return 0.0

    actual = Counter(int(x) for x in actual_edges)
    predicted = Counter(int(x) for x in predicted_edges)

    denominator = 0.0
    common = 0.0
    for edge, count in actual.items():
        length = float(edge_lengths[edge])
        denominator += length * count
        common += length * min(count, predicted.get(edge, 0))

    if denominator <= 0.0:
        return 0.0
    return common / denominator


def route_accuracy_from_overlap(
    overlap_ratio: float,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> float:
    """Paper metric: overlap length must exceed q times actual route length."""
    return float(float(overlap_ratio) > float(threshold))
