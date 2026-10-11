import math
from types import SimpleNamespace

from pwc.experiment import ExperimentConfig, batch_reward
from pwc.metrics import route_accuracy_from_overlap, weighted_route_overlap_ratio
from pwc.selector import FPLSelector


def test_weighted_overlap_uses_road_length_not_edge_count():
    lengths = {1: 90.0, 2: 5.0, 3: 5.0, 9: 5.0}
    actual = [1, 2, 3]
    predicted = [1, 9]
    assert math.isclose(
        weighted_route_overlap_ratio(actual, predicted, lengths), 0.9
    )


def test_weighted_overlap_preserves_repeated_edges():
    lengths = {1: 10.0, 2: 5.0}
    assert math.isclose(
        weighted_route_overlap_ratio([1, 1, 2], [1, 2], lengths),
        15.0 / 25.0,
    )


def test_paper_q_is_strictly_exceeded():
    assert route_accuracy_from_overlap(0.951, threshold=0.95) == 1.0
    assert route_accuracy_from_overlap(0.95, threshold=0.95) == 0.0


def test_default_experiment_reward_is_continuous_overlap():
    cfg = ExperimentConfig(city_dir="x", output_dir="y")
    assert cfg.reward_metric == "continuous-overlap"


def test_batch_reward_continuous_overlap_uses_mean_overlap():
    result = SimpleNamespace(
        accuracy=0.5,
        per_order_overlap=[0.82, 0.94],
    )
    assert math.isclose(
        batch_reward(result, "continuous-overlap"),
        0.88,
    )


def test_batch_reward_binary_threshold_uses_accuracy():
    result = SimpleNamespace(
        accuracy=0.5,
        per_order_overlap=[0.82, 0.94],
    )
    assert math.isclose(
        batch_reward(result, "binary-threshold"),
        0.5,
    )


def test_batch_reward_rejects_unknown_metric():
    result = SimpleNamespace(accuracy=0.0, per_order_overlap=[])
    try:
        batch_reward(result, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown reward metric should be rejected")


def test_selector_shape_and_positive_eta():
    s = FPLSelector(4, horizon=168, chase_length=3, seed=0)
    assert s.eta > 0
    assert math.isclose(s.eta, math.sqrt(math.log(4) / (3 * 168)))
    idx = s.choose([0.0, 0.0, 0.0, 0.0])
    assert 0 <= idx < 4
