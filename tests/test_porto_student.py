import itertools
import json

import numpy as np
import pytest

from scripts.public_benchmark.hourly_tracking_benchmark import (
    Hedge, expected_reward, fpl_sequence, switching_oracle,
)
from scripts.public_benchmark.student_benchmark import verify


def test_current_hour_rewards_cannot_change_current_selection():
    rewards = np.array([[0.2, 0.8], [0.1, 0.9], [1.0, 0.0]])
    changed = rewards.copy()
    changed[1:] = 1 - changed[1:]
    original, _ = fpl_sequence(rewards, eta=3, seed=4)
    counterfactual, _ = fpl_sequence(changed, eta=3, seed=4)
    assert original[:2] == counterfactual[:2]
    # A selector has no training history on the first test hour.
    assert expected_reward(lambda: Hedge(2, 100), np.array([[0.0, 1.0]])) == 0.5


def test_switching_oracle_matches_exhaustive_feasible_sequences():
    rewards = np.array([[0.1, 0.9], [1.0, 0.2], [0.3, 0.8], [0.7, 0.4]])
    for budget in range(4):
        feasible = [seq for seq in itertools.product(range(2), repeat=4)
                    if sum(a != b for a, b in zip(seq, seq[1:])) <= budget]
        expected = max(sum(rewards[t, m] for t, m in enumerate(seq)) / 4 for seq in feasible)
        assert switching_oracle(rewards, budget) == pytest.approx(expected)


def test_tampered_pickle_is_rejected_before_deserialization(tmp_path):
    unsafe = tmp_path / "sampled_orders.pkl"
    unsafe.write_bytes(b"not a pickle")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "files": {unsafe.name: {"bytes": unsafe.stat().st_size, "sha256": "0" * 64}}
    }))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path)
