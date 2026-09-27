import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FPLSelector:
    """Follow-the-Perturbed-Leader selector from Algorithm 1.

    Selection at slot t uses cumulative rewards through t-1 only. The caller
    must update cumulative rewards after observing slot t.
    """
    n_models: int
    horizon: int
    chase_length: int
    seed: int = 0
    eta: float | None = None

    def __post_init__(self) -> None:
        if self.n_models < 2:
            raise ValueError("PWC needs at least two candidate models")
        if self.horizon <= 0 or self.chase_length <= 0:
            raise ValueError("horizon and chase_length must be positive")
        if self.eta is None:
            self.eta = math.sqrt(
                math.log(self.n_models) / (self.chase_length * self.horizon)
            )
        self.rng = np.random.default_rng(self.seed)

    def choose(self, historical_cumulative_accuracy: Sequence[float]) -> int:
        scores = np.asarray(historical_cumulative_accuracy, dtype=np.float64)
        if scores.shape != (self.n_models,):
            raise ValueError("unexpected cumulative-accuracy shape")
        # Density eta * exp(-eta*x), x >= 0.
        perturbation = self.rng.exponential(scale=1.0 / float(self.eta),
                                            size=self.n_models)
        return int(np.argmax(scores + perturbation))
