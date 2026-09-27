from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .metrics import DEFAULT_OVERLAP_THRESHOLD
from .router import RoadRouter
from .selector import FPLSelector

MODEL_NAMES = ("Length", "ConSTGAT", "POWER-S", "POWER-D")


@dataclass
class ExperimentConfig:
    city_dir: str
    output_dir: str
    n_orders: int = 20
    horizon: int = 168
    chase_length: int = 3
    seed: int = 0
    eta: float | None = None
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    weight_calibration: str = "raw"


class WeightBank:
    def __init__(self, city_dir: Path, calibration: str = "raw"):
        self.city_dir = city_dir
        self.static = [
            np.load(city_dir / "weight_length.npy", mmap_mode="r"),
            np.load(city_dir / "weight_time.npy", mmap_mode="r"),
            np.load(city_dir / "weight_power_s.npy", mmap_mode="r"),
        ]
        self.power_d = np.load(city_dir / "weight_power_d.npy", mmap_mode="r")
        self.n_edges = len(self.static[0])
        if self.power_d.shape[1] != self.n_edges:
            raise ValueError("POWER-D/topology edge count mismatch")
        if calibration not in {"raw", "median"}:
            raise ValueError("calibration must be one of: raw, median")
        self.calibration = calibration
        self.scale = self._calibration_scales()

    @staticmethod
    def _sample_positive_median(x: np.ndarray, max_samples: int = 200000) -> float:
        flat = np.asarray(x).reshape(-1)
        step = max(1, flat.size // max_samples)
        sample = flat[::step]
        sample = sample[np.isfinite(sample) & (sample > 0)]
        if sample.size == 0:
            raise ValueError("cannot calibrate weights without positive values")
        return float(np.median(sample))

    def _calibration_scales(self) -> np.ndarray:
        if self.calibration == "raw":
            return np.ones(len(MODEL_NAMES), dtype=np.float64)

        medians = [
            self._sample_positive_median(self.static[0]),
            self._sample_positive_median(self.static[1]),
            self._sample_positive_median(self.static[2]),
            self._sample_positive_median(self.power_d),
        ]
        # Positive per-model scaling does not change any standalone shortest
        # path. It only chooses a common unit before heterogeneous PWE outputs
        # are linearly mixed by the time-decay/chasing state.
        return 1.0 / np.asarray(medians, dtype=np.float64)

    def raw(self, model: int, slot: int) -> np.ndarray:
        if model < 3:
            x = self.static[model]
        else:
            x = self.power_d[slot]
        factor = float(self.scale[model])
        if factor == 1.0:
            return x
        return np.asarray(x, dtype=np.float32) * np.float32(factor)

    @staticmethod
    def _coeff(delta: int) -> float:
        return math.exp(-float(delta))

    def base_effective(self, model: int, slot: int, L: int) -> np.ndarray:
        if model < 3:
            return self.raw(model, slot)

        first = max(0, slot - L)
        out = np.zeros(self.n_edges, dtype=np.float32)
        factor = np.float32(self.scale[model])
        for s in range(first, slot + 1):
            out += (
                np.float32(self._coeff(slot - s))
                * factor
                * self.power_d[s]
            )
        return out

    def pwc_effective(self, history: list[int], slot: int, L: int) -> np.ndarray:
        first = max(0, slot - L)
        out = np.zeros(self.n_edges, dtype=np.float32)
        for s in range(first, slot + 1):
            model = int(history[s])
            out += np.float32(self._coeff(slot - s)) * self.raw(model, s)
        return out


def _load_orders(city_dir: Path, n_orders: int, horizon: int) -> list[list[dict]]:
    with (city_dir / "sampled_orders.pkl").open("rb") as f:
        slots = pickle.load(f)
    if horizon > len(slots):
        raise ValueError(f"requested T={horizon}, only {len(slots)} slots prepared")
    result = []
    for t in range(horizon):
        if len(slots[t]) < n_orders:
            raise ValueError(
                f"slot {t} has {len(slots[t])} sampled orders, need {n_orders}"
            )
        result.append(slots[t][:n_orders])
    return result


def run_experiment(config: ExperimentConfig) -> dict:
    city_dir = Path(config.city_dir)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = _load_orders(city_dir, config.n_orders, config.horizon)
    router = RoadRouter(
        city_dir, overlap_threshold=config.overlap_threshold
    )
    bank = WeightBank(city_dir, calibration=config.weight_calibration)
    selector = FPLSelector(
        n_models=len(MODEL_NAMES),
        horizon=config.horizon,
        chase_length=config.chase_length,
        seed=config.seed,
        eta=config.eta,
    )

    cumulative = np.zeros(len(MODEL_NAMES), dtype=np.float64)
    cumulative_curve = np.zeros(
        (config.horizon, len(MODEL_NAMES)), dtype=np.float64
    )
    pwc_cumulative_curve = np.zeros(config.horizon, dtype=np.float64)
    slot_baseline = np.zeros(
        (config.horizon, len(MODEL_NAMES)), dtype=np.float64
    )
    slot_pwc = np.zeros(config.horizon, dtype=np.float64)
    selected: list[int] = []
    failures = {"PWC": 0, **{name: 0 for name in MODEL_NAMES}}
    timings = []

    for t in range(config.horizon):
        tic = time.time()

        chosen = selector.choose(cumulative)
        selected.append(chosen)

        pwc_weights = bank.pwc_effective(selected, t, config.chase_length)
        pwc_res = router.evaluate(pwc_weights, orders[t])
        slot_pwc[t] = pwc_res.accuracy
        failures["PWC"] += pwc_res.failures

        for m, name in enumerate(MODEL_NAMES):
            w = bank.base_effective(m, t, config.chase_length)
            res = router.evaluate(w, orders[t])
            slot_baseline[t, m] = res.accuracy
            failures[name] += res.failures

        cumulative += slot_baseline[t]
        cumulative_curve[t] = cumulative
        pwc_cumulative_curve[t] = slot_pwc[: t + 1].sum()

        elapsed = time.time() - tic
        timings.append(elapsed)
        print(
            f"slot={t:03d} selected={MODEL_NAMES[chosen]:8s} "
            f"pwc={100.0 * slot_pwc[t]:6.2f}% "
            f"base={(100.0 * slot_baseline[t]).round(2).tolist()} "
            f"sec={elapsed:.2f}",
            flush=True,
        )

    best_idx = int(np.argmax(cumulative))
    best_fixed_reward = float(cumulative[best_idx])
    pwc_reward = float(pwc_cumulative_curve[-1])

    best_fixed_accuracy_pct = 100.0 * best_fixed_reward / config.horizon
    pwc_accuracy_pct = 100.0 * pwc_reward / config.horizon
    relative_to_best_pct = (
        100.0 * (pwc_reward - best_fixed_reward) / best_fixed_reward
        if best_fixed_reward
        else 0.0
    )

    prefix_best = cumulative_curve.max(axis=1)
    prefix_regret = prefix_best - pwc_cumulative_curve
    avg_regret = prefix_regret / np.arange(1, config.horizon + 1)

    final = {
        "PWC_accuracy_pct": pwc_accuracy_pct,
        **{
            f"{MODEL_NAMES[i]}_accuracy_pct":
                100.0 * float(cumulative[i]) / config.horizon
            for i in range(len(MODEL_NAMES))
        },
        "best_fixed_model": MODEL_NAMES[best_idx],
        "best_fixed_accuracy_pct": best_fixed_accuracy_pct,
        "PWC_relative_to_best_pct": relative_to_best_pct,
        "average_regret_final": float(avg_regret[-1]),
        "PWC_cumulative_reward": pwc_reward,
        "best_fixed_cumulative_reward": best_fixed_reward,
    }

    result = {
        "config": asdict(config),
        "models": list(MODEL_NAMES),
        "selector_eta": float(selector.eta),
        "weight_calibration_scales": {
            MODEL_NAMES[i]: float(bank.scale[i])
            for i in range(len(MODEL_NAMES))
        },
        "selected_models": [MODEL_NAMES[i] for i in selected],
        "slot_pwc": slot_pwc.tolist(),
        "slot_baseline": slot_baseline.tolist(),
        "pwc_cumulative": pwc_cumulative_curve.tolist(),
        "baseline_cumulative": cumulative_curve.tolist(),
        "final": final,
        "prefix_average_regret": avg_regret.tolist(),
        "path_failures": failures,
        "slot_seconds": timings,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    np.savetxt(
        out_dir / "slot_metrics.csv",
        np.c_[
            np.arange(config.horizon),
            slot_pwc,
            slot_baseline,
            pwc_cumulative_curve,
            cumulative_curve,
        ],
        delimiter=",",
        header=(
            "slot,pwc,"
            + ",".join(MODEL_NAMES)
            + ",pwc_cum,"
            + ",".join(f"{x}_cum" for x in MODEL_NAMES)
        ),
        comments="",
    )
    return result
