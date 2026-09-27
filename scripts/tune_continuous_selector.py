#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pwc.experiment import WeightBank, MODEL_NAMES
from pwc.router import RoadRouter
from pwc.selector import FPLSelector

SOURCE_BETA = 5.348
DEFAULT_BETAS = [1.0, 2.0, 3.0, SOURCE_BETA, 8.0, 12.0, 20.0, 100.0]
NS = [10, 15, 20]
L = 3
T = 168


def load_orders(city: Path, nmax: int = 20):
    with (city / "sampled_orders.pkl").open("rb") as f:
        slots = pickle.load(f)
    return [slot[:nmax] for slot in slots[:T]]


def precompute_baselines(city: Path, orders, cache_path: Path):
    if cache_path.exists():
        z = np.load(cache_path)
        arr = z["overlaps"]
        if arr.shape == (T, len(MODEL_NAMES), len(orders[0])):
            print(f"reuse baseline cache: {cache_path}", flush=True)
            return arr

    router = RoadRouter(city)
    bank = WeightBank(city, calibration="median")
    nmax = len(orders[0])
    overlaps = np.zeros((T, len(MODEL_NAMES), nmax), dtype=np.float32)
    for t in range(T):
        for m in range(len(MODEL_NAMES)):
            res = router.evaluate(bank.base_effective(m, t, L), orders[t])
            overlaps[t, m] = np.asarray(res.per_order_overlap, dtype=np.float32)
        if t == 0 or (t + 1) % 24 == 0:
            print(f"baseline cache {t+1}/{T}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, overlaps=overlaps)
    return overlaps


def baseline_slot_rewards(overlaps: np.ndarray, n: int):
    return overlaps[:, :, :n].mean(axis=2).astype(np.float64)


def run_pwc(city: Path, orders, baseline_rewards, beta: float, seed: int, n: int):
    router = RoadRouter(city)
    bank = WeightBank(city, calibration="median")
    selector = FPLSelector(
        n_models=len(MODEL_NAMES),
        horizon=T,
        chase_length=L,
        seed=seed,
        eta=beta,
    )

    cumulative = np.zeros(len(MODEL_NAMES), dtype=np.float64)
    slot_pwc = np.zeros(T, dtype=np.float64)
    selected = []
    seconds = []

    for t in range(T):
        tic = time.time()
        chosen = selector.choose(cumulative)
        selected.append(chosen)
        res = router.evaluate(bank.pwc_effective(selected, t, L), orders[t][:n])
        slot_pwc[t] = float(np.mean(res.per_order_overlap))
        cumulative += baseline_rewards[t]
        seconds.append(time.time() - tic)

    direct = np.asarray(
        [baseline_rewards[t, selected[t]] for t in range(T)], dtype=np.float64
    )
    best_idx = int(np.argmax(cumulative))
    best = float(cumulative[best_idx]) / T
    pwc = float(slot_pwc.mean())
    switches = sum(a != b for a, b in zip(selected[:-1], selected[1:]))

    checkpoints = {}
    for k in [24, 48, 72, 96, 120, 144, 168]:
        b = baseline_rewards[:k].sum(axis=0)
        bb = float(b.max()) / k
        pp = float(slot_pwc[:k].mean())
        checkpoints[str(k)] = {
            "pwc_pct": 100 * pp,
            "best_fixed_pct": 100 * bb,
            "gap_pp": 100 * (pp - bb),
            "average_regret": bb - pp,
        }

    return {
        "beta": beta,
        "seed": seed,
        "N": n,
        "pwc_pct": 100 * pwc,
        "best_fixed_pct": 100 * best,
        "best_fixed_model": MODEL_NAMES[best_idx],
        "gap_pp": 100 * (pwc - best),
        "selector_direct_pct": 100 * float(direct.mean()),
        "chasing_gain_pp": 100 * float(pwc - direct.mean()),
        "switches": switches,
        "switch_rate": switches / (T - 1),
        "selected_model_counts": {
            MODEL_NAMES[i]: selected.count(i) for i in range(len(MODEL_NAMES))
        },
        "mean_pwc_slot_seconds": statistics.mean(seconds),
        "checkpoints": checkpoints,
    }


def aggregate(rows):
    keys = ["pwc_pct", "best_fixed_pct", "gap_pp", "selector_direct_pct",
            "chasing_gain_pp", "switch_rate"]
    out = {"seeds": [r["seed"] for r in rows], "n_runs": len(rows)}
    for key in keys:
        vals = [r[key] for r in rows]
        out[key + "_mean"] = statistics.mean(vals)
        out[key + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    city = Path(args.city_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    orders = load_orders(city, 20)
    baseline_overlaps = precompute_baselines(
        city, orders, out / "baseline_overlap_cache_N20.npz"
    )

    stage1 = []
    rewards20 = baseline_slot_rewards(baseline_overlaps, 20)
    for beta in DEFAULT_BETAS:
        for seed in [0, 1]:
            print(f"stage1 beta={beta} seed={seed}", flush=True)
            row = run_pwc(city, orders, rewards20, beta, seed, 20)
            stage1.append(row)

    stage1_agg = {}
    for beta in DEFAULT_BETAS:
        rows = [r for r in stage1 if r["beta"] == beta]
        stage1_agg[str(beta)] = aggregate(rows)

    ranked = sorted(
        DEFAULT_BETAS,
        key=lambda b: (
            stage1_agg[str(b)]["gap_pp_mean"],
            -stage1_agg[str(b)]["gap_pp_std"],
        ),
        reverse=True,
    )
    shortlist = ranked[:4]
    if SOURCE_BETA not in shortlist:
        shortlist[-1] = SOURCE_BETA
    shortlist = list(dict.fromkeys(shortlist))

    stage2 = list(stage1)
    for beta in shortlist:
        for seed in [2, 3, 4]:
            print(f"stage2 beta={beta} seed={seed}", flush=True)
            stage2.append(run_pwc(city, orders, rewards20, beta, seed, 20))

    robust = {}
    for beta in shortlist:
        rows = [r for r in stage2 if r["beta"] == beta and r["seed"] in range(5)]
        robust[str(beta)] = aggregate(rows)

    robust_beta = max(
        shortlist,
        key=lambda b: (
            robust[str(b)]["gap_pp_mean"],
            -robust[str(b)]["gap_pp_std"],
        ),
    )

    sensitivity = {}
    for n in NS:
        rewards = baseline_slot_rewards(baseline_overlaps, n)
        rows = []
        for seed in range(5):
            print(f"sensitivity N={n} beta={robust_beta} seed={seed}", flush=True)
            rows.append(run_pwc(city, orders, rewards, robust_beta, seed, n))
        sensitivity[str(n)] = {
            "aggregate": aggregate(rows),
            "runs": rows,
        }

    source_rows = [
        r for r in stage2
        if r["beta"] == SOURCE_BETA and r["seed"] in range(5)
    ]
    source_agg = aggregate(source_rows) if len(source_rows) == 5 else None

    summary = {
        "setting": {
            "reward": "mean physical-length overlap",
            "calibration": "median",
            "T": T,
            "L_previous_slots": L,
            "sample": "canonical topology-valid reservoir",
        },
        "source_beta": SOURCE_BETA,
        "candidates": DEFAULT_BETAS,
        "stage1_2seed": stage1_agg,
        "shortlist": shortlist,
        "robust_5seed": robust,
        "recommended_beta": robust_beta,
        "source_beta_5seed": source_agg,
        "N_sensitivity": sensitivity,
        "paper_reference_N20": {
            "ConSTGAT": 70.65,
            "POWER-S": 81.15,
            "POWER-D": 82.50,
            "PWC": 83.75,
        },
    }
    (out / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "recommended_beta": robust_beta,
        "source_beta_5seed": source_agg,
        "recommended_N20": sensitivity["20"]["aggregate"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
