#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pwc.experiment import WeightBank
from pwc.router import RoadRouter
from pwc.selector import FPLSelector

MODEL_NAMES = ("Length", "TravelTime-Static", "Preference-Static", "Preference-Dynamic")


def read_jsonl_gz(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_edges(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda x: int(x["edge_idx"]))
    return rows


def prepare_city(benchmark: Path, out: Path, n_orders: int):
    out.mkdir(parents=True, exist_ok=True)
    edges = read_edges(benchmark / "edges.csv.gz")
    val = read_jsonl_gz(benchmark / "trajectories_val.jsonl.gz")
    test = read_jsonl_gz(benchmark / "trajectories_test.jsonl.gz")
    heldout = sorted(val + test, key=lambda x: int(x["timestamp"]))
    horizon = len(heldout) // n_orders
    if horizon < 5:
        raise ValueError(f"too few event slots: N={n_orders}, T={horizon}")
    heldout = heldout[: horizon * n_orders]

    node_ids = np.unique(np.array(
        [int(r["u"]) for r in edges] + [int(r["v"]) for r in edges], dtype=np.int64
    ))
    node_to_idx = {int(x): i for i, x in enumerate(node_ids)}
    edge_u = np.array([node_to_idx[int(r["u"])] for r in edges], dtype=np.int32)
    edge_v = np.array([node_to_idx[int(r["v"])] for r in edges], dtype=np.int32)
    n_nodes = len(node_ids)
    n_edges = len(edges)

    keys = edge_u.astype(np.int64) * np.int64(n_nodes) + edge_v.astype(np.int64)
    sort_order = np.argsort(keys, kind="stable").astype(np.int32)
    sorted_keys = keys[sort_order]
    group_starts = np.r_[0, 1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])].astype(np.int32)
    unique_keys = sorted_keys[group_starts]
    unique_u = (unique_keys // n_nodes).astype(np.int32)
    unique_v = (unique_keys % n_nodes).astype(np.int32)
    counts_u = np.bincount(unique_u, minlength=n_nodes)
    indptr = np.r_[0, np.cumsum(counts_u, dtype=np.int64)]
    np.savez(
        out / "topology.npz",
        link_ids=np.arange(n_edges, dtype=np.int64),
        node_ids=node_ids,
        edge_u=edge_u,
        edge_v=edge_v,
        sort_order=sort_order,
        group_starts=group_starts,
        indices=unique_v,
        indptr=indptr,
        n_nodes=np.asarray([n_nodes], dtype=np.int64),
    )

    wdir = benchmark / "weights"
    length = np.load(wdir / "length.npy")
    time_static = np.load(wdir / "travel_time_static.npy")
    pref_static = np.load(wdir / "preference_static.npy")
    pref_dynamic_168 = np.load(wdir / "preference_dynamic_168.npy")
    if pref_dynamic_168.shape != (168, n_edges):
        raise ValueError(f"unexpected dynamic preference shape {pref_dynamic_168.shape}")

    np.save(out / "weight_length.npy", length)
    np.save(out / "weight_time.npy", time_static)
    np.save(out / "weight_power_s.npy", pref_static)

    slots = []
    aligned_dynamic = np.empty((horizon, n_edges), dtype=np.float32)
    slot_meta = []
    for t in range(horizon):
        batch = heldout[t * n_orders : (t + 1) * n_orders]
        orders = []
        hours = []
        for tr in batch:
            route_raw = [int(x) for x in tr["route_nodes"]]
            route = tuple(node_to_idx[x] for x in route_raw)
            hours.append(int(tr["start_hour_of_week"]))
            orders.append({
                "order_id": tr["trip_id"],
                "timestamp": int(tr["timestamp"]),
                "aux": "porto-public",
                "start_node": int(route[0]),
                "end_node": int(route[-1]),
                "route_nodes": route,
            })
        slots.append(orders)
        aligned_dynamic[t] = np.mean(pref_dynamic_168[np.asarray(hours, dtype=np.int64)], axis=0)
        slot_meta.append({
            "slot": t,
            "first_timestamp": min(int(x["timestamp"]) for x in batch),
            "last_timestamp": max(int(x["timestamp"]) for x in batch),
            "hour_of_week_values": hours,
        })

    np.save(out / "weight_power_d.npy", aligned_dynamic)
    with (out / "sampled_orders.pkl").open("wb") as f:
        pickle.dump(slots, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "slot_semantics": "event-based chronological slots from held-out Porto trajectories",
        "n_orders": n_orders,
        "horizon": horizon,
        "source_heldout_trips": len(val) + len(test),
        "used_heldout_trips": horizon * n_orders,
        "dropped_remainder": (len(val) + len(test)) - horizon * n_orders,
        "candidate_models": list(MODEL_NAMES),
        "dynamic_alignment": "slot POWER-D proxy is the mean of train-derived hour-of-week preference vectors for orders in that slot",
        "leakage": "candidate weights were produced from training trajectories only; validation/test routes are used only as online feedback/evaluation",
        "slots": slot_meta,
    }
    (out / "slot_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return horizon, meta


def run_continuous(city: Path, output: Path, horizon: int, n_orders: int, L: int, seed: int, beta_rate: float):
    output.mkdir(parents=True, exist_ok=True)
    with (city / "sampled_orders.pkl").open("rb") as f:
        slots = pickle.load(f)

    router = RoadRouter(city)
    bank = WeightBank(city, calibration="median")
    selector = FPLSelector(
        n_models=4,
        horizon=horizon,
        chase_length=L,
        seed=seed,
        eta=beta_rate,
    )

    cumulative = np.zeros(4, dtype=np.float64)
    cumulative_curve = np.zeros((horizon, 4), dtype=np.float64)
    pwc_curve = np.zeros(horizon, dtype=np.float64)
    slot_base = np.zeros((horizon, 4), dtype=np.float64)
    slot_pwc = np.zeros(horizon, dtype=np.float64)
    selected = []

    for t in range(horizon):
        chosen = selector.choose(cumulative)
        selected.append(chosen)

        pw = bank.pwc_effective(selected, t, L)
        pres = router.evaluate(pw, slots[t])
        slot_pwc[t] = float(np.mean(pres.per_order_overlap))

        for m in range(4):
            w = bank.base_effective(m, t, L)
            res = router.evaluate(w, slots[t])
            slot_base[t, m] = float(np.mean(res.per_order_overlap))

        cumulative += slot_base[t]
        cumulative_curve[t] = cumulative
        pwc_curve[t] = slot_pwc[: t + 1].sum()
        print(
            f"N={n_orders} slot={t:03d}/{horizon} select={MODEL_NAMES[chosen]:20s} "
            f"pwc={slot_pwc[t]:.4f} base={np.round(slot_base[t],4).tolist()}",
            flush=True,
        )

    best_idx = int(np.argmax(cumulative))
    best_fixed = float(cumulative[best_idx] / horizon)
    pwc_mean = float(pwc_curve[-1] / horizon)
    oracle = float(np.mean(np.max(slot_base, axis=1)))
    avg_regret = float((np.max(cumulative) - pwc_curve[-1]) / horizon)
    counts = Counter(MODEL_NAMES[i] for i in selected)
    switch_rate = float(np.mean(np.asarray(selected[1:]) != np.asarray(selected[:-1]))) if horizon > 1 else 0.0

    final = {
        "slot_semantics": "event-based chronological held-out trajectories",
        "N": n_orders,
        "T": horizon,
        "L": L,
        "seed": seed,
        "beta_rate_eta": beta_rate,
        "PWC_mean_overlap_pct": 100.0 * pwc_mean,
        **{f"{MODEL_NAMES[i]}_mean_overlap_pct": 100.0 * float(cumulative[i] / horizon) for i in range(4)},
        "best_fixed_model": MODEL_NAMES[best_idx],
        "best_fixed_mean_overlap_pct": 100.0 * best_fixed,
        "PWC_minus_best_fixed_pp": 100.0 * (pwc_mean - best_fixed),
        "slotwise_oracle_mean_overlap_pct": 100.0 * oracle,
        "oracle_headroom_over_best_fixed_pp": 100.0 * (oracle - best_fixed),
        "average_regret_final": avg_regret,
        "selection_counts": dict(counts),
        "switch_rate": switch_rate,
        "weight_calibration_scales": {MODEL_NAMES[i]: float(bank.scale[i]) for i in range(4)},
    }
    result = {
        "final": final,
        "selected_models": [MODEL_NAMES[i] for i in selected],
        "slot_pwc_overlap": slot_pwc.tolist(),
        "slot_baseline_overlap": slot_base.tolist(),
        "baseline_cumulative": cumulative_curve.tolist(),
        "pwc_cumulative": pwc_curve.tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--orders", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-rate", type=float, default=3.0)
    args = ap.parse_args()

    benchmark = Path(args.benchmark)
    work_root = Path(args.work_root)
    output_root = Path(args.output_root)
    finals = []
    for n_orders in args.orders:
        city = work_root / f"N{n_orders}"
        out = output_root / f"N{n_orders}"
        horizon, meta = prepare_city(benchmark, city, n_orders)
        finals.append(run_continuous(city, out, horizon, n_orders, args.L, args.seed, args.beta_rate))
    summary = {"runs": finals}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
