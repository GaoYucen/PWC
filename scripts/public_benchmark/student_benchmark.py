#!/usr/bin/env python3
"""Student entry point for the frozen Porto Medium v0.2 release."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(data):
    import numpy as np

    manifest = json.loads((data / "MANIFEST.json").read_text())
    for name, info in manifest["files"].items():
        path = data / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"missing file or unexpected size: {name}")
        if sha256(path) != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    meta = json.loads((data / "metadata.json").read_text())
    with np.load(data / "topology.npz", allow_pickle=False) as z:
        node_ids = z["node_ids"].copy()
        n = int(z["n_nodes"][0])
        edge_u, edge_v = z["edge_u"], z["edge_v"]
        e = len(edge_u)
        if len(edge_v) != e or len(node_ids) != n:
            raise ValueError("topology dimensions disagree")
        if not ((edge_u >= 0).all() and (edge_u < n).all()
                and (edge_v >= 0).all() and (edge_v < n).all()):
            raise ValueError("invalid topology node index")
        pairs = set(zip(edge_u.tolist(), edge_v.tolist()))
    for name in ("weight_length.npy", "weight_time.npy", "weight_power_s.npy", "weight_power_d.npy"):
        a = np.load(data / name, mmap_mode="r", allow_pickle=False)
        shape = (168, e) if name == "weight_power_d.npy" else (e,)
        if a.shape != shape or not np.isfinite(a).all() or not (a > 0).all():
            raise ValueError(f"invalid weights: {name}")
    node_to_idx = {int(x): i for i, x in enumerate(node_ids)}
    tz = ZoneInfo("Europe/Lisbon")
    starts = [datetime.fromisoformat(meta[k]) for k in
              ("train_start", "validation_start", "test_start", "end")]
    seen = set()
    counts = {}
    trajectory_by_id = {}
    for i, split in enumerate(("train", "val", "test")):
        count = 0
        with gzip.open(data / f"trajectories_{split}.jsonl.gz", "rt") as f:
            for line in f:
                trip = json.loads(line)
                tid = trip["trip_id"]
                dt = datetime.fromtimestamp(trip["timestamp"], tz)
                if tid in seen or not starts[i] <= dt < starts[i + 1]:
                    raise ValueError("duplicate trajectory or chronological split violation")
                seen.add(tid)
                route = tuple(node_to_idx[int(x)] for x in trip["route_nodes"])
                if len(route) < 2 or any(p not in pairs for p in zip(route[:-1], route[1:])):
                    raise ValueError(f"route absent from topology: {tid}")
                if split != "train":
                    trajectory_by_id[tid] = (route, int(trip["timestamp"]))
                count += 1
        key = "validation" if split == "val" else split
        if count != meta["mapped"][key]:
            raise ValueError(f"trajectory count mismatch: {split}")
        counts[key] = count
    # Load pickles only after all release file checksums have passed.
    for name, key, start in (("validation_orders.pkl", "validation", starts[1]),
                             ("sampled_orders.pkl", "test", starts[2])):
        with (data / name).open("rb") as f:
            slots = pickle.load(f)
        if len(slots) != 168 or [len(x) for x in slots] != meta[f"{key}_per_hour"]:
            raise ValueError(f"slot count mismatch: {name}")
        for t, slot in enumerate(slots):
            times = [int(o["timestamp"]) for o in slot]
            if times != sorted(times):
                raise ValueError("orders must be chronological inside each hour")
            for order in slot:
                route, stamp = trajectory_by_id[order["order_id"]]
                if tuple(order["route_nodes"]) != route or order["timestamp"] != stamp:
                    raise ValueError("packed order differs from trajectory source")
                if (int((datetime.fromtimestamp(stamp, tz) - start).total_seconds() // 3600) != t
                        or order["start_node"] != route[0] or order["end_node"] != route[-1]):
                    raise ValueError("order hour or endpoint mismatch")
    with np.load(data / "reference/reward_matrices.npz", allow_pickle=False) as rewards:
        for key in ("validation", "test"):
            r = rewards[key]
            if r.shape != (168, 4) or not np.isfinite(r).all() or not ((r >= 0) & (r <= 1)).all():
                raise ValueError("invalid reference reward matrix")
    return {"status": "passed", "files_verified": len(manifest["files"]),
            "nodes": n, "edges": e, "trajectories": counts,
            "test_hours": 168, "test_min_orders": meta["test_min_per_hour"],
            "manifest_sha256": sha256(data / "MANIFEST.json")}


def replay(data, seeds):
    import numpy as np
    from scripts.public_benchmark.hourly_tracking_benchmark import (
        MODELS, expected_reward, fpl_sequence, sampled_sequence, switching_oracle,
        tune_fpl, tune_prob,
    )
    with np.load(data / "reference/reward_matrices.npz", allow_pickle=False) as z:
        rv, rt = z["validation"], z["test"]
    original = json.loads((data / "reference/summary.json").read_text())
    ref = {a["name"]: a for a in original["algorithms"]}
    rows = []
    tuned, _ = tune_fpl(rv)
    for name, eta in (("FPL-beta3", 3.0), ("FPL-val-tuned", float(tuned[2]))):
        vals = [fpl_sequence(rt, eta, seed)[1] for seed in range(seeds)]
        rows.append({"name": name, "params": {"eta": eta},
                     "direct_pct": float(100 * np.mean(vals)),
                     "direct_std_pp": float(100 * np.std(vals))})
    for name in ("Hedge", "FixedShare", "DiscountedHedge", "SlidingHedge"):
        (_, params, factory), _ = tune_prob(name, rv)
        vals = [sampled_sequence(factory, rt, 1000 + seed)[1] for seed in range(seeds)]
        rows.append({"name": name, "params": params,
                     "direct_pct": float(100 * np.mean(vals)),
                     "direct_std_pp": float(100 * np.std(vals)),
                     "test_expected_direct_pct": float(100 * expected_reward(factory, rt))})
    if seeds == 20:
        for row in rows:
            if row["params"] != ref[row["name"]]["params"]:
                raise ValueError(f"validation-selected parameters differ: {row['name']}")
            for field in ("direct_pct", "direct_std_pp", "test_expected_direct_pct"):
                if field in row and not np.isclose(row[field], ref[row["name"]][field], atol=1e-9, rtol=0):
                    raise ValueError(f"reference mismatch: {row['name']} {field}")
    fixed = float(100 * rt.mean(axis=0).max())
    oracle = float(100 * rt.max(axis=1).mean())
    if not np.isclose(fixed, original["best_fixed_pct"], atol=1e-9, rtol=0) or not np.isclose(
            oracle, original["full_oracle_pct"], atol=1e-9, rtol=0):
        raise ValueError("reference comparator mismatch")
    sw = {s: 100 * switching_oracle(rt, int(s)) for s in original["switching_oracle_pct"]}
    if any(not np.isclose(v, original["switching_oracle_pct"][s], atol=1e-9, rtol=0)
           for s, v in sw.items()):
        raise ValueError("switching oracle mismatch")
    return {"verification_scope": "cached rewards and direct selector replay; no routing/chasing rerun",
            "reference_match": seeds == 20, "seeds": seeds, "models": list(MODELS),
            "base_means_pct": (100 * rt.mean(axis=0)).tolist(),
            "best_fixed_pct": fixed, "full_oracle_pct": oracle,
            "switching_oracle_pct": sw, "algorithms": rows}


def smoke(data, hours):
    import numpy as np
    from scripts.public_benchmark.hourly_tracking_benchmark import (
        baseline_rewards, chase_reward, fpl_sequence, load_orders,
    )
    orders = load_orders(data / "sampled_orders.pkl")[:hours]
    r, _ = baseline_rewards(data, orders, n_orders=5)
    with np.load(data / "reference/reward_matrices.npz", allow_pickle=False) as z:
        error = float(np.max(np.abs(r - z["test"][:hours])))
    if error > 1e-7:
        raise ValueError(f"route reward mismatch: maximum error {error}")
    seq, direct = fpl_sequence(r, 3, 0)
    chase = chase_reward(data, orders, seq, 5, 3)
    return {"verification_scope": "fresh routing smoke test; not full 168-hour reproduction",
            "hours": hours, "orders_per_hour": 5, "seed": 0, "beta": 3,
            "max_baseline_reward_error": error, "selected_models": seq,
            "direct_pct": 100 * direct, "chase_pct": 100 * chase}


def compare(data, results):
    import numpy as np

    reference = json.loads((data / "reference/summary.json").read_text())
    actual = json.loads((results / "summary.json").read_text())
    if actual["protocol"]["stochastic_seeds"] != 20 or actual["protocol"]["N_test"] != 5:
        raise ValueError("reference comparison requires N=5 and 20 seeds")
    with np.load(data / "reference/reward_matrices.npz", allow_pickle=False) as ref, np.load(
            results / "reward_matrices.npz", allow_pickle=False) as run:
        error = max(float(np.max(np.abs(ref[key] - run[key]))) for key in ("validation", "test"))
    if error > 1e-7:
        raise ValueError(f"baseline reward matrices differ: {error}")
    ref_rows = {a["name"]: a for a in reference["algorithms"]}
    if {a["name"] for a in actual["algorithms"]} != set(ref_rows):
        raise ValueError("algorithm set differs")
    for field in ("best_fixed_pct", "full_oracle_pct", "oracle_headroom_pp"):
        if not np.isclose(actual[field], reference[field], atol=1e-6, rtol=0):
            raise ValueError(f"comparator differs: {field}")
    max_error = 0.0
    for row in actual["algorithms"]:
        expected = ref_rows[row["name"]]
        if row["params"] != expected["params"]:
            raise ValueError(f"parameters differ: {row['name']}")
        for field in ("direct_pct", "direct_std_pp", "chase_pct", "chase_std_pp"):
            diff = abs(row[field] - expected[field])
            max_error = max(max_error, diff)
            if diff > 1e-6:
                raise ValueError(f"reproduction differs: {row['name']} {field}")
    records = json.loads((results / "per_seed.json").read_text())
    if len(records) != 120:
        raise ValueError("expected 6 algorithms × 20 seed records")
    return {"verification_scope": "fresh 168-hour baseline and 20-seed routing/chasing reproduction",
            "reference_match": True, "max_baseline_reward_error": error,
            "max_summary_error_pp": max_error, "seed_records": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "replay", "smoke", "compare"))
    parser.add_argument("--data", type=Path, default=ROOT / "data/porto-medium-hourly-v0.2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--hours", type=int, default=4)
    parser.add_argument("--results", type=Path, help="full experiment output for compare")
    args = parser.parse_args()
    if args.seeds < 1 or not 1 <= args.hours <= 168:
        parser.error("seeds must be positive and hours must be in 1..168")
    result = {"data_verification": verify(args.data)}
    if args.mode == "replay":
        result["result"] = replay(args.data, args.seeds)
    elif args.mode == "smoke":
        result["result"] = smoke(args.data, args.hours)
    elif args.mode == "compare":
        if not args.results:
            parser.error("compare requires --results")
        result["result"] = compare(args.data, args.results)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
