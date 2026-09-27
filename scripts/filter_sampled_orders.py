#!/usr/bin/env python3
"""Filter an existing deterministic sample to topology-continuous routes.

This is a migration helper for processed data produced before the continuity
filter was added to prepare_data.py. New full preparations do not need it.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-dir", required=True)
    ap.add_argument("--min-required", type=int, default=20)
    args = ap.parse_args()

    city = Path(args.city_dir)
    z = np.load(city / "topology.npz", mmap_mode="r")
    n = int(z["n_nodes"][0])
    edge_u = np.asarray(z["edge_u"], dtype=np.int64)
    edge_v = np.asarray(z["edge_v"], dtype=np.int64)
    unique_keys = np.unique(edge_u * np.int64(n) + edge_v)

    p = city / "sampled_orders.pkl"
    with p.open("rb") as f:
        slots = pickle.load(f)

    filtered = []
    counts = []
    for slot in slots:
        keep = []
        for o in slot:
            r = np.asarray(o["route_nodes"], dtype=np.int64)
            keys = r[:-1] * np.int64(n) + r[1:]
            pos = np.searchsorted(unique_keys, keys)
            ok = pos < len(unique_keys)
            valid = bool(np.all(ok)) and bool(
                np.all(unique_keys[pos] == keys)
            )
            if valid:
                keep.append(o)
        filtered.append(keep)
        counts.append(len(keep))

    if min(counts) < args.min_required:
        raise RuntimeError(
            f"only {min(counts)} topology-valid orders in the sparsest slot; "
            f"need {args.min_required}"
        )

    backup = city / "sampled_orders.pre_continuity_filter.pkl"
    if not backup.exists():
        p.replace(backup)
    with p.open("wb") as f:
        pickle.dump(filtered, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_path = city / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["sampled_per_slot"] = counts
    meta["continuity_filter_migrated"] = True
    meta["route_filter"] = (
        "every consecutive node pair must exist in directed topology"
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({
        "min_valid_per_slot": min(counts),
        "max_valid_per_slot": max(counts),
        "valid_total": sum(counts),
        "backup": str(backup),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
