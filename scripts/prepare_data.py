#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pickle
import random
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

TZ_CN = timezone(timedelta(hours=8))
START_DATE = datetime(2023, 4, 16, tzinfo=TZ_CN).date()
HORIZON = 168


def member_map(tar: tarfile.TarFile):
    return {m.name.lstrip("./"): m for m in tar.getmembers()}


def open_text(tar: tarfile.TarFile, members: dict, name: str):
    raw = tar.extractfile(members[name])
    if raw is None:
        raise FileNotFoundError(name)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def load_link_info(tar: tarfile.TarFile, members: dict, city: str):
    name = f"for_exp/{city}_link_info"
    raw = tar.extractfile(members[name])
    if raw is None:
        raise FileNotFoundError(name)
    arr = np.loadtxt(raw, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"unexpected link_info shape {arr.shape}")
    return arr


def load_scalar_weights(tar, members, name, link_to_idx, n_edges):
    out = np.full(n_edges, np.nan, dtype=np.float32)
    with open_text(tar, members, name) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, sval = line.split(":", 1)
            idx = link_to_idx.get(int(sid))
            if idx is not None:
                out[idx] = float(sval)
    missing = int(np.count_nonzero(~np.isfinite(out)))
    if missing:
        raise ValueError(f"{name}: {missing} topology links have no weight")
    return out


def load_dynamic_weights(tar, members, name, link_to_idx, n_edges, out_path):
    tmp = out_path.with_suffix(".edge_time.npy")
    edge_time = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(n_edges, HORIZON)
    )
    filled = np.zeros(n_edges, dtype=np.bool_)
    seen = 0
    with open_text(tar, members, name) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, values = line.split(":", 1)
            idx = link_to_idx.get(int(sid))
            if idx is None:
                continue
            vals = np.fromstring(values, sep=",", dtype=np.float32)
            if vals.size != HORIZON:
                raise ValueError(f"{name}: expected 168 weights, got {vals.size}")
            edge_time[idx] = vals
            filled[idx] = True
            seen += 1
            if seen % 250000 == 0:
                print(f"dynamic weights: {seen:,} links", flush=True)
    missing = int(n_edges - int(filled.sum()))
    if missing:
        raise ValueError(f"{name}: {missing} topology links have no dynamic weight")

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(HORIZON, n_edges)
    )
    chunk = 100000
    for a in range(0, n_edges, chunk):
        b = min(n_edges, a + chunk)
        out[:, a:b] = edge_time[a:b].T
    out.flush()
    del out, edge_time
    tmp.unlink()


def _route_is_topology_continuous(route_nodes, n_nodes, unique_edge_keys):
    if len(route_nodes) < 2:
        return False
    r = np.asarray(route_nodes, dtype=np.int64)
    keys = r[:-1] * np.int64(n_nodes) + r[1:]
    pos = np.searchsorted(unique_edge_keys, keys)
    ok = pos < len(unique_edge_keys)
    if not np.all(ok):
        return False
    return bool(np.all(unique_edge_keys[pos] == keys))


def reservoir_sample_orders(
    tar, members, city, node_to_idx, n_nodes, unique_edge_keys, nmax, seed
):
    """Sample topology-valid real orders by hour.

    Recovered trajectory schema:
      order_id, timestamp, auxiliary_field, start_node, end_node, node_sequence

    Only routes whose every consecutive node pair exists in the directed road
    graph are eligible for the experiment. This matches the usable-order
    assumption implicit in shortest-path evaluation and avoids inventing
    connector edges that are absent from the recovered road network.
    """
    rng = random.Random(seed)
    slots = [[] for _ in range(HORIZON)]
    valid_counts = [0] * HORIZON
    stats = {
        "rows": 0,
        "malformed": 0,
        "missing_endpoint_node": 0,
        "bad_route_node": 0,
        "disconnected_route": 0,
        "endpoint_match": 0,
        "valid_orders": 0,
    }

    prefix = f"for_exp/traj/{city}/"
    traj_members = sorted(
        (
            m
            for n, m in members.items()
            if n.startswith(prefix) and "/part-" in n and m.isfile()
        ),
        key=lambda m: m.name,
    )

    for m in traj_members:
        name = m.name.lstrip("./")
        parts_name = name.split("/")
        folder_date = datetime.strptime(parts_name[3], "%Y%m%d").date()
        day = (folder_date - START_DATE).days
        raw = tar.extractfile(m)
        if raw is None:
            continue
        f = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")

        for line in f:
            stats["rows"] += 1
            cols = line.rstrip("\n").split("\t", 5)
            if len(cols) != 6:
                stats["malformed"] += 1
                continue

            oid, sts, aux, s_node_s, e_node_s, route_s = cols
            try:
                ts = int(sts)
                s_raw = int(s_node_s)
                e_raw = int(e_node_s)
                route_raw = [int(x) for x in route_s.split(",") if x]
            except ValueError:
                stats["malformed"] += 1
                continue

            s_node = node_to_idx.get(s_raw)
            e_node = node_to_idx.get(e_raw)
            if s_node is None or e_node is None:
                stats["missing_endpoint_node"] += 1
                continue
            if not route_raw:
                stats["bad_route_node"] += 1
                continue
            try:
                route_nodes = tuple(int(node_to_idx[x]) for x in route_raw)
            except KeyError:
                stats["bad_route_node"] += 1
                continue

            if not _route_is_topology_continuous(
                route_nodes, n_nodes, unique_edge_keys
            ):
                stats["disconnected_route"] += 1
                continue

            hour = datetime.fromtimestamp(ts, TZ_CN).hour
            slot = day * 24 + hour
            if not (0 <= slot < HORIZON):
                continue

            if route_raw[0] == s_raw and route_raw[-1] == e_raw:
                stats["endpoint_match"] += 1

            stats["valid_orders"] += 1
            valid_counts[slot] += 1
            k = valid_counts[slot]

            order = {
                "order_id": oid,
                "timestamp": ts,
                "aux": aux,
                "start_node": int(s_node),
                "end_node": int(e_node),
                "route_nodes": route_nodes,
            }

            if len(slots[slot]) < nmax:
                slots[slot].append(order)
            else:
                j = rng.randrange(k)
                if j < nmax:
                    slots[slot][j] = order

    for x in slots:
        rng.shuffle(x)
    return slots, valid_counts, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument(
        "--city",
        choices=["beijing", "shanghai", "qingdao"],
        required=True,
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-max", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    out = Path(args.output) / args.city
    out.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.tar, "r:") as tar:
        members = member_map(tar)
        links = load_link_info(tar, members, args.city)
        link_ids = links[:, 0].astype(np.int64)
        u_node = links[:, 1].astype(np.int64)
        v_node = links[:, 2].astype(np.int64)
        n_edges = len(link_ids)
        print(f"{args.city}: {n_edges:,} edges", flush=True)

        if len(np.unique(link_ids)) != n_edges:
            raise ValueError("link ids are not unique")
        link_to_idx = {int(x): i for i, x in enumerate(link_ids)}

        node_ids = np.unique(np.r_[u_node, v_node])
        node_to_idx = {int(x): i for i, x in enumerate(node_ids)}
        edge_u = np.searchsorted(node_ids, u_node).astype(np.int32)
        edge_v = np.searchsorted(node_ids, v_node).astype(np.int32)
        n_nodes = len(node_ids)
        print(f"{args.city}: {n_nodes:,} nodes", flush=True)

        keys = edge_u.astype(np.int64) * np.int64(n_nodes) + edge_v.astype(
            np.int64
        )
        sort_order = np.argsort(keys, kind="stable").astype(np.int32)
        sorted_keys = keys[sort_order]
        group_starts = np.r_[
            0, 1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])
        ].astype(np.int32)
        unique_keys = sorted_keys[group_starts]
        unique_u = (unique_keys // n_nodes).astype(np.int32)
        unique_v = (unique_keys % n_nodes).astype(np.int32)
        counts_u = np.bincount(unique_u, minlength=n_nodes)
        indptr = np.r_[0, np.cumsum(counts_u, dtype=np.int64)]
        np.savez(
            out / "topology.npz",
            link_ids=link_ids,
            node_ids=node_ids,
            edge_u=edge_u,
            edge_v=edge_v,
            sort_order=sort_order,
            group_starts=group_starts,
            indices=unique_v,
            indptr=indptr,
            n_nodes=np.asarray([n_nodes], dtype=np.int64),
        )

        specs = [
            ("weight_length.npy", f"for_exp/{args.city}_dist.txt"),
            ("weight_time.npy", f"for_exp/{args.city}_time.txt"),
            ("weight_power_s.npy", f"for_exp/{args.city}_sim_weight.txt"),
        ]
        for filename, member in specs:
            print(f"loading {member}", flush=True)
            w = load_scalar_weights(tar, members, member, link_to_idx, n_edges)
            np.save(out / filename, w)

        print("loading POWER-D 168-slot weights", flush=True)
        load_dynamic_weights(
            tar,
            members,
            f"for_exp/{args.city}_sim_weight_v2.txt",
            link_to_idx,
            n_edges,
            out / "weight_power_d.npy",
        )

        print("sampling topology-valid real node trajectories", flush=True)
        slots, counts, stats = reservoir_sample_orders(
            tar,
            members,
            args.city,
            node_to_idx,
            n_nodes,
            unique_keys,
            args.sample_max,
            args.seed,
        )
        with (out / "sampled_orders.pkl").open("wb") as f:
            pickle.dump(slots, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "city": args.city,
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "horizon": HORIZON,
        "sample_max": args.sample_max,
        "seed": args.seed,
        "valid_orders_per_slot": counts,
        "sampled_per_slot": [len(x) for x in slots],
        "trajectory_stats": stats,
        "trajectory_schema": (
            "order_id,timestamp,aux,start_node,end_node,node_sequence"
        ),
        "route_filter": "every consecutive node pair must exist in directed topology",
        "model_file_mapping": {
            "Length": "dist.txt",
            "ConSTGAT": "time.txt",
            "POWER-S": "sim_weight.txt",
            "POWER-D": "sim_weight_v2.txt (168 values/link)",
        },
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
