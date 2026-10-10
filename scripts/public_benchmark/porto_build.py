#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import networkx as nx
import numpy as np
import osmnx as ox

PORTO_TZ = ZoneInfo("Europe/Lisbon")
CENTER = (41.1579, -8.6291)
NETWORK_RADIUS_M = 22000
RAW_BBOX = (40.90, 41.40, -8.95, -8.25)  # lat_min, lat_max, lon_min, lon_max
SAMPLE_SECONDS = 15.0

SPEED_PRIOR_KMH = {
    "motorway": 85.0,
    "motorway_link": 60.0,
    "trunk": 70.0,
    "trunk_link": 50.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 45.0,
    "secondary_link": 35.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
    "unclassified": 30.0,
}


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def highway_name(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else "unclassified"
    return str(value or "unclassified")


def speed_prior(highway: str) -> float:
    return SPEED_PRIOR_KMH.get(highway, 30.0)


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371008.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def polyline_length_m(points) -> float:
    return sum(
        haversine_m(a[0], a[1], b[0], b[1])
        for a, b in zip(points[:-1], points[1:])
    )


def hour_of_week(ts: float) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(PORTO_TZ)
    return dt.weekday() * 24 + dt.hour


def prepare_graph(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "osmnx-cache"
    cache.mkdir(exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache)
    ox.settings.log_console = True
    graph_path = root / "porto_drive.graphml"
    if graph_path.exists():
        print(f"graph already exists: {graph_path}")
        return
    print(f"downloading OSM drive network around {CENTER}, radius={NETWORK_RADIUS_M} m", flush=True)
    g = ox.graph.graph_from_point(
        CENTER,
        dist=NETWORK_RADIUS_M,
        network_type="drive",
        simplify=True,
        retain_all=False,
    )
    ox.io.save_graphml(g, filepath=graph_path)
    print(f"saved {graph_path}: nodes={len(g.nodes):,}, edges={len(g.edges):,}", flush=True)


def to_simple_digraph(g):
    dg = nx.DiGraph()
    for n, d in g.nodes(data=True):
        dg.add_node(int(n), **dict(d))
    for u, v, k, d in g.edges(keys=True, data=True):
        u, v = int(u), int(v)
        length = float(d.get("length") or 1.0)
        hw = highway_name(d.get("highway"))
        item = {
            "length": max(length, 0.1),
            "highway": hw,
            "name": str(d.get("name") or ""),
            "osmid": str(d.get("osmid") or ""),
            "source_key": str(k),
        }
        if (not dg.has_edge(u, v)) or length < float(dg[u][v].get("length", float("inf"))):
            dg.add_edge(u, v, **item)
    return dg


def select_candidates(raw_zip: Path, pool_size: int, seed: int):
    heap = []
    stats = Counter()
    seed_b = str(seed).encode()
    with zipfile.ZipFile(raw_zip) as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"expected one CSV in {raw_zip}, found {csv_names[:10]}")
        with zf.open(csv_names[0], "r") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
            for row in reader:
                stats["rows"] += 1
                if str(row.get("MISSING_DATA", "")).strip().lower() not in {"false", "0"}:
                    stats["missing"] += 1
                    continue
                try:
                    pts = json.loads(row["POLYLINE"])
                    ts = int(row["TIMESTAMP"])
                except Exception:
                    stats["parse_error"] += 1
                    continue
                if not (8 <= len(pts) <= 160):
                    stats["point_count_gate"] += 1
                    continue
                try:
                    pts = [(float(p[0]), float(p[1])) for p in pts]
                except Exception:
                    stats["coord_parse_error"] += 1
                    continue
                if not all(
                    RAW_BBOX[0] <= lat <= RAW_BBOX[1] and RAW_BBOX[2] <= lon <= RAW_BBOX[3]
                    for lon, lat in pts
                ):
                    stats["bbox_gate"] += 1
                    continue
                gps_len = polyline_length_m(pts)
                if not (500.0 <= gps_len <= 35000.0):
                    stats["length_gate"] += 1
                    continue
                max_step = max(
                    haversine_m(a[0], a[1], b[0], b[1])
                    for a, b in zip(pts[:-1], pts[1:])
                )
                if max_step > 2500.0:
                    stats["jump_gate"] += 1
                    continue
                trip_id = str(row["TRIP_ID"])
                h = int.from_bytes(hashlib.blake2b(seed_b + trip_id.encode(), digest_size=8).digest(), "big")
                item = {
                    "hash": h,
                    "trip_id": trip_id,
                    "timestamp": ts,
                    "points": pts,
                    "gps_length_m": gps_len,
                }
                if len(heap) < pool_size:
                    heapq.heappush(heap, (-h, item))
                elif h < -heap[0][0]:
                    heapq.heapreplace(heap, (-h, item))
                stats["eligible"] += 1
                if stats["rows"] % 200000 == 0:
                    print(f"scanned={stats['rows']:,}, eligible={stats['eligible']:,}", flush=True)
    items = [x[1] for x in heap]
    items.sort(key=lambda x: x["hash"])
    stats["candidate_pool"] = len(items)
    return items, stats


def build_edge_index(dg):
    pairs = sorted((int(u), int(v)) for u, v in dg.edges())
    edge_idx = {p: i for i, p in enumerate(pairs)}
    n = len(pairs)
    length = np.empty(n, dtype=np.float32)
    highway = []
    rows = []
    for i, (u, v) in enumerate(pairs):
        d = dg[u][v]
        length[i] = float(d["length"])
        hw = highway_name(d.get("highway"))
        highway.append(hw)
        rows.append(
            {
                "edge_idx": i,
                "u": u,
                "v": v,
                "length_m": float(d["length"]),
                "highway": hw,
                "name": d.get("name", ""),
                "osmid": d.get("osmid", ""),
            }
        )
    return pairs, edge_idx, length, highway, rows


def bounded_path(dg, u, v, gps_distance_m):
    if u == v:
        return [u], 0.0
    cutoff = max(700.0, min(6000.0, gps_distance_m * 4.0 + 500.0))
    try:
        dist, path = nx.single_source_dijkstra(dg, u, target=v, cutoff=cutoff, weight="length")
        return path, float(dist)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None


def map_trip(g, dg, edge_idx, length_arr, item):
    points = item["points"]
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    try:
        nn, snap_dist = ox.distance.nearest_nodes(g, X=xs, Y=ys, return_dist=True)
    except Exception as e:
        return None, f"nearest:{type(e).__name__}"
    nn = [int(x) for x in np.atleast_1d(nn)]
    snap = np.asarray(snap_dist, dtype=float)
    if float(np.quantile(snap, 0.95)) > 100.0 or float(np.max(snap)) > 180.0:
        return None, "snap_distance"

    route_nodes = []
    route_edges = []
    segment_records = []
    total_bridge = 0.0
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        u, v = nn[i], nn[i + 1]
        gps_d = haversine_m(a[0], a[1], b[0], b[1])
        if u == v:
            continue
        path, bridge_len = bounded_path(dg, u, v, gps_d)
        if path is None or len(path) < 2:
            return None, "no_bridge"
        if gps_d >= 30.0 and bridge_len / gps_d > 4.5:
            return None, "segment_stretch"
        eids = []
        lens = []
        for x, y in zip(path[:-1], path[1:]):
            eid = edge_idx.get((int(x), int(y)))
            if eid is None:
                return None, "edge_index"
            eids.append(eid)
            lens.append(float(length_arr[eid]))
        if not eids:
            continue
        seg_len = float(sum(lens))
        alloc = [SAMPLE_SECONDS * l / seg_len for l in lens]
        slot = hour_of_week(item["timestamp"] + i * SAMPLE_SECONDS)
        segment_records.append((slot, eids, alloc))
        total_bridge += seg_len
        if not route_nodes:
            route_nodes.extend(int(x) for x in path)
        else:
            route_nodes.extend(int(x) for x in path[1:])
        route_edges.extend(eids)

    if len(route_edges) < 3:
        return None, "too_short_matched"
    ratio = total_bridge / max(item["gps_length_m"], 1.0)
    if not (0.60 <= ratio <= 2.60):
        return None, "trip_stretch"
    out = {
        "trip_id": hashlib.sha256(item["trip_id"].encode()).hexdigest()[:20],
        "timestamp": int(item["timestamp"]),
        "start_hour_of_week": hour_of_week(item["timestamp"]),
        "gps_point_count": len(points),
        "gps_length_m": float(item["gps_length_m"]),
        "matched_length_m": float(total_bridge),
        "stretch_ratio": float(ratio),
        "snap_p50_m": float(np.median(snap)),
        "snap_p95_m": float(np.quantile(snap, 0.95)),
        "route_nodes": route_nodes,
        "route_edges": route_edges,
        "_segments": segment_records,
    }
    return out, None


def write_csv_gz(path: Path, rows, fields):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_jsonl_gz(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(public, separators=(",", ":")) + "\n")


def build_weights(train, length_arr, highways, n_edges):
    static_time_sum = np.zeros(n_edges, dtype=np.float64)
    static_time_count = np.zeros(n_edges, dtype=np.float64)
    dynamic_time_sum = np.zeros((168, n_edges), dtype=np.float64)
    dynamic_time_count = np.zeros((168, n_edges), dtype=np.float32)
    static_pref_count = np.zeros(n_edges, dtype=np.float64)
    dynamic_pref_count = np.zeros((168, n_edges), dtype=np.float32)

    for trip in train:
        static_pref_count += np.bincount(trip["route_edges"], minlength=n_edges)
        for slot, eids, alloc in trip["_segments"]:
            ids = np.asarray(eids, dtype=np.int64)
            vals = np.asarray(alloc, dtype=np.float64)
            np.add.at(static_time_sum, ids, vals)
            np.add.at(static_time_count, ids, 1.0)
            np.add.at(dynamic_time_sum[slot], ids, vals)
            np.add.at(dynamic_time_count[slot], ids, 1.0)
            np.add.at(dynamic_pref_count[slot], ids, 1.0)

    prior_speed = np.asarray([speed_prior(h) for h in highways], dtype=np.float64)
    free_time = length_arr.astype(np.float64) / (prior_speed * 1000.0 / 3600.0)
    lambda_static = 3.0
    tt_static = (static_time_sum + lambda_static * free_time) / (static_time_count + lambda_static)
    lambda_dyn = 2.0
    tt_dynamic = (
        dynamic_time_sum + lambda_dyn * tt_static[None, :]
    ) / (dynamic_time_count.astype(np.float64) + lambda_dyn)

    density = static_pref_count / np.maximum(length_arr.astype(np.float64) / 1000.0, 0.05)
    signal = np.log1p(density)
    pos = signal[signal > 0]
    scale = float(np.median(pos)) if len(pos) else 1.0
    normalized = signal / max(scale, 1e-6)
    pref_static = length_arr.astype(np.float64) / (1.0 + 0.50 * normalized)

    pref_dynamic = np.empty((168, n_edges), dtype=np.float64)
    static_per_slot = static_pref_count / 168.0
    for slot in range(168):
        smooth_count = dynamic_pref_count[slot].astype(np.float64) + 0.50 * static_per_slot
        d = smooth_count / np.maximum(length_arr.astype(np.float64) / 1000.0, 0.05)
        sig = np.log1p(d)
        norm = sig / max(scale, 1e-6)
        pref_dynamic[slot] = length_arr.astype(np.float64) / (1.0 + 0.50 * norm)

    return {
        "length": length_arr.astype(np.float32),
        "travel_time_static": tt_static.astype(np.float32),
        "travel_time_dynamic_168": tt_dynamic.astype(np.float32),
        "preference_static": pref_static.astype(np.float32),
        "preference_dynamic_168": pref_dynamic.astype(np.float32),
        "train_edge_traversal_count": static_pref_count.astype(np.float32),
    }


def overlap_score(obs_edges, pred_edges, length_arr):
    obs = set(int(x) for x in obs_edges)
    pred = set(int(x) for x in pred_edges)
    denom = float(sum(float(length_arr[i]) for i in obs))
    if denom <= 0:
        return 0.0
    return float(sum(float(length_arr[i]) for i in obs.intersection(pred))) / denom


def shortest_edge_route(dg, edge_idx, start, end, weights):
    def wf(u, v, d):
        eid = edge_idx.get((int(u), int(v)))
        if eid is None:
            return float("inf")
        return float(weights[eid])
    try:
        path = nx.shortest_path(dg, int(start), int(end), weight=wf)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return [edge_idx[(int(u), int(v))] for u, v in zip(path[:-1], path[1:])]


def evaluate_baselines(test, dg, edge_idx, length_arr, weights, max_eval=200):
    methods = {
        "Length": weights["length"],
        "TravelTime-Static": weights["travel_time_static"],
        "Preference-Static": weights["preference_static"],
    }
    scores = {k: [] for k in ["Length", "TravelTime-Static", "TravelTime-Dynamic", "Preference-Static", "Preference-Dynamic"]}
    used = 0
    for trip in test[:max_eval]:
        if len(trip["route_nodes"]) < 2:
            continue
        s, t = trip["route_nodes"][0], trip["route_nodes"][-1]
        for name, w in methods.items():
            pred = shortest_edge_route(dg, edge_idx, s, t, w)
            if pred is not None:
                scores[name].append(overlap_score(trip["route_edges"], pred, length_arr))
        slot = int(trip["start_hour_of_week"])
        pred = shortest_edge_route(dg, edge_idx, s, t, weights["travel_time_dynamic_168"][slot])
        if pred is not None:
            scores["TravelTime-Dynamic"].append(overlap_score(trip["route_edges"], pred, length_arr))
        pred = shortest_edge_route(dg, edge_idx, s, t, weights["preference_dynamic_168"][slot])
        if pred is not None:
            scores["Preference-Dynamic"].append(overlap_score(trip["route_edges"], pred, length_arr))
        used += 1
    return {
        "evaluated_test_trips": used,
        "mean_physical_length_overlap": {
            k: (float(np.mean(v)) if v else None) for k, v in scores.items()
        },
        "n_success": {k: len(v) for k, v in scores.items()},
    }


def build(args):
    root = Path(args.root)
    graph_path = root / "porto_drive.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"missing graph: {graph_path}; run prepare-graph first")
    out = root / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)

    print("loading OSM graph", flush=True)
    g = ox.io.load_graphml(graph_path)
    dg = to_simple_digraph(g)
    pairs, edge_idx, length_arr, highways, edge_rows = build_edge_index(dg)
    print(f"simple directed graph nodes={len(dg.nodes):,}, edges={len(pairs):,}", flush=True)

    pool_size = max(args.target_trips * args.candidate_multiplier, args.target_trips)
    print(f"selecting deterministic candidate pool size={pool_size:,}", flush=True)
    candidates, scan_stats = select_candidates(Path(args.raw_zip), pool_size, args.seed)
    print(f"candidate pool={len(candidates):,}", flush=True)

    matched = []
    reject = Counter()
    for j, item in enumerate(candidates, 1):
        trip, why = map_trip(g, dg, edge_idx, length_arr, item)
        if trip is None:
            reject[why] += 1
        else:
            matched.append(trip)
        if j % 100 == 0:
            print(f"mapped candidates={j:,}, accepted={len(matched):,}", flush=True)
        if len(matched) >= args.target_trips:
            break
    if len(matched) < max(50, args.target_trips // 2):
        raise RuntimeError(f"too few mapped trips: {len(matched)} / target {args.target_trips}")

    matched.sort(key=lambda x: x["timestamp"])
    n = len(matched)
    a = max(1, int(n * 0.70))
    b = max(a + 1, int(n * 0.85))
    train = matched[:a]
    val = matched[a:b]
    test = matched[b:]
    if not test:
        raise RuntimeError("empty test split")

    print(f"splits train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    weights = build_weights(train, length_arr, highways, len(pairs))
    for name, arr in weights.items():
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"non-finite values in {name}")
        if name != "train_edge_traversal_count" and not np.all(arr > 0):
            raise RuntimeError(f"non-positive values in {name}")

    node_rows = []
    for n_id, d in dg.nodes(data=True):
        node_rows.append({"node_id": int(n_id), "lon": float(d.get("x", 0.0)), "lat": float(d.get("y", 0.0))})
    write_csv_gz(out / "nodes.csv.gz", node_rows, ["node_id", "lon", "lat"])
    write_csv_gz(out / "edges.csv.gz", edge_rows, ["edge_idx", "u", "v", "length_m", "highway", "name", "osmid"])
    write_jsonl_gz(out / "trajectories_train.jsonl.gz", train)
    write_jsonl_gz(out / "trajectories_val.jsonl.gz", val)
    write_jsonl_gz(out / "trajectories_test.jsonl.gz", test)

    np.save(out / "weights" / "length.npy", weights["length"])
    np.save(out / "weights" / "travel_time_static.npy", weights["travel_time_static"])
    np.save(out / "weights" / "travel_time_dynamic_168.npy", weights["travel_time_dynamic_168"])
    np.save(out / "weights" / "preference_static.npy", weights["preference_static"])
    np.save(out / "weights" / "preference_dynamic_168.npy", weights["preference_dynamic_168"])
    np.save(out / "weights" / "train_edge_traversal_count.npy", weights["train_edge_traversal_count"])
    ox.io.save_graphml(g, filepath=out / "graph.graphml")

    print("evaluating candidate costs on held-out trips", flush=True)
    baseline = evaluate_baselines(test, dg, edge_idx, length_arr, weights, max_eval=args.max_eval)
    jdump(out / "baseline_scores.json", baseline)

    covered = int(np.count_nonzero(weights["train_edge_traversal_count"] > 0))
    ts = [x["timestamp"] for x in matched]
    metadata = {
        "schema_version": 1,
        "name": args.name,
        "source": {
            "trajectory_dataset": "UCI Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015",
            "trajectory_license": "CC BY 4.0",
            "trajectory_doi": "10.24432/C55W25",
            "road_network": "OpenStreetMap",
            "road_network_license": "ODbL",
        },
        "mapping": {
            "method": "nearest-node snap + bounded shortest-path bridge v0.1",
            "gps_interval_seconds": 15,
            "snap_p95_gate_m": 100,
            "snap_max_gate_m": 180,
            "segment_stretch_max": 4.5,
            "trip_stretch_range": [0.60, 2.60],
        },
        "split": {
            "policy": "chronological 70/15/15 after deterministic hash candidate selection and mapping quality gates",
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "train_latest_timestamp": train[-1]["timestamp"],
            "validation_latest_timestamp": val[-1]["timestamp"] if val else None,
            "test_latest_timestamp": test[-1]["timestamp"],
            "all_earliest_timestamp": min(ts),
            "all_latest_timestamp": max(ts),
            "leakage_rule": "all candidate edge weights estimated from training trajectories only",
        },
        "graph": {
            "nodes": len(dg.nodes),
            "directed_edges": len(pairs),
            "train_observed_edges": covered,
            "train_observed_edge_fraction": covered / max(len(pairs), 1),
        },
        "weights": {
            "length": "OSM physical edge length in metres",
            "travel_time_static": "15-second GPS segment time allocated by matched edge length and shrunk to OSM road-class free-flow prior",
            "travel_time_dynamic_168": "hour-of-week travel-time estimate shrunk to static travel time",
            "preference_static": "physical length discounted by log historical training-route traversal density",
            "preference_dynamic_168": "hour-of-week traversal-density preference proxy shrunk to static usage",
            "direction": "lower cost is preferred for routing",
        },
        "selection_scan_stats": dict(scan_stats),
        "map_rejections": dict(reject),
        "seed": args.seed,
    }
    jdump(out / "metadata.json", metadata)
    summary = {
        "output": str(out),
        "mapped_trips": len(matched),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "graph_nodes": len(dg.nodes),
        "graph_edges": len(pairs),
        "train_observed_edges": covered,
        "baseline_scores": baseline,
        "mapping_rejections": dict(reject),
    }
    jdump(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare-graph")
    p.add_argument("--root", required=True)
    b = sp.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--raw-zip", required=True)
    b.add_argument("--name", default="porto-small-v0.1")
    b.add_argument("--target-trips", type=int, default=1000)
    b.add_argument("--candidate-multiplier", type=int, default=6)
    b.add_argument("--max-eval", type=int, default=200)
    b.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()
    if args.cmd == "prepare-graph":
        prepare_graph(Path(args.root))
    else:
        build(args)


if __name__ == "__main__":
    main()
