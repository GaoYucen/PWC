# Porto Public Benchmark for PWC

This branch builds a student-shareable benchmark from the public Porto taxi trajectory dataset and OpenStreetMap.

The released **Medium v0.2** package is documented in [the student guide](PORTO_MEDIUM_STUDENT_GUIDE.md). The commands and output below describe the earlier Small pilot; do not use its event-based runner for the Medium real-hour protocol.

## Design goals

- no Beijing/Qingdao/Shanghai recovered research data;
- real observed taxi GPS trajectories from UCI;
- a public OSM driving network;
- deterministic trajectory-to-road recovery;
- chronological train/validation/test splits;
- candidate edge weights derived from **training trajectories only**;
- direct compatibility with PWC-style route-overlap evaluation.

## Data sources

1. UCI Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015 (Porto), CC BY 4.0.
2. OpenStreetMap road data, ODbL.

The processed benchmark must preserve the required attribution notices when redistributed.

## Pilot mapping semantics

Version 0.1 uses a deterministic **snap + bounded shortest-path bridge** map-matching baseline:

1. each 15-second GPS observation is snapped to the nearest OSM driving-network node;
2. consecutive snapped nodes are connected with a bounded shortest path by physical road length;
3. trips are rejected when snap distance, local bridge stretch, or whole-trip stretch exceeds quality gates.

This is intentionally conservative and auditable. It is not claimed to be the final HMM/FMM map matcher. The pilot establishes the public-data pipeline and benchmark interface before scaling.

## Candidate edge costs

All arrays align with `edges.csv.gz`.

- `length.npy`: physical OSM edge length in meters.
- `travel_time_static.npy`: training-trajectory travel-time estimate with free-flow shrinkage.
- `travel_time_dynamic_168.npy`: hour-of-week travel-time costs, smoothed toward the static estimate.
- `preference_static.npy`: frequency-adjusted physical-length cost from training-route usage.
- `preference_dynamic_168.npy`: hour-of-week frequency-adjusted preference cost, shrunk toward static usage.

Lower values are better for shortest-path routing.

The preference arrays are **revealed route-choice proxy costs**, not ground-truth psychological preference labels.

## Output

```text
porto-small-v0.1/
  graph.graphml
  nodes.csv.gz
  edges.csv.gz
  trajectories_train.jsonl.gz
  trajectories_val.jsonl.gz
  trajectories_test.jsonl.gz
  weights/
    length.npy
    travel_time_static.npy
    travel_time_dynamic_168.npy
    preference_static.npy
    preference_dynamic_168.npy
  baseline_scores.json
  metadata.json
```

## Leakage rule

All candidate weights are estimated from the chronologically earlier training split only. Validation/test trajectories are never used to construct edge weights.

## Build

```bash
python scripts/public_benchmark/porto_build.py prepare-graph --root /path/to/work
python scripts/public_benchmark/porto_build.py build \
  --root /path/to/work \
  --raw-zip /path/to/train.csv.zip \
  --target-trips 1000 \
  --seed 20261011
```
