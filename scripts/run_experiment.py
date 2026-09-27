#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pwc.experiment import ExperimentConfig, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--orders", type=int, default=20)
    ap.add_argument("--T", type=int, default=168)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument(
        "--weight-calibration",
        choices=["raw", "median"],
        default="raw",
    )
    args = ap.parse_args()

    result = run_experiment(
        ExperimentConfig(
            city_dir=args.city_dir,
            output_dir=args.output,
            n_orders=args.orders,
            horizon=args.T,
            chase_length=args.L,
            seed=args.seed,
            eta=args.eta,
            overlap_threshold=args.q,
            weight_calibration=args.weight_calibration,
        )
    )
    print(json.dumps(result["final"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
