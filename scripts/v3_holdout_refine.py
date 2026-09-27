#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.tune_continuous_selector as tune


def positive_count(rows):
    return sum(float(r["gap_pp"]) >= 0.0 for r in rows)


def gate(rows):
    agg = tune.aggregate(rows)
    pos = positive_count(rows)
    mean_gap = float(agg["gap_pp_mean"])
    if mean_gap > 0.0 and pos >= 3:
        state = "GREEN"
    elif mean_gap >= -0.20 or pos >= 3:
        state = "YELLOW"
    else:
        state = "RED"
    return state, agg, pos


def run_rows(city, orders, baseline, beta, seeds, n=20):
    rows = []
    for seed in seeds:
        print(f"holdout/refine beta={beta} seed={seed} N={n}", flush=True)
        rows.append(tune.run_pwc(city, orders, baseline, float(beta), int(seed), int(n)))
    return rows


def unique_sorted(values):
    out = []
    for x in values:
        x = float(x)
        if not any(abs(x-y) < 1e-12 for y in out):
            out.append(x)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-dir", required=True)
    ap.add_argument("--autotune-summary", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    city = Path(args.city_dir)
    auto_path = Path(args.autotune_summary)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    auto = json.loads(auto_path.read_text())

    orders = tune.load_orders(city, 20)
    cache = auto_path.parent / "baseline_overlap_cache_N20.npz"
    baseline_overlaps = tune.precompute_baselines(city, orders, cache)
    rewards20 = tune.baseline_slot_rewards(baseline_overlaps, 20)

    selected_beta = float(auto["recommended_beta"])
    source_beta = float(auto.get("source_beta", tune.SOURCE_BETA))

    first = {}
    for label, beta in [("selected", selected_beta), ("source", source_beta)]:
        rows = run_rows(city, orders, rewards20, beta, range(5, 10), 20)
        state, agg, pos = gate(rows)
        first[label] = {
            "beta": beta,
            "state": state,
            "aggregate": agg,
            "positive_gap_seeds": pos,
            "runs": rows,
        }

    decision = first["selected"]["state"]
    final_beta = selected_beta
    refinement = None

    if decision == "YELLOW":
        robust = auto.get("robust_5seed") or {}
        def dev_gap(beta):
            row = robust.get(str(beta)) or robust.get(str(float(beta)))
            return float((row or {}).get("gap_pp_mean", -1e9))

        center = selected_beta
        if dev_gap(source_beta) > dev_gap(selected_beta):
            center = source_beta

        candidates = unique_sorted(
            [source_beta] + [center*m for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
        )
        screen_rows = {}
        for beta in candidates:
            rows = run_rows(city, orders, rewards20, beta, (10, 11), 20)
            screen_rows[str(beta)] = {"aggregate": tune.aggregate(rows), "runs": rows}

        ranked = sorted(
            candidates,
            key=lambda b: (
                screen_rows[str(b)]["aggregate"]["gap_pp_mean"],
                -screen_rows[str(b)]["aggregate"]["gap_pp_std"],
            ),
            reverse=True,
        )
        shortlist = ranked[:3]
        if source_beta not in shortlist:
            shortlist[-1] = source_beta
        shortlist = unique_sorted(shortlist)

        robust_rows = {}
        for beta in shortlist:
            rows = list(screen_rows[str(beta)]["runs"])
            rows.extend(run_rows(city, orders, rewards20, beta, (12, 13, 14), 20))
            robust_rows[str(beta)] = {"aggregate": tune.aggregate(rows), "runs": rows}

        refined_beta = max(
            shortlist,
            key=lambda b: (
                robust_rows[str(b)]["aggregate"]["gap_pp_mean"],
                -robust_rows[str(b)]["aggregate"]["gap_pp_std"],
            ),
        )
        holdout_rows = run_rows(city, orders, rewards20, refined_beta, range(15, 20), 20)
        final_state, final_agg, final_pos = gate(holdout_rows)
        refinement = {
            "center_beta": center,
            "candidates": candidates,
            "screen_2seed": {k:v["aggregate"] for k,v in screen_rows.items()},
            "shortlist": shortlist,
            "robust_5seed": {k:v["aggregate"] for k,v in robust_rows.items()},
            "refined_beta": refined_beta,
            "fresh_holdout": {
                "state": final_state,
                "aggregate": final_agg,
                "positive_gap_seeds": final_pos,
                "runs": holdout_rows,
            },
        }
        decision = final_state
        final_beta = float(refined_beta)

    if decision == "GREEN":
        evidence_slot_closed = True
        claim_delta = "strengthened"
    elif decision == "YELLOW":
        evidence_slot_closed = False
        claim_delta = "inconclusive"
    else:
        evidence_slot_closed = True
        claim_delta = "weakened"

    if abs(final_beta - selected_beta) < 1e-12:
        sensitivity = auto["N_sensitivity"]
    else:
        sensitivity = {}
        for n in tune.NS:
            rewards = tune.baseline_slot_rewards(baseline_overlaps, n)
            rows = run_rows(city, orders, rewards, final_beta, range(5), n)
            sensitivity[str(n)] = {"aggregate": tune.aggregate(rows), "runs": rows}

    final_holdout = (
        refinement["fresh_holdout"] if refinement is not None
        else first["selected"]
    )

    result = {
        "status": "success",
        "phase": "conference-reconstruction-selector-closure-v1",
        "round_id": "PWC-V3-01",
        "decision_state": decision,
        "claim_delta": claim_delta,
        "evidence_slot_closed": evidence_slot_closed,
        "selected_beta_from_development": selected_beta,
        "source_beta": source_beta,
        "final_beta": final_beta,
        "first_holdout": first,
        "refinement_used": refinement is not None,
        "refinement": refinement,
        "final_holdout": final_holdout,
        "N_sensitivity": sensitivity,
        "next_action": (
            "Run PWC-V3-03 conference-freeze audit and stop before journal innovation."
            if decision == "GREEN"
            else "Stop for normal-chat judgment before any method redesign."
        ),
    }
    (out / "closure_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "decision_state": decision,
        "final_beta": final_beta,
        "final_holdout_gap_pp_mean": final_holdout["aggregate"]["gap_pp_mean"],
        "final_holdout_gap_pp_std": final_holdout["aggregate"]["gap_pp_std"],
        "positive_gap_seeds": final_holdout["positive_gap_seeds"],
        "refinement_used": refinement is not None,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
