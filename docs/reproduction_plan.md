# PWC automated reconstruction and validation plan

## Goal
Recover a conference-faithful implementation that reproduces the qualitative and numerical behavior of KDD'24 PWC without overfitting a single run.

## Frozen semantics
- Dataset: recovered Didi Beijing 2023-04-16..2023-04-22.
- Orders: topology-valid orders first, then deterministic reservoir sampling (seed 2024).
- Slot length: 1 hour, T=168.
- Chasing memory: previous L=3 slots plus current slot.
- Reward/accuracy hypothesis: mean physical road-length overlap.
- PWE models: Length, ConSTGAT, POWER-S, POWER-D.
- Chasing unit alignment: per-model positive median scaling. This cannot change any standalone shortest path.
- No current-slot ground-truth leakage.

## Stage A — selector-rate reconstruction
Use the source-code clue beta=5.348 as the anchor and sweep a small interpretable family of exponential perturbation rates:
1, 2, 3, 5.348, 8, 12, 20, 100.

Run seeds 0 and 1 for every candidate. Keep the top four candidates by mean PWC-minus-best-fixed overlap, always retaining 5.348.

## Stage B — robust selector validation
For the shortlisted rates, add seeds 2, 3, 4. Select the robust rate by:
1. highest 5-seed mean PWC-minus-best-fixed overlap;
2. lower standard deviation as tie-breaker.

Report source beta=5.348 separately even if another rate wins.

## Stage C — N sensitivity
With the robust rate, run N=10,15,20 for seeds 0..4 and report mean/std:
- PWC overlap;
- best fixed overlap;
- PWC minus best fixed;
- switch rate;
- selector-direct overlap;
- chasing gain.

The expected conference behavior is that PWC becomes increasingly competitive as N grows.

## Runtime optimization
Baseline routes/rewards are independent of beta/seed. Precompute and cache all four baseline per-order overlaps once, then reuse them during selector sweeps. Each beta/seed thereafter only reruns the PWC mixed-weight route calculation.

## Acceptance criteria
Primary:
- PWC mean overlap >= best fixed mean overlap on Beijing N=20 for the source beta or a stable nearby rate.
- Advantage remains non-negative or near zero across multiple seeds.
- Average regret shrinks with T.

Secondary:
- ConSTGAT / POWER-S / POWER-D remain in the same approximate range as the original paper.
- N sensitivity follows the published qualitative trend.

Do not tune q or manipulate test orders to improve PWC.
