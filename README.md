# Preference Weight Chasing (PWC)

Official research code for the KDD 2024 paper:

**Online Preference Weight Estimation Algorithm with Vanishing Regret for Car-Hailing in Road Network**

This repository contains a cleaned implementation of PWC, evaluation utilities, reproducibility scripts, tests, and a lightweight Chengdu example.

## Method overview

PWC addresses online preference-weight selection for route planning when multiple preference-weight estimators (PWEs) are available. At each time slot, an online selector chooses a target PWE, and a finite-history chasing mechanism combines recent preference weights for routing.

The maintained implementation includes four PWE baselines:

- Length
- ConSTGAT
- POWER-S
- POWER-D

The implementation is strictly online: ground-truth routes from the current time slot are incorporated only after the slot-level model choice is made.

## Reproduced experimental setting

The main reproduced setting uses:

- horizon: `T=168`
- orders per slot: `N=20`
- history length: previous `L=3` slots plus the current slot
- evaluation reward: mean physical road-length overlap
- cross-model weight normalization: positive median scaling
- FPL selector with exponential perturbation
- multi-seed development and independent holdout evaluation

Representative results:

| Method | Mean overlap |
| --- | ---: |
| ConSTGAT | 74.002% |
| POWER-S | 78.776% |
| POWER-D | 78.796% |
| PWC | **78.892%** |

The source-code parameter `beta=5.348` is retained as a historical reference. Multi-seed reconstruction selected `beta=3.0` as a stable setting under the reproduced protocol.

More details are available in:

- `configs/conference_freeze_v1.json`
- `docs/CONFERENCE_REPRODUCTION_V1.md`
- `reports/CONFERENCE_BASELINE_V1.md`

## Repository layout

```text
pwc/                         Core PWC implementation
scripts/                     Data preparation and experiment scripts
configs/                     Reproducibility configuration
docs/                        Experimental and implementation notes
reports/                     Compact result summaries
tests/                       Unit tests
examples/chengdu/            Lightweight Chengdu example
```

## Installation

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Running an experiment

Prepare a processed city dataset outside the repository, then run:

```bash
python scripts/run_experiment.py \
  --city-dir /path/to/processed/city \
  --output /path/to/results/run \
  --orders 20 --T 168 --L 3 --seed 0 \
  --weight-calibration median
```

For multi-seed selector evaluation, see:

- `scripts/tune_continuous_selector.py`
- `scripts/v3_holdout_refine.py`

## Chengdu example

A small Chengdu example is included for lightweight inspection:

```bash
cd examples/chengdu
python PWCA.py
```

The Chengdu example is intentionally small and is separate from the full experimental datasets used in the paper.

## Data

Large research datasets are not distributed in this repository. In particular, no Beijing or Qingdao experiment data are tracked in Git.

Only the lightweight Chengdu example under `examples/chengdu/` is included.

## Notes on the reproduced evaluation

The recovered implementation behavior is most consistent with continuous physical road-length overlap as the online reward. This differs from the thresholded route-accuracy description in the manuscript text. The repository records this implementation detail explicitly for reproducibility.

POWER-S and POWER-D are nearly tied under the reproduced setting; additional analysis is documented in `docs/APPENDIX_POWER_S_VS_D.md`.
