# Preference Weight Chasing (PWC)

Official research code for the KDD 2024 paper:

**Online Preference Weight Estimation Algorithm with Vanishing Regret for Car-Hailing in Road Network**

This repository contains a cleaned implementation of PWC, evaluation utilities, reproducibility scripts, tests, and a lightweight Chengdu example.

## Citation

If you find this work useful, please cite:

> **Yucen Gao**, Zhehao Zhu, Mingqian Ma, Fei Gao, Hui Gao, Yangguang Shi, Xiaofeng Gao.  
> *Online Preference Weight Estimation Algorithm with Vanishing Regret for Car-Hailing in Road Network.*  
> ACM Knowledge Discovery and Data Mining (KDD), 2024.

```bibtex
@inproceedings{gao2024pwc,
  author    = {Yucen Gao and Zhehao Zhu and Mingqian Ma and Fei Gao and Hui Gao and Yangguang Shi and Xiaofeng Gao},
  title     = {Online Preference Weight Estimation Algorithm with Vanishing Regret for Car-Hailing in Road Network},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD)},
  pages     = {863--871},
  year      = {2024}
}
```

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

The default online reward is `continuous-overlap`: the mean physical road-length overlap between the predicted and observed routes in each slot. This is the frozen conference-reproduction semantics.

To reproduce the q-threshold binary route-accuracy definition described in the manuscript text, request it explicitly:

```bash
python scripts/run_experiment.py \
  --city-dir /path/to/processed/city \
  --output /path/to/results/binary \
  --orders 20 --T 168 --L 3 --seed 0 \
  --weight-calibration median \
  --reward-metric binary-threshold --q 0.95
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

The historical Chengdu example remains under `examples/chengdu/`. A separate public **Porto Medium v0.2 student benchmark** is available as a versioned Release download.

## Porto Medium v0.2 student package

The public Porto benchmark includes processed trajectories, a frozen OSM road network, train-derived candidate edge costs, reference rewards, locked environments, and reproducibility scripts. It supports `T=168`, `N=5`, `L=3` on a continuous test week. No GPU or private city data are required.

```bash
python -m pip install -r requirements-porto-eval.lock.txt
python scripts/public_benchmark/download_porto_medium.py
python scripts/public_benchmark/student_benchmark.py verify
python scripts/public_benchmark/student_benchmark.py replay
python scripts/public_benchmark/student_benchmark.py smoke
```

Use Python 3.11 or 3.12. Start with the [student guide](docs/PORTO_MEDIUM_STUDENT_GUIDE.md) for environment setup, full routing reproduction, expected results, file schemas, and research boundaries. Download the [versioned student release](https://github.com/GaoYucen/PWC/releases/tag/porto-medium-v0.2) for the data and an offline kit containing code and data together.

Porto uses Length, TravelTime-Static, Preference-Static and Preference-Dynamic **proxy costs**, not the original learned ConSTGAT/POWER models. The strongest tested online selector still trails the best fixed expert on this week; the reference results retain that negative finding. Cached reward replay validates selectors, while the complete command in the guide reruns routing and chasing.

This test week has already been inspected. New method development requires a fresh unseen evaluation period for confirmatory claims. Preserve the [UCI and OpenStreetMap data attribution](docs/PORTO_DATA_ATTRIBUTION.md) when redistributing the data.

## Notes on the reproduced evaluation

The recovered implementation behavior is most consistent with continuous physical road-length overlap as the online reward. This differs from the thresholded route-accuracy description in the manuscript text.

The generic `scripts/run_experiment.py` entry point now defaults to the frozen continuous-overlap protocol. The manuscript-described q-threshold metric remains available only through the explicit `--reward-metric binary-threshold` compatibility option.

POWER-S and POWER-D are nearly tied under the reproduced setting; additional analysis is documented in `docs/APPENDIX_POWER_S_VS_D.md`.
