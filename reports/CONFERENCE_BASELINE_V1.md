# Conference Baseline V1

## Core comparison

| Method | Reconstructed score | Published N=20 reference |
| --- | ---: | ---: |
| Length | 45.844% | 13.60% |
| ConSTGAT | 74.002% | 70.65% |
| POWER-S | 78.776% | 81.15% |
| POWER-D | 78.796% | 82.50% |
| PWC | **78.892%** | **83.75%** |

Primary reconstructed metric: mean physical road-length overlap.

## Selector robustness

- robust beta selected from frozen candidate family: **3.0**
- development seeds: 0..4
- independent holdout seeds: 5..9
- holdout PWC minus best fixed: **+0.0959 pp**
- gap standard deviation: **0.1530 pp**
- non-negative gap seeds: **3/5**

Historical source anchor `beta=5.348` remains reproducible and nearly tied with best fixed:
5-seed mean gap **+0.0034 pp**.

## Frozen interpretation

This result is sufficient to freeze the conference reconstruction.
Journal-extension work should start from this baseline and target a materially larger,
stable improvement rather than continue conference parameter tuning.
