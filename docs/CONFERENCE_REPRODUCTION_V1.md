# PWC Conference Reconstruction V1

## Status

Frozen on 2026-09-27 after governed V3 multi-seed development and independent holdout.

## Reconstructed experimental semantics

The recovered evidence is most consistent with the following implementation:

1. The online reward is **mean physical road-length overlap**, not the q-threshold binary value described in the manuscript text.
2. PWC is strictly online: slot-t ground truth is incorporated only after selecting the slot-t target model.
3. The chasing state contains the previous three slots plus the current slot.
4. Heterogeneous PWE outputs are brought to a common positive scale before cross-model chasing. Positive per-model scaling does not change standalone shortest paths.
5. FPL uses a relatively small exponential perturbation. The public historical source contains `beta=5.348`; robust multi-seed reconstruction selected `beta=3.0`.

## Frozen main result

| Method | Current reconstruction |
| --- | ---: |
| Length | 45.844% |
| ConSTGAT | 74.002% |
| POWER-S | 78.776% |
| POWER-D | 78.796% |
| PWC holdout | **78.892%** |

PWC holdout minus best fixed PWE: **+0.096 pp**, standard deviation of the gap **0.153 pp**, non-negative on 3/5 holdout seeds.

## Order-count sensitivity

| N | PWC | Best fixed | Gap |
| ---: | ---: | ---: | ---: |
| 10 | 78.358% | 78.397% | -0.039 pp |
| 15 | 78.447% | 78.402% | +0.045 pp |
| 20 | 78.877% | 78.796% | +0.081 pp |

The reconstructed trend is consistent with PWC becoming more competitive as more per-slot feedback is available.

## Known limitations

- The available recovered ConSTGAT bundle is static in the present archive, whereas the manuscript describes time-varying ConSTGAT weights.
- The recovered road graph is not byte-identical to the graph statistics reported in the manuscript.
- The manuscript describes q-threshold binary accuracy, while recovered numerical behavior strongly supports continuous overlap as the historical implementation metric.
- POWER-S and POWER-D are nearly tied in the current reconstruction. This does not block the conference baseline freeze; the mechanism is deferred to an appendix diagnostic.

## Data release policy

The repository must not contain recovered Beijing or Qingdao data. Only the historical Chengdu sample is kept in Git.
