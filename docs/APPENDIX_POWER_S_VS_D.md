# Appendix TODO — Why POWER-D only slightly exceeds POWER-S

This question is intentionally deferred from the conference-baseline freeze.

## Current observation

Under the frozen continuous-overlap reconstruction:

- POWER-S: **78.776%**
- POWER-D: **78.796%**
- POWER-D minus POWER-S: **+0.020 pp**

The manuscript reference reports POWER-S 81.15% and POWER-D 82.50%.

## Why the near-tie is not automatically an error

Dynamic edge weights only improve route quality when temporal changes are large enough to alter relative path costs and therefore the selected shortest path. A strong static model may already capture most stable spatial preference structure. The finite time-decay state also smooths POWER-D across neighboring slots.

## Follow-up appendix analyses

1. **Weight similarity**
   - POWER-S vs hourly POWER-D Pearson/Spearman correlation.
   - Distribution of hourly rank changes on edges.
2. **Temporal variability**
   - Per-edge coefficient of variation across 168 POWER-D slots.
   - Identify edges / time periods with genuinely strong dynamics.
3. **Route-level complementarity**
   - Fraction of orders where POWER-D wins, POWER-S wins, or both tie.
   - Mean absolute per-order overlap difference.
4. **Time-decay effect**
   - Compare raw hourly POWER-D with the L=3 decayed POWER-D used by the routing state.
   - Quantify how much temporal smoothing suppresses dynamic path changes.
5. **Data provenance**
   - Verify POWER-S and POWER-D came from matched preprocessing/model versions.
   - Verify hourly alignment and test-week compatibility.
6. **Journal implication**
   - If advantages are state-dependent rather than globally dominant, use that complementarity as motivation for context-aware online PWE selection rather than assuming Dynamic should always dominate Static.

This analysis is appendix / mechanism evidence only. It must not be used to retune the frozen conference baseline.
