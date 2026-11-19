# D7 Targeted Selector-Bottleneck Analysis

This note records the first d7 follow-up after freezing the d3/d5 successful
structure. It does not add a new model feature. It narrows the next d7 work to
sentinel seed groups that any new objective must preserve, recover, or block.

Source artifact:

- `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`
- `artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json`

Input artifacts:

- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json`
- `artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json`

Tool:

- `tools/analyze_d7_selector_bottleneck.py`
- `tools/build_d7_harmful_edit_taxonomy_summary.py`
- `tools/build_d7_validation_heldout_scatter.py`
- `tools/search_d7_sentinel_adoption_grid.py`

## Current D7 State

| metric | value |
| --- | ---: |
| checked seeds | `58` |
| mean selected delta | `+0.000151536` |
| mean candidate delta | `-0.001515356` |
| mean candidate-oracle delta | `+0.096679688` |
| selected positive seeds | `2, 11` |
| missed candidate-positive seeds | `0, 28, 43, 45` |
| harmful candidate seeds blocked by selected mode | `17` |
| neutral candidate seeds with oracle headroom | `35` |

Interpretation:

- The current support guard is selected-safe but mostly no-edit.
- Candidate coverage is not the main limitation because oracle headroom is
  positive across all checked seeds.
- The blocker is candidate ranking and stage generalization.

## Validation-To-Holdout Mismatch

The key d7 problem is that validation-positive candidate evidence does not
reliably transfer to held-out `stage_c_corr`.

| validation candidate class | held-out harmful | held-out neutral | held-out positive |
| --- | ---: | ---: | ---: |
| neutral | `4` | `31` | `1` |
| positive | `13` | `4` | `5` |

Interpretation:

- A positive validation delta is more often harmful than positive on held-out
  candidate behavior in the current d7 recipe.
- The seed-level scatter artifact
  `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg`
  visualizes this mismatch directly. The Pearson correlation between validation
  delta and held-out candidate delta is `-0.452872`, so the current validation
  signal is not a reliable positive ranking signal at d7.
- This rejects another scalar adoption-threshold sweep as the next rational
  step.
- A useful d7 improvement must change the selector ranking signal, not merely
  loosen adoption.

## Harmful-Edit Taxonomy

The taxonomy artifact is:

- `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`

The harmful candidate branch appears in `17` seeds:

```text
5, 8, 9, 13, 17, 18, 26, 32, 33, 34, 36, 38, 41, 53, 54, 55, 56
```

All `17/17` are blocked by selected mode and fall back to `raw_no_edit`.

The main harmful type is validation-positive false positive:

| category | count | mean held-out candidate delta | mean validation delta | mean oracle delta |
| --- | ---: | ---: | ---: | ---: |
| validation false-positive harmful | `13` | `-0.007286659` | `+0.005496596` | `+0.101036659` |
| validation nonpositive harmful | `4` | `-0.001708984` | `+0.000000000` | `+0.115966797` |
| severe harmful | `5` | `-0.013867188` | `+0.006488792` | `+0.097460938` |

Among validation-positive candidate seeds, held-out outcomes are harmful `13`,
neutral `4`, and positive `5`, so the false-positive ratio is `59.09%`.
This is why d7 should not be treated as a simple calibration problem.

## Preserve/Recover/Block Sentinel Sets

Any next d7 training objective should first be tested on these groups.

| role | seeds | reason |
| --- | --- | --- |
| preserve true positives | `2, 11` | already selected-positive under support guard |
| recover missed positives | `0, 28, 43, 45` | candidate branch is held-out positive but selected mode falls back to raw no-edit |
| block validation false positives | `13, 17, 33, 54, 53, 32, 8, 18` | validation candidate evidence is positive, but held-out candidate is harmful |
| inspect high-oracle misses | `3, 5, 9, 13, 34, 36, 47, 49` | candidate-oracle delta is at least `0.12`, but actual candidate delta is non-positive |

The most important false-positive sentinels remain:

| seed | validation delta | held-out candidate delta | improved/harmed | note |
| ---: | ---: | ---: | ---: | --- |
| 13 | `+0.009755672` | `-0.000976562` | `5/6` | validation-clean but held-out harmful |
| 17 | `+0.009746037` | `-0.004882812` | `8/13` | plateau guard blocks selected adoption |
| 33 | `+0.009738103` | `-0.016601562` | `21/38` | large harmful candidate branch |
| 54 | `+0.006508300` | `-0.006835938` | `6/13` | known seed54 false-positive gate |
| 8 | `+0.006481003` | `-0.019531250` | `7/27` | largest harmful candidate delta |

## Next D7 Rule

Do not run another broad d7 seed extension or scalar adoption policy sweep.

The adoption-grid diagnostic confirms this is not just a conservative
preference:

| metric | value |
| --- | ---: |
| artifact | `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json` |
| simple policies checked | `183040` |
| policies passing preserve/recover/block gate | `0` |
| best recover set | `0, 28, 43, 45` |
| best false-positive failures | `13, 17, 18, 54` |
| best mean selected delta | `-0.000454607` |

Interpretation:

- If a simple threshold policy recovers all currently missed-positive seeds, it
  also opens harmful false-positive seeds.
- The strongest observed simple-policy recovery still fails the block gate and
  has negative mean selected delta.
- Therefore the next d7 work must change candidate ranking/generalization, not
  just selected-mode adoption thresholds.

## Rejected Candidate-Compatibility Top-K

Candidate-compatibility pairwise top-k was tested as the first existing
ranking/gating path after the adoption-grid rejection.

Artifacts:

- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed54_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_pairwise_seq`
- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_pairwise_seq`
- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`

| seed | sentinel role | selected delta | candidate delta | improved/harmed | verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| 54 | block false positive | `+0.000000000` | `+0.000000000` | `0/0` | blocked |
| 2 | preserve true positive | `+0.000000000` | `-0.136718750` | `56/196` | destroyed |

Verdict:

- This recipe blocks the seed54 false-positive case.
- It destroys the seed2 true-positive candidate branch.
- Therefore it should not be expanded to seed11 or missed-positive recovery
  seeds.

The next d7 experiment must satisfy this local gate first:

1. Preserve selected-positive seeds `2` and `11`.
2. Improve candidate ranking on at least one missed-positive seed among
   `0, 28, 43, 45`.
3. Keep false-positive sentinels, especially `8, 13, 17, 33, 54`, blocked.
4. Report candidate and selected deltas separately.

If a candidate objective cannot pass that sentinel gate, it should be rejected
without extending to the full `0..57` seed set.
