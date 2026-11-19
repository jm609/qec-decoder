# D7 Harmful-Edit Taxonomy

This document records the d7 harmful-candidate taxonomy after the d3/d5
positive selected-mode result was fixed. It is an analysis artifact, not a new
feature branch.

Source artifacts:

- `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`

Builder:

- `tools/build_d7_harmful_edit_taxonomy_summary.py`

## Purpose

The d7 result has high oracle headroom but almost no selected-mode gain:

| metric | value |
| --- | ---: |
| checked seeds | `58` |
| mean selected delta | `+0.000151536` |
| mean actual candidate delta | `-0.001515356` |
| mean candidate-oracle delta | `+0.096679688` |
| candidate outcomes | positive `6`, neutral `35`, harmful `17` |
| oracle-positive seeds | `58` |

The taxonomy asks what type of harmful candidate behavior appears when the
selected-mode guard correctly falls back to raw no-edit.

## Main Finding

The dominant d7 failure pattern is not lack of candidates. It is
validation-to-held-out false-positive selector behavior.

Among validation-positive candidate seeds:

| validation-positive outcome on held-out | count |
| --- | ---: |
| held-out harmful | `13` |
| held-out neutral | `4` |
| held-out positive | `5` |

Thus, in the current d7 recipe, a validation-positive candidate branch is more
often harmful than beneficial on held-out `stage_c_corr`. The false-positive
ratio among validation-positive seeds is `13/22 = 59.09%`.

## Validation-To-Held-Out Crosstab

| validation candidate class | held-out harmful | held-out neutral | held-out positive |
| --- | ---: | ---: | ---: |
| neutral | `4` | `31` | `1` |
| positive | `13` | `4` | `5` |

This is the central evidence for treating d7 as a selector-ranking and
generalization limitation rather than a scalar adoption-threshold problem.

## Harmful Candidate Summary

All `17` harmful candidate seeds were blocked by selected mode, so the final
selected path remained `raw_no_edit` for those seeds.

| metric over harmful candidate seeds | value |
| --- | ---: |
| count | `17` |
| seeds | `5, 8, 9, 13, 17, 18, 26, 32, 33, 34, 36, 38, 41, 53, 54, 55, 56` |
| mean held-out candidate delta | `-0.005974265` |
| mean validation delta | `+0.004203279` |
| mean candidate-oracle delta | `+0.104549632` |
| mean candidate-to-oracle gap | `+0.110523897` |
| mean improved/harmed shots | `6.88 / 13.00` |
| mean selected-edit fraction | `0.019703585` |
| selected mode | `raw_no_edit` for `17/17` |

## Taxonomy

| category | count | seeds | mean held-out candidate delta | mean validation delta | mean oracle delta | interpretation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| validation false-positive harmful | `13` | `5, 8, 13, 17, 18, 26, 32, 33, 38, 41, 53, 54, 55` | `-0.007286659` | `+0.005496596` | `+0.101036659` | main d7 harmful type |
| validation nonpositive harmful | `4` | `9, 34, 36, 56` | `-0.001708984` | `+0.000000000` | `+0.115966797` | easier to block because validation is not positive |
| high-oracle harmful | `9` | `5, 8, 9, 13, 32, 34, 36, 54, 55` | `-0.005750868` | `+0.003609635` | `+0.126736111` | candidate coverage exists but ranking picks bad edits |
| broad over-edit harmful | `9` | `5, 8, 17, 18, 26, 32, 33, 53, 55` | `-0.009331597` | `+0.005410464` | `+0.100043403` | many edited shots; mean improved/harmed `11.00 / 20.56` |
| sparse harmful | `5` | `9, 34, 36, 38, 56` | `-0.001757812` | `+0.000649911` | `+0.108789062` | rare edits, but still net harmful |
| severe harmful | `5` | `8, 26, 32, 33, 53` | `-0.013867188` | `+0.006488792` | `+0.097460938` | largest held-out candidate losses |

## Top Harmful Seeds

| seed | held-out candidate delta | validation delta | improved/harmed | oracle delta | candidate-to-oracle gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `-0.019531250` | `+0.006481003` | `7 / 27` | `+0.125976562` | `+0.145507812` |
| `33` | `-0.016601562` | `+0.009738103` | `21 / 38` | `+0.080078125` | `+0.096679688` |
| `26` | `-0.011718750` | `+0.003243972` | `11 / 23` | `+0.080078125` | `+0.091796875` |
| `32` | `-0.010742188` | `+0.006487965` | `16 / 27` | `+0.125976562` | `+0.136718750` |
| `53` | `-0.010742188` | `+0.006492915` | `9 / 20` | `+0.075195312` | `+0.085937500` |
| `54` | `-0.006835938` | `+0.006508300` | `6 / 13` | `+0.126953125` | `+0.133789062` |
| `17` | `-0.004882812` | `+0.009746037` | `8 / 13` | `+0.080078125` | `+0.084960938` |
| `55` | `-0.004882812` | `+0.003244387` | `8 / 13` | `+0.126953125` | `+0.131835938` |

## Sentinel Implication

The recommended false-positive block set is:

```text
8, 13, 17, 18, 32, 33, 53, 54
```

Any future d7 objective should be tested first on the existing
preserve/recover/block gate:

| role | seeds | required behavior |
| --- | --- | --- |
| preserve true positives | `2, 11` | keep selected/candidate gains |
| recover missed positives | at least one of `0, 28, 43, 45` | improve candidate ranking without opening false positives |
| block false positives | `8, 13, 17, 18, 32, 33, 53, 54` | keep harmful candidate behavior blocked |

The next d7 idea must distinguish validation-positive false positives from
true positives. Broad threshold tuning, simple support guards, and top-k
compatibility sweeps have already failed that standard.

## Thesis Use

Use this document in the d7 limitation chapter to make the negative result
constructive:

- d7 is not being dismissed because it failed once.
- The candidate oracle shows recoverable local-edit headroom.
- The harmful taxonomy shows why the current learned selector cannot safely
  use that headroom.
- The selected-mode guard is doing useful safety work by blocking `17/17`
  harmful candidate branches.
