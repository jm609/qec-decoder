# Hyperparameter Sensitivity

This note addresses the evaluation-report request for a small hyperparameter
sensitivity table. It uses already-generated artifacts and does not introduce
new training runs.

Scope:

- distance: d7
- sentinel seeds: `0, 2, 5`
- varied hyperparameter: `selector_identity_margin_loss_weight`
- fixed companion setting: diagnostic selector epoch selection with
  candidate-first safety

Source artifacts:

- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin025_diagselect_selection_compare_seed0_2_5.json`
- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_2_5.json`
- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin10_diagselect_selection_compare_seed0_2_5.json`
- `artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json`

Builder:

- `tools/build_hyperparameter_sensitivity_summary.py`

## Sensitivity Table

All deltas are held-out `stage_c_corr` accuracy deltas over raw no-edit
PyMatching. The seed set is deliberately small because it is a sentinel check:
seed `2` is a true-positive recovery case, while seeds `0` and `5` expose
over-adoption and fallback behavior.

| identity-margin weight | selected modes | mean selected delta | seed0 selected | seed2 selected | seed5 selected | selected positive/neutral/harmful | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0.25` | local `2`, raw `1` | `+0.000651042` | `-0.002929688` | `+0.004882812` | `+0.000000000` | `1/1/1` | too weak: admits a harmful selected seed0 branch |
| `0.50` | local `2`, raw `1` | `+0.002278646` | `+0.001953125` | `+0.004882812` | `+0.000000000` | `2/1/0` | best sentinel balance among checked values |
| `1.00` | raw `3` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `0/3/0` | too conservative: suppresses true-positive seed2 recovery |

## Interpretation

The sensitivity is not monotonic in a useful way:

- `0.25` is too weak. It allows a harmful selected branch on seed `0`.
- `0.50` is the best local compromise on the sentinel set. It preserves seed
  `2`, recovers seed `0`, and keeps seed `5` at raw no-edit.
- `1.00` is too conservative. It blocks all selected recovery, including the
  known true-positive seed `2`.

This supports the active `0.5` identity-margin setting used in the d7 follow-up
experiments. It also reinforces the broader d7 conclusion: a reasonable local
hyperparameter choice can improve a small sentinel set, but it does not solve
the full d7 validation-to-held-out generalization problem by itself.

## Thesis Use

Use this table as a compact sensitivity paragraph, not as a claim of exhaustive
hyperparameter optimization. The defensible statement is:

> A small d7 sentinel sensitivity check shows that the identity-margin weight is
> behaviorally important: too low admits a harmful edit, too high suppresses true
> positives, and `0.5` is the best checked compromise. However, this local
> sensitivity does not remove the full d7 selector-ranking bottleneck.
