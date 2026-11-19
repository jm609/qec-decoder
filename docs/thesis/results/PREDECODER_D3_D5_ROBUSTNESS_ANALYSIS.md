# D3/D5 Robustness Analysis

This document strengthens the successful d3/d5 result before further d7 work.
It uses the completed seed `0..7` runs and separates held-out
performance from validation-family behavior.

Source artifact:

- `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json`
- `artifacts/eval/nn/sedp_d3_d5_seed0_7_bootstrap_ci_summary.json`
- `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json`

Builder:

- `tools/build_d3_d5_robustness_summary.py`
- `tools/build_d3_d5_paired_statistics_summary.py`

## Held-Out Stage-C Robustness

All rows use held-out `stage_c_corr` accuracy delta over raw no-edit
PyMatching.

| distance | seeds | local selector seeds | raw no-edit seeds | positive/neutral/harmful seeds | mean selected delta | min selected delta | selected improved/harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d3 | `0..7` | `8` | `0` | `8/0/0` | `+0.006591797` | `+0.001953125` | `205/151` |
| d5 | `0..7` | `2` | `6` | `2/6/0` | `+0.005615234` | `+0.000000000` | `56/10` |

Interpretation:

- d3 is a uniformly positive selected-mode result across the eight checked
  training seeds.
- d5 is not uniformly local-edit emitting, but the selected-mode safety policy
  is doing useful work: weak seeds fall back to raw no-edit, and adopted seeds
  have a strong improved/harmed ratio.
- Neither d3 nor d5 has a harmful selected seed in the current seed `0..7`
  evidence.

## Bootstrap Confidence Intervals

The CI is computed by bootstrapping the seed-level selected delta distribution
with `20,000` resamples.

| distance | selected delta mean | 95% bootstrap CI | selected seed classes | candidate delta mean | candidate seed classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `+0.006591797` | `[+0.004516602, +0.008544922]` | `8/0/0` | `+0.006591797` | `8/0/0` |
| d5 | `+0.005615234` | `[+0.000000000, +0.013671875]` | `2/6/0` | `+0.003173828` | `2/4/2` |

Interpretation:

- d3 remains clearly positive across the expanded seed set.
- d5 should be stated more conservatively: selected mode is non-harmful over
  seeds `0..7`, but its CI touches zero because six seeds fall back to raw
  no-edit.
- The d5 candidate branch is not safe by itself: seed `4` and seed `6` are
  harmful candidates, and selected-mode adoption blocks both by choosing raw
  no-edit.

## Exact Paired Seed-Level Tests

These tests use the same seed-level selected deltas as the bootstrap CI. Zero
fallback seeds are excluded from the sign/sign-flip tests but remain included
in the reported mean and CI.

| distance | nonzero selected seeds | exact sign p, one-sided | exact sign-flip mean p, one-sided | two-sided sign-flip p | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| d3 | `8` | `0.003906250` | `0.003906250` | `0.007812500` | all eight seeds are positive |
| d5 | `2` | `0.250000000` | `0.250000000` | `0.500000000` | two positive local-selector seeds and six zero fallback seeds |

This reinforces the conservative wording: d3 can be described as uniformly
positive over the checked seeds, while d5 should be described as selected-mode
safety with positive mean gain and no harmful selected seed, not as a strong
seed-level significance result.

## Accuracy Summary

| distance | raw PyMatching | selected predecoder | candidate branch | selected delta |
| --- | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.935302734` | `0.935302734` | `+0.006591797` |
| d5 | `0.888671875` | `0.894287109` | `0.891845703` | `+0.005615234` |

The selected branch and candidate branch are identical for d3. For d5, selected
mode is better than the unconditional candidate branch because it falls back to
raw no-edit on harmful candidate seeds.

## Validation-Family Behavior

The training/validation families are `stage_a_si1000` and `stage_b_local`.
These are not the final held-out claim, but they show how stable the selected
decision was before evaluating `stage_c_corr`.

| distance | validation family | mean selected delta | positive/neutral/harmful seeds |
| --- | --- | ---: | ---: |
| d3 | `stage_a_si1000` | `+0.021103896` | `7/1/0` |
| d3 | `stage_b_local` | `+0.042207792` | `8/0/0` |
| d5 | `stage_a_si1000` | `+0.003246753` | `1/6/1` |
| d5 | `stage_b_local` | `-0.000000000` | `1/6/1` |

Interpretation:

- d3 has positive validation-family evidence and positive held-out behavior.
- d5 validation-family evidence is mixed at the family level, but selected
  held-out behavior remains non-harmful because the adoption policy falls back
  to raw no-edit on weak seeds.
- This supports the claim that selected-mode adoption, not unconditional local
  editing, is part of the successful d3/d5 method.

## Seed-Level Held-Out Rows

| distance | seed | selected mode | held-out delta | improved/harmed |
| --- | ---: | --- | ---: | ---: |
| d3 | 0 | `local_motif_selector` | `+0.010742188` | `31/20` |
| d3 | 1 | `local_motif_selector` | `+0.006835938` | `20/13` |
| d3 | 2 | `local_motif_selector` | `+0.008789062` | `29/20` |
| d3 | 3 | `local_motif_selector` | `+0.003906250` | `24/20` |
| d3 | 4 | `local_motif_selector` | `+0.001953125` | `29/27` |
| d3 | 5 | `local_motif_selector` | `+0.003906250` | `23/19` |
| d3 | 6 | `local_motif_selector` | `+0.009765625` | `30/20` |
| d3 | 7 | `local_motif_selector` | `+0.006835938` | `19/12` |
| d5 | 0 | `raw_no_edit` | `+0.000000000` | `0/0` |
| d5 | 1 | `raw_no_edit` | `+0.000000000` | `0/0` |
| d5 | 2 | `local_motif_selector` | `+0.021484375` | `28/6` |
| d5 | 3 | `local_motif_selector` | `+0.023437500` | `28/4` |
| d5 | 4 | `raw_no_edit` | `+0.000000000` | `0/0` |
| d5 | 5 | `raw_no_edit` | `+0.000000000` | `0/0` |
| d5 | 6 | `raw_no_edit` | `+0.000000000` | `0/0` |
| d5 | 7 | `raw_no_edit` | `+0.000000000` | `0/0` |

## Research Decision

The expanded d3/d5 evidence is strong enough to support a conservative positive
claim in a graduation thesis:

> The proposed transition-aware neural predecoder improves raw PyMatching on
> d3 and d5 under selected-mode candidate-first safety.

The claim should remain conservative. d3 has a positive CI over the checked
seeds. d5 is best described as selected-mode safety with positive mean gain,
because the CI touches zero and candidate-only behavior can be harmful.
