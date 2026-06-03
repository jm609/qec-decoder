# D3/D5 Paired Statistics

This note addresses the evaluation-report request to check whether the final
writeup overstates the d3/d5 seed-expanded result. It uses the completed
seed `0..7` selected deltas and adds exact paired seed-level tests.

Source artifact:

- `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json`

Builder:

- `tools/build_d3_d5_paired_statistics_summary.py`

Input artifact:

- `artifacts/eval/nn/sedp_d3_d5_seed0_7_bootstrap_ci_summary.json`

## Paired Test Summary

All values are held-out `stage_c_corr` selected deltas over raw no-edit
PyMatching. Zero deltas are raw no-edit fallback seeds and are excluded from
the sign/sign-flip tests, while still counted in the mean and bootstrap CI.

| distance | seed classes | mean selected delta | 95% bootstrap CI | exact sign p, one-sided | exact sign-flip mean p, one-sided | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| d3 | `8/0/0` | `+0.006591797` | `[+0.004516602, +0.008544922]` | `0.003906250` | `0.003906250` | uniformly positive over the checked seeds |
| d5 | `2/6/0` | `+0.005615234` | `[+0.000000000, +0.013671875]` | `0.250000000` | `0.250000000` | positive mean and non-harmful selected mode, but not a strong seed-level significance claim |

Two-sided exact p-values are d3 `0.007812500` and d5 `0.500000000`.

## D5 Seed-Level Fallback Table

This table is the key reason d5 must be described conservatively. Selected
mode gains come from two adopted local-selector seeds, while six seeds use raw
no-edit fallback. Candidate-only seeds `4` and `6` would be harmful, and the
selected policy blocks both.

| seed | selected mode | selected delta | candidate delta | adoption reason |
| ---: | --- | ---: | ---: | --- |
| 0 | `raw_no_edit` | `+0.000000000` | `+0.000000000` | `default_no_edit` |
| 1 | `raw_no_edit` | `+0.000000000` | `+0.000000000` | `default_no_edit` |
| 2 | `local_motif_selector` | `+0.021484375` | `+0.021484375` | `candidate_tie_with_high_margin_evidence` |
| 3 | `local_motif_selector` | `+0.023437500` | `+0.023437500` | `candidate_positive_delta_with_margin` |
| 4 | `raw_no_edit` | `+0.000000000` | `-0.007812500` | `candidate_positive_delta_harm_guard` |
| 5 | `raw_no_edit` | `+0.000000000` | `+0.000000000` | `default_no_edit` |
| 6 | `raw_no_edit` | `+0.000000000` | `-0.011718750` | `default_no_edit` |
| 7 | `raw_no_edit` | `+0.000000000` | `+0.000000000` | `default_no_edit` |

## Thesis Use

Use this exact phrasing for d5:

> d5 shows a positive selected-mode mean and no harmful selected seed over
> seed `0..7`, but the bootstrap CI touches zero and the exact paired tests are
> not strong because six seeds fall back to raw no-edit. Therefore d5 should be
> reported as conservative selected-mode recovery, not as a uniformly positive
> result.

Use this exact phrasing for d3:

> d3 is uniformly positive over seed `0..7`, with bootstrap CI above zero and
> exact one-sided sign/sign-flip p-value `0.003906250`.
