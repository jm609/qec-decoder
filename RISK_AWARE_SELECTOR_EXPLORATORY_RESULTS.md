# Risk-Aware Selector Exploratory Results

This document records the first isolated post-presentation selector experiments.
The runs are exploratory and must not replace the frozen thesis results.

## Implementation Summary

Implemented selector options in `decoders/syndrome_edit_predecoder.py`:

| selector model | output utility | intent |
| --- | --- | --- |
| `scalar` | one scalar selector logit | frozen baseline behavior |
| `risk_aware` | `benefit_logit - lambda * harm_logit` | directly separate beneficial and harmful candidates |
| `risk_guard` | `rank_logit - lambda * harm_logit` | preserve scalar ranking while adding a harm-risk penalty |

Both new models are enabled only through `--selector-model`; default behavior
remains `scalar`.

## Seed-2 Preservation Check

The first sentinel requirement is to preserve known d7 true-positive seeds.
Seed `2` was tested first because the frozen scalar selector selects the local
motif branch and improves held-out `stage_c_corr`.

Frozen scalar reference:

| run | selector model | candidate delta | selected delta | selected mode | adoption reason |
| --- | --- | ---: | ---: | --- | --- |
| frozen seed2 | scalar | `+0.004882812` | `+0.004882812` | `local_motif_selector` | `candidate_positive_delta_with_margin` |

### Initial Smoke Runs

The first exploratory runs were useful as smoke tests for the new selector
modules, but they were not a fully fair comparison to the frozen scalar recipe:
they omitted the scalar run's local-motif candidate settings
(`--selector-policy-candidate-mode none`,
`--selector-local-motif-max-classes 16`,
`--selector-local-motif-top-k 32`). As a result, the candidate row count was much
smaller than the scalar reference. Treat these as implementation checks only.

| run directory | selector model | harm utility weight | harm loss weight | candidate delta | selected delta | selected mode | adoption reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `risk_aware_selector_exploratory/d7_seed2_sentinel` | `risk_aware` | `1.00` | `1.00` | `+0.000000000` | `+0.000000000` | `raw_no_edit` | `default_no_edit` |
| `risk_aware_selector_exploratory/d7_seed2_harm025_sentinel` | `risk_aware` | `0.25` | `0.25` | `-0.000976562` | `+0.000000000` | `raw_no_edit` | `default_no_edit` |
| `risk_aware_selector_exploratory/d7_seed2_riskguard025_sentinel` | `risk_guard` | `0.25` | `1.00` | `-0.000976562` | `+0.000000000` | `raw_no_edit` | `default_no_edit` |
| `risk_aware_selector_exploratory/d7_seed2_riskguard010_sentinel` | `risk_guard` | `0.10` | `0.10` | `-0.000976562` | `+0.000000000` | `raw_no_edit` | `default_no_edit` |

### Fair Risk-Guard Rerun

A fair rerun was then made with the same local-motif candidate settings as the
frozen scalar seed-2 run:

| run directory | selector model | harm utility weight | harm loss weight | candidate delta | selected delta | selected mode | adoption reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `risk_aware_selector_exploratory/d7_seed2_riskguard010_fair_sentinel` | `risk_guard` | `0.10` | `0.10` | `+0.001953125` | `+0.000000000` | `raw_no_edit` | `candidate_positive_delta_support_guard` |

Compared with the frozen scalar seed-2 result:

| metric | frozen scalar | fair risk-guard |
| --- | ---: | ---: |
| held-out candidate delta | `+0.004882812` | `+0.001953125` |
| held-out selected delta | `+0.004882812` | `+0.000000000` |
| selected validation nonzero support | `5` | `2` |
| adopted margin | `1.25` | blocked at `1.5` |

The fair run preserves the candidate set size but still fails selected-mode
preservation. The candidate branch remains slightly positive, but it is weaker
than the scalar reference and the selected-mode safety policy falls back to
raw no-edit because validation support is too sparse.

Diagnostic artifact:

```text
artifacts/eval/nn/risk_aware_selector_exploratory/d7_seed2_selector_preservation_diagnostic_fair_harmguard.json
```

Key diagnostic rows for the fair risk-guard run:

| family/split | margin | selected nonzero | positive target | negative target | delta over raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage_b_local:val` | `1.25` | `5` | `4` | `1` | `+0.019481` |
| `stage_c_corr:full` | `1.25` | `12` | `8` | `4` | `+0.003906` |
| `stage_b_local:val` | `1.50` | `2` | `2` | `0` | `+0.012987` |
| `stage_c_corr:full` | `1.50` | `2` | `2` | `0` | `+0.001953` |

The `1.25` margin has more positive selected candidates than scalar, but it
also introduces more harmful candidates. The `1.50` margin is clean in this
seed-2 diagnostic, but only two validation edits survive, so it fails the
existing support guard.

## Harm-Guard Diagnostic

The fair risk-guard model has a separate harm logit. Across all candidate rows,
the harm head appears to separate harmful rows from positive rows:

| family/split | positive median harm logit | negative median harm logit | identity median harm logit |
| --- | ---: | ---: | ---: |
| `stage_a_si1000:val` | `-0.057839` | `8.741053` | `-0.117667` |
| `stage_b_local:val` | `-0.565914` | `8.776325` | `-0.127199` |
| `stage_c_corr:full` | `-0.354522` | `8.714895` | `-0.130385` |

However, using the harm logit as a simple hard threshold does not fix the
selected candidates. At margin `1.25`, a strict threshold of `-1.0` still leaves
harmful selected edits and removes useful edits:

| family/split | harm threshold | selected nonzero | positive target | negative target | delta over raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage_a_si1000:val` | `-1.0` | `1` | `0` | `1` | `-0.006494` |
| `stage_b_local:val` | `-1.0` | `4` | `3` | `1` | `+0.012987` |
| `stage_c_corr:full` | `-1.0` | `7` | `4` | `3` | `+0.000977` |

Interpretation: the harm head is informative in aggregate, but it is not
calibrated enough on the exact high-utility candidates that control adoption.
The problem is therefore still selector ranking/calibration, not merely the
absence of a harm output.

## Final Combined Hard-Ranking Check

As one final low-risk experiment, the fair `risk_guard` seed-2 setup was rerun
with the existing within-shot positive-vs-negative hard ranking loss enabled:

```text
--selector-model risk_guard
--selector-risk-aware-harm-logit-weight 0.1
--selector-risk-aware-harm-loss-weight 0.1
--selector-positive-negative-hard-loss-weight 0.5
--selector-positive-negative-hard-margin 0.5
```

Run directory:

```text
artifacts/eval/nn/risk_aware_selector_exploratory/d7_seed2_riskguard010_posneghard05_m05_fair_sentinel/
```

Diagnostic artifact:

```text
artifacts/eval/nn/risk_aware_selector_exploratory/d7_seed2_riskguard_posneghard05_preservation_diagnostic.json
```

Held-out `stage_c_corr` seed-2 comparison:

| run | candidate delta | selected delta | selected mode | adoption reason |
| --- | ---: | ---: | --- | --- |
| frozen scalar | `+0.004882812` | `+0.004882812` | `local_motif_selector` | `candidate_positive_delta_with_margin` |
| fair risk-guard | `+0.001953125` | `+0.000000000` | `raw_no_edit` | `candidate_positive_delta_support_guard` |
| risk-guard + posneg-hard `0.5/0.5` | `+0.002929688` | `+0.000000000` | `raw_no_edit` | `candidate_positive_delta_harm_guard` |

The combined variant improves the candidate branch relative to plain
`risk_guard`, but it still fails selected-mode preservation. Validation selects
too many harmful edits at the adopted margin:

| family/split | margin | selected nonzero | positive target | negative target |
| --- | ---: | ---: | ---: | ---: |
| `stage_a_si1000:val` | `1.0` | `1` | `0` | `1` |
| `stage_b_local:val` | `1.0` | `8` | `6` | `2` |
| combined validation | `1.0` | `9` | `6` | `3` |
| `stage_c_corr:full` | `1.0` | `19` | `11` | `8` |
| `stage_c_corr:full` | `1.25` | `1` | `0` | `1` |

The safety policy blocks adoption because validation harmed count is `3`, above
the configured cap of `2`. Raising the margin is not a solution: at margin
`1.25`, the positive selected support disappears and only one negative edit
survives on held-out `stage_c_corr`.

## Interpretation

The current risk-aware variants fail the first preserve-positive check. Even
after the fair rerun, they do not preserve the seed `2` selected-mode gain that
the frozen scalar selector already has.

The failure mode is a ranking/adoption tradeoff:

- at lower margin, selected support is available but harmful candidates remain
- at higher margin, selected candidates are clean but too sparse for the
  support guard
- the harm head separates row populations in aggregate, but does not provide a
  sufficient hard guard for the selected high-score candidates
- adding within-shot positive-vs-negative hard ranking to `risk_guard` improves
  candidate delta slightly, but still trips the selected-mode harm guard

This means the current risk-aware selector should not be expanded to the full
d7 sentinel set or to seeds `0..57`.

## Decision

Current decision: stop this exact risk-aware path unless a new objective changes
the preservation behavior on seed `2`.

Do not report these runs in the poster or main thesis results. They are useful
as internal evidence that a simple dual-head, risk-penalty, post-hoc
harm-threshold, or combined risk-guard plus same-shot hard-ranking selector is
not sufficient under the current training objective.

## Remaining Allowed Follow-Up

No further d7 expansion is justified from this path. A future follow-up would
need a genuinely different objective, not another small weight/margin variant:

- change the objective so the selected high-utility harmful candidates are
  explicitly penalized against positive candidates, not just labeled by an
  auxiliary harm BCE
- test seed `2` only first if such a new objective is introduced
- proceed to the full sentinel gate only if seed `2` recovers at least the
  frozen positive selected delta behavior

Avoid further d7 expansion until the seed-2 preservation check passes.
