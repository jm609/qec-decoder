# Risk-Aware Selector Exploratory Plan

This document fixes the plan for the post-presentation d7 selector experiment.
It is intentionally separated from the frozen main thesis evidence.

## Scope And Separation

Status: optional exploratory work.

The current thesis/poster results remain frozen unless this experiment passes
the explicit gates below. The main claim is still:

- d3 and d5 show positive selected-mode predecoder gains.
- d7 shows a controlled scaling limitation.
- the d7 bottleneck is selector ranking/generalization, not candidate coverage.

This experiment must not overwrite existing tables, figures, checkpoints, or
summary JSON files used by the thesis. Use a separate artifact root such as:

```text
artifacts/eval/nn/risk_aware_selector_exploratory/
artifacts/reports/risk_aware_selector_exploratory/
```

If the experiment fails, it should be excluded from the poster and main thesis
results. At most, it can be mentioned internally as an attempted but rejected
follow-up.

## Frozen Baseline

Held-out `stage_c_corr` baseline evidence:

| distance | raw PyMatching | selected predecoder | candidate branch | oracle | selected delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.935302734` | `0.935302734` | `0.992187500` | `+0.006591797` |
| d5 | `0.888671875` | `0.894287109` | `0.891845703` | `0.978515625` | `+0.005615234` |
| d7 | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

D7 fixed observations:

- mean actual candidate delta: `-0.001515356`
- mean candidate-oracle delta: `+0.096679688`
- all `58` checked d7 seeds have positive oracle headroom
- selected-positive seeds: `2, 11`
- missed candidate-positive seeds: `0, 28, 43, 45`
- validation-positive d7 candidate branches are held-out harmful in `13/22`
  cases

Interpretation: the candidate set often contains useful edits, but the selector
does not rank/adopt them reliably at d7.

## Hypothesis

The current selector compresses benefit and harm into one adoption score. This
is fragile at d7 because a candidate can look locally plausible while being
held-out harmful after the PyMatching handoff.

The next reasonable experiment is therefore not another scalar threshold sweep.
It is a risk-aware selector that separately estimates:

- probability or score that an edit is beneficial
- probability or score that an edit is harmful
- no-edit fallback preference

The intended gain is not a new feature family. The intended gain is better
candidate adoption under the existing candidate generator.

## Proposed Selector Change

Keep fixed:

- dataset construction
- noise stages
- `36`-channel input volume
- `SyndromeEditPreDecoder` 3D residual trunk
- local motif candidate generation
- PyMatching handoff
- logical-class evaluation

Change only the selector/adoption layer.

Candidate scoring should have separate outputs, for example:

```text
candidate_features, shot_features
  -> shared MLP
  -> benefit_logit
  -> harm_logit
  -> optional no_edit_logit or no_edit_margin
```

A simple adoption utility can be:

```text
utility = benefit_logit - lambda_harm * harm_logit
```

Adopt an edit only when all of these are satisfied:

- benefit score is high enough
- harm score is low enough
- utility beats the raw no-edit fallback by a margin

This is a selector redesign, not a new predecoder architecture.

Implementation status:

- `decoders/syndrome_edit_predecoder.py` now supports
  `--selector-model risk_aware` and `--selector-model risk_guard`.
- The new selector keeps the existing shot/candidate/patch feature inputs.
- The `risk_aware` forward score is the adoption utility:

```text
utility_logit = benefit_logit - harm_weight * harm_logit
```

- The `risk_guard` forward score is:

```text
utility_logit = rank_logit - harm_weight * harm_logit
```

- Auxiliary BCE losses train `benefit_logit` from positive target-score
  candidates and `harm_logit` from negative target-score candidates.
- Default scalar selector behavior remains unchanged under
  `--selector-model scalar`.

Recommended risk-aware CLI switches:

```text
--selector-model risk_aware
--selector-target-mode benefit_harm
--selector-objective group_rank
--selector-risk-aware-harm-logit-weight 1.0
--selector-risk-aware-benefit-loss-weight 1.0
--selector-risk-aware-harm-loss-weight 1.0
```

Use explicit exploratory output paths under:

```text
artifacts/eval/nn/risk_aware_selector_exploratory/
```

Initial seed-2 preservation result:

- recorded in `RISK_AWARE_SELECTOR_EXPLORATORY_RESULTS.md`
- tested `risk_aware` and `risk_guard` variants
- early smoke runs validated the new selector code but were not a fully fair
  candidate-set comparison because scalar local-motif settings were omitted
- the fair `risk_guard` rerun restored the scalar candidate-set settings, but
  still failed to preserve the frozen seed `2` selected-mode gain
- a post-hoc harm-logit hard-guard diagnostic also failed: the harm head
  separates harmful rows in aggregate, but does not reliably filter the
  high-utility harmful candidates that control adoption
- a final combined check with `risk_guard` plus within-shot positive-vs-negative
  hard ranking (`0.5/0.5`) improved seed-2 candidate delta relative to plain
  `risk_guard`, but still failed selected-mode preservation because validation
  harmed count exceeded the safety cap
- do not expand this exact path to the full sentinel set unless seed `2`
  preservation is first recovered by a genuinely different objective

## Sentinel Gate

Before any full d7 run, the new selector must pass a small sentinel gate.

| gate role | required seeds | required behavior |
| --- | --- | --- |
| preserve true positives | `2, 11` | keep the existing selected/candidate gains |
| recover missed positives | at least one of `0, 28, 43, 45` | recover useful candidate edits that current selected mode misses |
| block compact false positives | `8, 13, 17, 33, 54` | avoid known harmful adoption cases |
| block extended false positives | recommended check: `8, 13, 17, 18, 32, 33, 53, 54` | stronger guard before reporting as an exploratory success |

If the compact gate fails, stop. Do not expand to d7 seeds `0..57`.

If the compact gate passes but the extended false-positive check fails, treat
the result as unstable and do not include it in the poster or main thesis.

## Experiment Phases

| phase | action | output | stop rule |
| ---: | --- | --- | --- |
| 0 | create isolated run names and artifact directories | no main-result files touched | stop if isolation is not clean |
| 1 | implement risk-aware selector head/adoption path | syntax check and tiny smoke run | stop if implementation changes the predecoder trunk or data schema |
| 2 | run d7 sentinel only | per-seed raw/candidate/selected/oracle deltas | stop if compact sentinel fails |
| 3 | run d7 seeds `0..57` only after sentinel pass | mean d7 selected delta and per-seed taxonomy | stop if selected gain remains baseline-level |
| 4 | run d3/d5 regression checks | confirm successful distances are not damaged | stop if d3 or d5 becomes harmful |
| 5 | document include/exclude decision | one short result note | exclude unless all success criteria are met |

## Success Criteria

Minimum success:

- d7 selected delta is meaningfully above the frozen value `+0.000151536`
- at least one missed-positive d7 seed is recovered
- existing true-positive d7 seeds `2` and `11` are preserved
- compact and preferably extended false-positive seeds remain blocked
- d3 and d5 do not regress into harmful selected-mode behavior

Stronger success:

- d7 selected delta reaches approximately `+0.002` or higher
- selected-mode recovery uses a simple rule that can be explained in the
  thesis or poster
- full d7 per-seed results show fewer harmful selected adoptions, not just a
  better mean due to one lucky seed

## Exclusion Criteria

Exclude this experiment from the poster and main thesis results if any of these
occur:

- sentinel gate fails
- d7 selected delta stays near the frozen baseline
- d3 or d5 selected-mode behavior regresses
- the adoption policy requires a broad held-out threshold search
- the method becomes too complex to explain as a selector-risk modification
- success depends on tuning directly to the held-out reporting seeds

The failure case is still scientifically useful as an internal conclusion:
d7 improvement likely requires a stronger ranking/generalization method than a
small selector-head change.

## Reporting Policy

If successful:

- report it as a clearly labeled exploratory selector improvement
- keep the frozen d3/d5/d7 table as the main baseline table
- add a small supplemental table comparing old selected mode and risk-aware
  selected mode
- state that the method was developed after the main presentation and needs
  further validation

If unsuccessful:

- do not include it in the poster
- do not alter the main thesis claim
- preserve the frozen d7 limitation narrative
- record only a short internal note if needed

## Schedule

Recommended schedule given the remaining poster and thesis deadlines:

| day | target |
| ---: | --- |
| 1 | implement isolated risk-aware selector path and smoke test |
| 2 | run d7 sentinel gate |
| 3 | if sentinel passes, run d7 seeds `0..57`; otherwise stop |
| 4 | if d7 improves, run d3/d5 regression checks |
| 5 | decide poster inclusion/exclusion before poster finalization |
| after poster | include in thesis only if the result is stable and explainable |

Do not let this experiment consume the thesis-writing path. If it is not
clearly successful by the poster decision point, freeze it out.

## Work To Avoid

- no broad scalar-threshold sweep
- no new 3D trunk
- no new noise model or dataset regeneration
- no feature expansion before the selector-risk hypothesis is tested
- no full d7 run before sentinel pass
- no modification of main thesis artifacts during exploratory runs
