# Ablation and Failure-Path Synthesis

This document explains why the final research direction is a neural
predecoder plus PyMatching, rather than a standalone neural decoder or another
d7 threshold-tuning loop.

Source artifact:

- `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json`

Builder:

- `tools/build_ablation_failure_synthesis_summary.py`

Primary source inputs:

- `artifacts/eval/nn/sedp_baseline_comparison_summary.json`
- `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`
- `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json`
- `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
- `artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json`

## Main Synthesis

The ablation trail supports this final interpretation:

> The neural model should not replace PyMatching directly. Its useful role is
> to propose local syndrome edits under a selected-mode safety policy, while
> PyMatching remains the final decoder.

## Architecture Decision Table

| decision point | tested path | key evidence | verdict |
| --- | --- | --- | --- |
| Use a standalone direct neural `logical_class4` decoder | FLFD-small | d3 `0.792968750`, d5 `0.761230469`, d7 `0.195312500`; all below same-ladder PyMatching refresh | reject as final model family |
| Fix direct classification with a multiscale dense trunk | M3D-FLFD | d3 `0.731933594`, d5 `0.761230469`; stronger d5 collapses to `0.077148438` | reject as final model family |
| Use RectCNN as the main comparison | RectCNN readiness | only 24-shot d3 readiness artifact, accuracy `0.041666667` | retain only as context/readiness baseline |
| Use neural pre-decoding rather than neural replacement | patch-head local motif selector plus PyMatching | d3 `0.928710938 -> 0.936279297`, d5 `0.888671875 -> 0.899902344` | retain as final proposed structure |

Interpretation:

- Direct neural classifiers were useful negative baselines, but they did not
  justify replacing PyMatching.
- Making the direct neural classifier more multiscale did not solve the
  distance-scaling problem.
- The first robust positive direction is not end-to-end classification; it is
  selected local syndrome editing before PyMatching.

## D7 Failure-Path Table

| decision point | key evidence | verdict |
| --- | --- | --- |
| Explain d7 as candidate-set exhaustion | all `58` checked d7 seeds have positive oracle headroom; mean candidate-oracle delta `+0.096679688` | reject |
| Solve d7 by scalar adoption-threshold tuning | `183040` policies checked, `0` pass the preserve/recover/block sentinel gate | reject |
| Solve d7 by identity-margin weight alone | d7 sentinel seeds `0,2,5` prefer weight `0.5`, but `0.25` admits harmful seed0 and `1.0` suppresses true-positive seed2 | reject as full d7 solution; retain `0.5` as local compromise |
| Use cross-family hard positive-vs-negative objective | seed54 false-positive gate remains blocked only by fallback; candidate deltas `-0.006835938` and `-0.009765625` | reject |
| Use candidate-compatibility pairwise top-k | blocks seed54, but seed2 candidate delta collapses to `-0.136718750` | reject |

Interpretation:

- D7 is not failing because there are no useful candidates.
- D7 is also not fixed by scalar threshold calibration.
- The real d7 bottleneck is learned selector ranking/generalization: the model
  often assigns too much rank to neutral or harmful nonzero edits under
  held-out `stage_c_corr`.

## Why Candidate-First Safety Remains Necessary

Candidate-first selected mode is not cosmetic. It is required because local
edits can help or harm final PyMatching.

| distance | selected improved/harmed | candidate improved/harmed | implication |
| --- | ---: | ---: | --- |
| d3 | `104/73` | `104/73` | positive but still has nontrivial harmful edits |
| d5 | `56/10` | `56/10` | cleanest successful distance |
| d7 | `16/7` | `161/251` | raw candidate branch is unsafe; selected fallback is essential |

D7 shows the safety point most clearly: the candidate branch harms more shots
than it improves, while selected mode mostly falls back to raw no-edit and
avoids a large accuracy drop.

## Final Structure Justification

The final model structure is justified by the following chain:

1. Raw PyMatching is strong and must remain the fair baseline.
2. Direct neural `logical_class4` classifiers do not scale reliably.
3. Local-edit oracle results show that syndrome edits can in principle recover
   PyMatching mistakes.
4. Patch-head selected pre-decoding recovers part of that oracle headroom at
   d3 and d5.
5. D7 exposes selector-ranking/generalization failure rather than missing
   candidate coverage.

Therefore the thesis should argue for:

```text
36-channel syndrome/noise volume
  -> 3D residual neural predecoder
  -> local motif candidates
  -> patch-head benefit/harm selector
  -> selected local edit or raw no-edit
  -> PyMatching
  -> logical_class4
```

## Paper Use

Use this synthesis in the thesis discussion and ablation sections.

Strong claim:

> The ablation results support neural pre-decoding as a useful front-end to
> PyMatching on d3 and d5, while d7 reveals a selector-ranking bottleneck.

Do not claim:

- the direct neural baselines are a strict final head-to-head comparison with
  the predecoder artifacts
- d7 is solved
- d7 candidate coverage is exhausted
- another scalar adoption sweep is likely to be the right next step
