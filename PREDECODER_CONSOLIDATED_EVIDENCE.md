# Predecoder Consolidated Evidence

As of 2026-05-09, the main predecoder evidence is consolidated around:

> 표면 코드 양자 오류 정정을 위한 전이 정보 기반 신경망 사전 디코더의 설계 및 성능 분석

English title:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Technical topic:

> Transition-aware neural pre-decoding for surface-code logical-frame inference.

The generated evidence artifact is:

- `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`

Paper-ready result tables:

- `GRADUATION_THESIS_DRAFT.md`
- `PREDECODER_METHOD_DESCRIPTION.md`
- `PREDECODER_CLEAN_HANDOFF.md`
- `PREDECODER_FINAL_RESULT_TABLES.md`
- `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
- `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md`
- `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
- `PREDECODER_BASELINE_COMPARISON.md`
- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
- `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`
- `PREDECODER_REMAINING_WORK.md`

Final result consistency check:

- `artifacts/eval/nn/sedp_final_result_consistency_check.json`
- `tools/build_final_result_consistency_summary.py`
- status: pass, `37` checks, `0` failures

## Current Best Selected Results

All rows use held-out `stage_c_corr` delta over raw no-edit PyMatching.

| distance | seeds | raw PyMatching | selected predecoder | target local-edit oracle | mean selected delta | selected improved/harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d3 | 8 | `0.928710938` | `0.935302734` | `0.992187500` | `+0.006591797` | `205/151` |
| d5 | 8 | `0.888671875` | `0.894287109` | `0.978515625` | `+0.005615234` | `56/10` |
| d7 | 58 | `0.873046875` | `0.873198411` | `0.984375000` | `+0.000151536` | `16/7` |

Interpretation:

- d3 and d5 are positive selected-mode results under the candidate-first safety
  policy.
- d7 selected mode is safe but effectively mostly raw no-edit.
- d7 should not be presented as a solved learned-recovery result.

The d3/d5 robustness follow-up is:

- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
- `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json`

It confirms that d3 has `8/0/0` positive/neutral/harmful held-out seeds and d5
has `2/6/0`; neither successful distance has a harmful selected seed in the
current seed `0..7` evidence. The bootstrap CI artifact adds seed-level 95%
intervals: d3 `[+0.004516602, +0.008544922]` and d5
`[+0.000000000, +0.013671875]`.

The noise-family follow-up is:

- `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
- `artifacts/eval/nn/sedp_noise_family_analysis_summary.json`

It records that d3 is positive across both validation families and held-out
`stage_c_corr`, while d5 has mixed validation-family slices but remains
non-harmful on held-out `stage_c_corr`. D7 is kept as a contrast because
validation-positive candidate evidence maps to held-out candidate harm more
often than held-out candidate gain.

The baseline-comparison follow-up is:

- `PREDECODER_BASELINE_COMPARISON.md`
- `artifacts/eval/nn/sedp_baseline_comparison_summary.json`

It fixes the comparison boundary: raw no-edit PyMatching on the same
predecoder target artifacts is the fair main baseline, while FLFD/M3D/RectCNN
are context baselines showing why the project moved away from standalone
neural `logical_class4` classifiers.

The ablation/failure-path synthesis is:

- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
- `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json`

It converts the negative and partial results into the final method rationale:
direct neural classifiers and multiscale direct classifiers are not the final
model family, scalar d7 adoption tuning fails the sentinel gate, and the final
defensible structure is neural local-edit pre-decoding followed by PyMatching.

The method description is:

- `PREDECODER_METHOD_DESCRIPTION.md`

It should be used as the source for the thesis method section: 36-channel
syndrome/noise input, 3D residual trunk, local motif candidates, patch-head
benefit/harm selector, selected-mode safety, and PyMatching handoff.

## D7 Oracle Gap

The support-guard d7 candidate-oracle analysis over seeds `0..57` shows:

| metric | value |
| --- | ---: |
| checked seeds | `58` |
| selected local seeds | `2, 11` |
| mean selected delta | `+0.000151536` |
| mean actual candidate delta | `-0.001515356` |
| mean candidate-oracle delta | `+0.096679688` |
| mean candidate-to-oracle gap | `+0.098195043` |
| candidate outcomes | positive `6`, neutral `35`, harmful `17` |
| oracle outcomes | positive `58` |

Interpretation:

- Candidate coverage is not the blocker at d7.
- The blocker is learned selector ranking/generalization: many runs contain
  oracle-positive candidates, but the learned selector ranks neutral or harmful
  nonzero edits too highly.

The targeted d7 bottleneck follow-up is:

- `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`
- `artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json`
- `artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json`
- `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json`
- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
- `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`
- `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md`
- `PREDECODER_REPRODUCIBILITY_PACKAGE.md`
- `PREDECODER_FIGURE_PACKAGE.md`
- `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`

The d7 validation-vs-heldout scatter figure is:

- `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg`
- validation-positive held-out outcomes: harmful `13`, neutral `4`, positive
  `5`
- validation-vs-heldout Pearson correlation: `-0.452872`

The seed-level oracle recovery distribution is:

- `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md`
- `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg`
- selected mean recovery of the target local-edit oracle gap:
  d3 `10.38%`, d5 `6.25%`, d7 `0.14%`
- d7 candidate-oracle recovery mean: `86.84%`

It fixes the next d7 sentinel sets:

| role | seeds |
| --- | --- |
| preserve true positives | `2, 11` |
| recover missed positives | `0, 28, 43, 45` |
| block validation false positives | `13, 17, 33, 54, 53, 32, 8, 18` |
| inspect high-oracle misses | `3, 5, 9, 13, 34, 36, 47, 49` |

The simple adoption-threshold grid checked `183040` policies and found `0`
that pass the preserve/recover/block sentinel gate. The best recovery policy
recovers `0,28,43,45` but also opens harmful false positives
`13,17,18,54` and has mean selected delta `-0.000454607`.

The harmful-edit taxonomy adds the held-out failure shape:

| harmful taxonomy metric | value |
| --- | ---: |
| harmful candidate seeds | `17` |
| harmful candidate seeds blocked by selected mode | `17/17` |
| validation-positive false-positive harmful seeds | `13` |
| validation-positive true-positive seeds | `5` |
| false-positive ratio among validation-positive seeds | `59.09%` |
| mean harmful candidate delta | `-0.005974265` |
| mean harmful candidate oracle delta | `+0.104549632` |

Interpretation:

- The selected-mode guard is useful: it blocks every harmful candidate branch
  found in the 58-seed d7 analysis.
- The learned selector is not yet useful enough at d7: validation-positive
  evidence frequently becomes held-out harmful.
- The next d7 objective must distinguish validation-positive true positives
  from validation-positive false positives before any broad sweep is justified.

## Recent Rejected D7 Directions

The first in-training cross-family hard positive-vs-negative objective was
implemented with default-off flags:

- `--selector-cross-family-positive-negative-loss-weight`
- `--selector-cross-family-positive-negative-margin`

Seed `54` was used as the first false-positive gate. The follow-up
candidate-compatibility top-k check added seed `2` as a true-positive
preservation gate.

| setting | selected delta | candidate delta | verdict |
| --- | ---: | ---: | --- |
| cross-family `0.25/0.5` | `+0.000000000` | `-0.006835938` | unchanged from support guard |
| cross-family `1.0/0.5` | `+0.000000000` | `-0.009765625` | worse candidate branch |
| candidate-compat pairwise top-k seed `2/54` | `+0.000000000` | `-0.068359375` | blocks seed54 but destroys seed2 |

Verdict:

- This simple cross-family hard-negative objective is rejected.
- It failed the seed54 false-positive gate, so expanding it to true-positive
  seeds `2` and `11` is not justified.
- Candidate-compatibility pairwise top-k is also rejected: it blocks seed54,
  but seed2 candidate delta collapses to `-0.136718750`.

## Claim Boundary

The strongest defensible claim is:

> A transition-aware patch-head predecoder can beat raw PyMatching on d3 and d5
> held-out `stage_c_corr` under selected-mode candidate-first safety, while d7
> exposes a scaling limitation: oracle headroom remains high but learned
> selector ranking is not reliable enough to recover it.

The current result should not claim:

- robust d7 learned improvement
- solved selected-mode d7 recovery
- candidate-set exhaustion at d7

## Next Work

The rational next step is tracked in
`PREDECODER_REMAINING_WORK.md`:

- do not repeat completed d3/d5 robustness, noise-family, or baseline-boundary
  checks as the next priority
- treat the d7 harmful-edit taxonomy as complete and use it in the limitation
  chapter
- treat the integrated core thesis draft and reproducibility package as the
  current writing baseline
- use `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md` as the clean Korean submission
  prose source
- use `PREDECODER_FIGURE_PACKAGE.md` and `artifacts/figures/predecoder/*.svg`
  for the final thesis figures
- integrate the ablation/failure-path synthesis table before any optional
  experiment
- write the final report around d3/d5 positive selected-mode results and d7
  scaling limitation
- keep any optional d7 work behind the preserve/recover/block sentinel gate
- if no new selector-ranking objective passes that gate, stop d7 optimization
  and present d7 as a controlled limitation
