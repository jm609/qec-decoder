# Predecoder Final Result Tables

Final thesis title:

> 표면 코드 양자 오류 정정을 위한 전이 정보 기반 신경망 사전 디코더의 설계 및 성능 분석

English title:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Source artifact:

- `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`

Consistency check:

- `artifacts/eval/nn/sedp_final_result_consistency_check.json`
- builder: `tools/build_final_result_consistency_summary.py`
- status: `pass`, `37` checks, `0` failures

Successful d3/d5 structure note:

- `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
- `PREDECODER_D3_D5_PAIRED_STATISTICS.md`
- `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json`

Method description:

- `PREDECODER_METHOD_DESCRIPTION.md`

Baseline comparison note:

- `PREDECODER_BASELINE_COMPARISON.md`
- `artifacts/eval/nn/sedp_baseline_comparison_summary.json`

Ablation/failure synthesis note:

- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
- `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json`

All metrics below use held-out `stage_c_corr` and class4 PyMatching accuracy.

## Main Accuracy Table

| distance | seeds | raw PyMatching | selected predecoder | candidate branch | target local-edit oracle | selected delta | oracle recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d3 | `0..7` | `0.928710938` | `0.935302734` | `0.935302734` | `0.992187500` | `+0.006591797` | `10.38%` |
| d5 | `0..7` | `0.888671875` | `0.894287109` | `0.891845703` | `0.978515625` | `+0.005615234` | `6.25%` |
| d7 | `0..57` | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` | `0.14%` |

Interpretation:

- d3 and d5 are positive selected-mode results.
- d3 is uniformly positive over the expanded eight-seed check.
- d5 is more conservative: selected mode stays non-harmful by falling back to
  raw no-edit on weak or harmful candidate seeds.
- Exact paired tests reinforce this wording: d3 one-sided sign/sign-flip
  p-value is `0.003906250`, while d5 is `0.250000000` because only two seeds
  have nonzero selected gains.
- d7 selected mode is safe but recovers essentially none of the available
  local-edit oracle gap.

## Selected-Mode Behavior

| distance | selected modes | selected improved/harmed | candidate improved/harmed |
| --- | --- | ---: | ---: |
| d3 | local selector `8/8` | `205/151` | `205/151` |
| d5 | local selector `2/8`, raw no-edit `6/8` | `56/10` | `130/104` |
| d7 | local selector `2/58`, raw no-edit `56/58` | `16/7` | `161/251` |

Interpretation:

- d3 improves more shots than it harms, but still has a nontrivial harm count.
- d5 selected mode has a cleaner improved/harmed ratio because safety adoption
  blocks harmful candidate branches.
- d7 candidate branch harms more shots than it improves; the adoption guard is
  doing necessary safety work.

## D7 Oracle Gap

The support-guard d7 analysis over seeds `0..57` gives:

| metric | value |
| --- | ---: |
| mean selected delta | `+0.000151536` |
| mean actual candidate delta | `-0.001515356` |
| target local-edit oracle delta | `+0.111328125` |
| learned candidate-oracle delta | `+0.096679688` |
| candidate-to-oracle gap | `+0.098195043` |
| actual candidate outcomes | positive `6`, neutral `35`, harmful `17` |
| oracle outcomes | positive `58` |

Interpretation:

- d7 has enough candidate/oracle headroom.
- The failure is not candidate coverage.
- The failure is that the learned selector often ranks harmful or neutral edits
  above the edits that would exploit the oracle headroom.

## Claim Boundary

Defensible claim:

> The transition-aware patch-head predecoder improves held-out logical-frame
> decoding over PyMatching on d3 and d5 under selected-mode safety. At d7, the
> same framework exposes a scaling limitation: high oracle headroom remains,
> but learned selector ranking does not reliably recover it.

Do not claim:

- solved d7 learned recovery
- d7 selected-mode improvement beyond a sparse guarded signal
- candidate-set exhaustion at d7
