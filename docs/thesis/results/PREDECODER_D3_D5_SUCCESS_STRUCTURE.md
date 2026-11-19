# D3/D5 Successful Predecoder Structure

This note freezes the successful d3/d5 structure before further d7 work.
It should be read together with:

- `PREDECODER_FINAL_RESULT_TABLES.md`
- `PREDECODER_CONSOLIDATED_EVIDENCE.md`
- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
- `artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json`
- `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_3.json`
- `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json`

## Fixed Inference Path

The successful d3/d5 system is not a standalone neural decoder. It is a
neural predecoder followed by PyMatching:

```text
36-channel syndrome/noise volume
  -> 3D residual syndrome-edit trunk
  -> local motif candidate set
  -> patch-head candidate selector
  -> selected local detector edit or raw no-edit fallback
  -> PyMatching on edited/raw syndrome
  -> logical_class4 prediction
```

The neural part edits detector events locally; PyMatching remains the final
logical-frame decoder.

## Shared Model Structure

Both d3 and d5 use the same neural structure:

| component | setting |
| --- | --- |
| trunk class | `SyndromeEditPreDecoder` |
| trunk | 3D convolution stem + `3` residual 3D blocks |
| hidden channels | `24` |
| dense hidden dim | `64` |
| dropout | `0.1` |
| parameters | `118834` |
| edit output | per-detector `edit_logits` volume |
| shot output | `needs_edit_logits` plus pooled shot embedding |
| final decoder | PyMatching |

The distance changes the volume shape, not the core neural module:

| distance | input shape |
| --- | --- |
| d3 | `[36, 4, 4, 4]` |
| d5 | `[36, 6, 6, 6]` |

The 36 input channels include detector events, valid mask, geometric/check
metadata, noise-family/stage indicators, distance/round statistics, event
fractions, and physical-noise summary channels.

## Candidate And Selector Structure

The successful recipe uses:

- `--selection-mode local_motif_selector`
- `--selector-target-mode benefit_harm`
- `--selector-patch-head`
- `--selector-policy-candidate-mode none`
- `--selector-local-motif-max-classes 16`
- `--selector-local-motif-top-k 32`
- geometry, pattern, local-evidence, and local-patch candidate features

`--selector-policy-candidate-mode none` disables the raw threshold/top-k policy
candidate branch while keeping identity and local-motif candidates. The local
motif vocabulary is built from observed hard-shot edit targets.

For each candidate edit, the selector receives:

- pooled shot embedding from the 3D trunk
- base candidate statistics such as edit weight and neural edit probabilities
- candidate geometry and local motif pattern features
- local detector-event/evidence features
- local patch features around the candidate anchor
- benefit/harm transition features relative to raw PyMatching

The patch-head path embeds the local patch feature slice with a small MLP and
then concatenates it with the non-patch candidate features and shot embedding.
The final `CandidateEditSelector` MLP scores each candidate edit.

## Training And Adoption

The training families are:

- `stage_a_si1000`
- `stage_b_local`

The held-out evaluation family is:

- `stage_c_corr`

The selected-mode adoption policy is `candidate_first_safety`: use the local
motif selector only when validation evidence says the candidate branch is
beneficial under the active safety guards; otherwise use raw no-edit PyMatching.
This fallback is part of the method, not a separate baseline.

## D3 Seed-Level Result

Source:

- `artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_3.json`

| seed | selected mode | epoch | val candidate delta | held-out delta | improved/harmed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `local_motif_selector` | 5 | `+0.022629404` | `+0.010742188` | `31/20` |
| 1 | `local_motif_selector` | 3 | `+0.042301928` | `+0.006835938` | `20/13` |
| 2 | `local_motif_selector` | 5 | `+0.022562343` | `+0.008789062` | `29/20` |
| 3 | `local_motif_selector` | 4 | `+0.029228952` | `+0.003906250` | `24/20` |
| 4 | `local_motif_selector` | 6 | `+0.035635202` | `+0.001953125` | `29/27` |
| 5 | `local_motif_selector` | 6 | `+0.025923769` | `+0.003906250` | `23/19` |
| 6 | `local_motif_selector` | 8 | `+0.032299385` | `+0.009765625` | `30/20` |
| 7 | `local_motif_selector` | 8 | `+0.042229009` | `+0.006835938` | `19/12` |

Aggregate d3 result:

| raw PyMatching | selected predecoder | delta | selected behavior |
| ---: | ---: | ---: | --- |
| `0.928710938` | `0.935302734` | `+0.006591797` | `8/8` local selector |

Interpretation: d3 is a stable positive selected-mode result. Every checked
seed adopts the local selector and improves held-out `stage_c_corr`.

## D5 Seed-Level Result

Source:

- `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_7.json`

| seed | selected mode | epoch | val candidate delta | held-out delta | improved/harmed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `raw_no_edit` | 1 | `+0.000000000` | `+0.000000000` | `0/0` |
| 1 | `raw_no_edit` | 1 | `+0.000000000` | `+0.000000000` | `0/0` |
| 2 | `local_motif_selector` | 2 | `+0.000007544` | `+0.021484375` | `28/6` |
| 3 | `local_motif_selector` | 1 | `+0.013000710` | `+0.023437500` | `28/4` |
| 4 | `raw_no_edit` | 1 | `+0.009587869` | `+0.000000000` | `0/0` |
| 5 | `raw_no_edit` | 2 | `+0.000000000` | `+0.000000000` | `0/0` |
| 6 | `raw_no_edit` | 1 | `+0.003174177` | `+0.000000000` | `0/0` |
| 7 | `raw_no_edit` | 1 | `+0.000000000` | `+0.000000000` | `0/0` |

Aggregate d5 result:

| raw PyMatching | selected predecoder | delta | selected behavior |
| ---: | ---: | ---: | --- |
| `0.888671875` | `0.894287109` | `+0.005615234` | `2/8` local selector, `6/8` raw no-edit |

Interpretation: d5 is a conservative selected-mode result. The method does not
need every seed to emit edits; the safety policy falls back to raw PyMatching
on weak seeds and adopts the local selector on seeds whose edits have enough
evidence. The adopted d5 seeds have a much cleaner improved/harmed ratio than
d3, and the raw no-edit fallback blocks harmful candidate-only seeds.

## Robustness Follow-Up

The robustness summary in `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md` confirms:

| distance | positive/neutral/harmful held-out seeds | min selected delta | selected improved/harmed |
| --- | ---: | ---: | ---: |
| d3 | `8/0/0` | `+0.001953125` | `205/151` |
| d5 | `2/6/0` | `+0.000000000` | `56/10` |

This means the d3/d5 success claim should be stated as selected-mode
robustness, not unconditional local-edit robustness. d3 emits edits in all
checked seeds; d5 emits edits only when the adoption policy accepts the
candidate branch and otherwise remains safely at raw no-edit.

## Paper Claim Supported By D3/D5

Supported claim:

> A transition-aware patch-head neural predecoder can improve PyMatching on
> held-out surface-code logical-frame decoding at d3 and d5 by applying local
> detector edits only when selected-mode safety accepts the candidate branch.

Important boundary:

- The claim is selected-mode, not candidate-oracle.
- The claim is d3/d5 positive, not d7 solved.
- PyMatching remains the final decoder; the neural model is a predecoder.
