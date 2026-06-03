# Predecoder Method Description

This document is the canonical method description for the thesis. It describes
the actual implemented model, not an idealized variant.

Primary implementation:

- `decoders/syndrome_edit_predecoder.py`

Related result documents:

- `PREDECODER_FINAL_RESULT_TABLES.md`
- `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
- `PREDECODER_BASELINE_COMPARISON.md`
- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`

## Method Boundary

The proposed model is a neural predecoder. It does not directly replace
PyMatching and it does not directly predict the final logical class.

The final inference path is:

```text
36-channel syndrome/noise volume
  -> SyndromeEditPreDecoder 3D residual trunk
  -> local motif candidate generation
  -> patch-head CandidateEditSelector
  -> selected local detector edit or raw no-edit fallback
  -> PyMatching on edited/raw syndrome
  -> logical_class4 prediction
```

The neural model proposes whether a small local detector-event edit should be
applied before PyMatching. PyMatching remains the final logical-frame decoder.

## Input Representation

Each shot is represented as a distance-dependent 3D volume with `36` channels.
The spatial and temporal dimensions change with code distance, while the
channel semantics remain fixed.

Examples from the successful d3/d5 runs:

| distance | input tensor shape |
| --- | --- |
| d3 | `[36, 4, 4, 4]` |
| d5 | `[36, 6, 6, 6]` |

The channels include detector events, a valid-detector mask, detector geometry
metadata, noise-family/stage indicators, distance/round statistics, event
fractions, and physical-noise summary channels.

## Neural Trunk

The trunk class is `SyndromeEditPreDecoder`.

| component | setting |
| --- | --- |
| stem | 3D convolution |
| residual trunk | `3` residual 3D convolution blocks |
| hidden channels | `24` |
| dense hidden dim | `64` |
| dropout | `0.1` |
| d3/d5 parameter count | `118834` |

The trunk produces two neural outputs:

| output | role |
| --- | --- |
| `edit_logits` | per-detector local edit score volume |
| `needs_edit_logits` | shot-level score for whether an edit is useful |

The trunk also produces pooled shot features. These pooled features are passed
to the candidate selector.

## Candidate Generation

The system does not allow arbitrary high-weight syndrome edits. It builds a
candidate set of small local edits.

The successful d3/d5 recipe uses:

- `--selection-mode local_motif_selector`
- `--selector-policy-candidate-mode none`
- `--selector-local-motif-max-classes 16`
- `--selector-local-motif-top-k 32`

`local_motif_selector` uses local motif candidates learned from observed
hard-shot edit targets. `--selector-policy-candidate-mode none` removes raw
threshold/top-k policy candidates while keeping identity and local-motif
candidates. This is important because unrestricted candidate emission can harm
PyMatching.

The candidate set always includes the identity/no-edit option. This makes
fallback to raw PyMatching part of the method.

## Candidate Features

For each candidate edit, the selector receives:

- pooled shot embedding from the 3D trunk
- candidate edit weight and neural edit-probability summaries
- local motif and pattern features
- detector geometry summaries
- local detector-event/evidence features
- local patch features around the candidate anchor
- benefit/harm transition features relative to raw PyMatching

The successful recipe enables geometry, pattern, local-evidence, and local
patch candidate features.

## Patch-Head Selector

The selector class is `CandidateEditSelector`.

The patch-head path first embeds the local patch feature slice with a small
MLP. The selector then concatenates:

```text
shot features
  + non-patch candidate features
  + embedded local patch features
```

The final selector MLP assigns a scalar score to each candidate edit. The best
candidate is not automatically adopted. It must still pass the selected-mode
safety policy.

## Selector Target

The successful recipe uses:

- `--selector-target-mode benefit_harm`
- pairwise benefit/harm ranking loss

The target is based on final system behavior after PyMatching, not merely on
detector-level reconstruction. A candidate is useful when applying it and then
running PyMatching improves the final `logical_class4` correctness relative to
raw no-edit PyMatching.

This is the central difference from a plain detector-level edit-mask model:
the selector is trained and evaluated around whether the candidate helps the
final decoder.

## Selected-Mode Adoption

The selected-mode policy is `candidate_first_safety`.

The policy compares validation-family behavior before deciding whether the
candidate selector should be used. If the candidate branch does not provide
enough validation evidence, the selected mode falls back to raw no-edit
PyMatching.

This fallback is not a separate baseline. It is part of the proposed system.
It is needed because local edits can be beneficial for some seeds/families and
harmful for others.

Current selected-mode behavior on held-out `stage_c_corr`:

| distance | selected behavior | interpretation |
| --- | --- | --- |
| d3 | local selector `8/8` seeds | stable positive selected-mode result |
| d5 | local selector `2/8`, raw no-edit `6/8` | conservative result; safe fallback matters |
| d7 | local selector `2/58`, raw no-edit `56/58` | controlled limitation, not solved recovery |

## PyMatching Handoff

After selection, the chosen local edit is applied to the detector-event
syndrome. The edited syndrome, or the unchanged raw syndrome when no edit is
selected, is decoded by PyMatching.

The final reported output is `logical_class4`, with classes:

| class | logical frame |
| --- | --- |
| `0` | `I` |
| `1` | `X` |
| `2` | `Z` |
| `3` | `Y` |

## What The Method Is Not

The method should not be described as:

- an end-to-end neural replacement for PyMatching
- a direct `logical_class4` neural classifier
- a correction-field decoder that emits the final correction without matching
- a solved d7 decoder

The correct description is:

> a transition-aware neural local-edit predecoder that selectively modifies
> detector-event syndromes before final PyMatching decoding.

## Thesis Claim Supported By This Method

The method supports this claim:

> A transition-aware patch-head neural predecoder can improve PyMatching on
> held-out d3 and d5 surface-code logical-frame decoding by selecting local
> detector edits before matching, while d7 reveals a selector-ranking and
> generalization limitation rather than a lack of local-edit candidate
> headroom.
