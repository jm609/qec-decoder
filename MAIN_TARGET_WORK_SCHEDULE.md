# Main Target And Work Schedule

This document fixes the main target after the reference correction on
2026-04-24.

Current detailed research plan:

- `RESEARCH_PLAN_PREDECODER_MAIN.md`

The fixed main research topic is:

> Transition-aware neural pre-decoding for surface-code logical-frame
> inference.

## 1. Main Target Paper

The main paper-format anchor is:

- Jung, Ali, Ha, "Convolutional Neural Decoder for Surface Codes",
  IEEE Transactions on Quantum Engineering, 2024
- DOI: `10.1109/TQE.2024.3419773`

Public reference pages:

- `https://pure.kaist.ac.kr/en/publications/convolutional-neural-decoder-for-surface-codes`
- `https://ieeexplore.ieee.org/document/10574322`
- `https://tqe.ieee.org/2024/06/27/convolutional-neural-decoder-for-surface-codes/`

The paper's useful constraint for this project is now the task/evaluation
format, not the exact neural architecture. The target format is:

- map surface-code syndrome data onto a rectangular lattice-like input
- fill invalid / non-syndrome lattice cells with an incoherent value
- output final logical-frame information, currently `logical_class4`
- judge by final logical decoding correctness against PyMatching

In this repository, the closest implementation of that format is:

- `geometry/rotated_rect.py`
- `decoders/baseline_rectcnn.py`
- `sample_dataset.py` rectangular layout metadata

## 2. Project Target

The project target is not to copy the paper architecture. The target is:

> Build a robust logical-frame decoder with the same output/evaluation contract
> as the paper-format surface-code decoder, then compare it honestly against
> PyMatching under distance scaling and noise-family shift.

After the d3/d5/d7 model-selection pass, the current mainline implementation
target is more specific:

> Recover local-edit oracle headroom with a transition-aware neural
> pre-decoder before PyMatching, while preserving the final `logical_class4`
> evaluation contract.

Concretely, the mainline target is:

- input: rectangular / volume syndrome representation
- output: logical frame, currently `logical_class4 in {I, X, Z, Y}`
- baseline comparison: PyMatching on the same shots
- primary metrics:
  - class4 accuracy
  - macro-F1
  - confusion matrix
  - seen vs held-out noise-family gap
  - final decoded correctness / logical failure proxy
- secondary metrics:
  - calibration
  - selective fallback / hybrid behavior
  - latency and model size later

## 3. Track Map

### Track A: Paper-Format CNN Baseline

Purpose:

- keep the repo anchored to the main target paper
- make sure rectangular syndrome layout, incoherent fill, and CNN baseline
  behavior are correct
- do not treat this architecture as mandatory for the final model

Primary files:

- `geometry/rotated_rect.py`
- `decoders/baseline_rectcnn.py`
- `sample_dataset.py`
- `decoders/baseline_pymatching.py`

Status:

- implemented
- PyMatching class4 refresh is complete for the current d3/d5/d7 2k manifests
- RectCNN refresh is optional baseline work, not a blocker for choosing a
  stronger model family

### Track B: Main CNN Extension

Purpose:

- improve over the paper-format baseline while preserving its evaluation format
- test stronger logical-frame output structure and noise context

Primary files:

- `decoders/factorized_logical_frame_decoder.py`
- `decoders/research_noise_aware_3d.py`
- `decoders/multiscale_factorized_decoder.py`
- `tools/evaluate_hybrid_fallback.py`
- `tools/evaluate_learned_hybrid_router.py`

Status:

- implemented and partially tested
- d3 can show useful behavior
- d5 scaling remains weak relative to PyMatching

### Track C: PyMatching-Assist Pre-Decoder

Purpose:

- secondary branch after direct CNN scaling issues
- use a CNN-style local syndrome model to assist PyMatching, not replace it

Primary files:

- `tools/build_pymatching_edit_targets.py`
- `decoders/syndrome_edit_predecoder.py`

Status:

- local-edit oracle headroom is strong across d3/d5/d7
- learned edit routing remains weak
- now the most promising model family, provided benefit/harm calibration can
  turn the oracle headroom into selected held-out gains

## 4. Immediate Work Schedule

### Phase 0: Reference And Objective Lock

Timebox:

- current session

Tasks:

- make `Convolutional Neural Decoder for Surface Codes` the explicit main
  paper-format anchor
- mark NVIDIA Ising / AI pre-decoder materials as secondary references for the
  pre-decoder branch only
- keep `RectCNN` as the paper-format baseline
- keep pre-decoder results documented but not mislabeled as the original main
  target

Acceptance:

- `NEXT_SESSION_HANDOFF.md`, `DECODER_RESEARCH_TARGET.md`,
  `PROJECT_REBUILD_STATUS.md`, and `project_status.py` all agree on the target
  hierarchy

### Phase 1: Distance Baseline And Model Selection Refresh

Timebox:

- next 1 focused session

Tasks:

- run or refresh PyMatching class4 baselines on:
  - `artifacts/datasets/dev_class4_2k/manifest.json`
  - `artifacts/datasets/dev_class4_d5_2k/manifest.json`
- `artifacts/datasets/dev_class4_d7_2k/manifest.json`
- run or refresh direct neural candidate checks on the same distance ladder
- run local-edit oracle target checks on the same distance ladder
- run `baseline_rectcnn.py` only as an optional paper-format neural baseline
- record:
  - train/eval families
  - exact CLI commands
  - artifact output paths
  - accuracy, macro-F1, confusion matrix, and held-out `stage_c_corr`

Recommended artifact paths:

- `artifacts/eval/nn/rectcnn_d3_2k_refresh`
- `artifacts/eval/nn/rectcnn_d5_2k_refresh`
- `artifacts/eval/pymatching/d3_2k_class4_refresh`
- `artifacts/eval/pymatching/d5_2k_class4_refresh`
- `artifacts/eval/pymatching/d7_2k_class4_refresh`

Acceptance:

- one table compares PyMatching, best direct neural candidate, and pre-decoder
  oracle headroom on d3/d5/d7 and all four families
- the result is written back into this document and
  `PROJECT_REBUILD_STATUS.md`

Current progress on 2026-04-24:

The PyMatching baseline is complete for the current constructed noise
environment:

| dataset | family | class4 accuracy | frame error rate | LER/cycle | shot errors |
| --- | --- | ---: | ---: | ---: | ---: |
| d3/r3 2k | ideal | 1.000000000 | 0.000000000 | 0.000000000 | 0 |
| d3/r3 2k | stage_a_si1000 | 0.937011719 | 0.062988281 | 0.021945185 | 129 |
| d3/r3 2k | stage_b_local | 0.917968750 | 0.082031250 | 0.028992372 | 168 |
| d3/r3 2k | stage_c_corr | 0.925292969 | 0.074707031 | 0.026257075 | 153 |
| d5/r5 2k | ideal | 1.000000000 | 0.000000000 | 0.000000000 | 0 |
| d5/r5 2k | stage_a_si1000 | 0.907226562 | 0.092773438 | 0.020108317 | 190 |
| d5/r5 2k | stage_b_local | 0.904296875 | 0.095703125 | 0.020800804 | 196 |
| d5/r5 2k | stage_c_corr | 0.899902344 | 0.100097656 | 0.021847101 | 205 |
| d7/r7 2k | ideal | 1.000000000 | 0.000000000 | 0.000000000 | 0 |
| d7/r7 2k | stage_a_si1000 | 0.891113281 | 0.108886719 | 0.017239422 | 223 |
| d7/r7 2k | stage_b_local | 0.868652344 | 0.131347656 | 0.021301097 | 269 |
| d7/r7 2k | stage_c_corr | 0.874511719 | 0.125488281 | 0.020221506 | 257 |

Artifacts:

- `artifacts/eval/pymatching/d3_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d5_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d7_2k_class4_refresh.json`

Direct neural and pre-decoder checks are summarized in
`MODEL_SELECTION_D3_D5_D7.md`. The current decision is that the best next model
family is a PyMatching-assist pre-decoder with benefit/harm calibration, not a
standalone direct CNN replacement and not an exact reimplementation of the
paper architecture.

Latest benefit/harm selector result on 2026-04-24:

- `decoders/syndrome_edit_predecoder.py` now supports
  `--selector-target-mode benefit_harm`
- benefit/harm mode adds logical-transition candidate features
- d3 router1k now selects `local_motif_selector` and improves held-out
  `stage_c_corr 0.928710938 -> 0.939453125`
- d5/d7 still select `global_policy`; their candidate oracle remains high, so
  the next bottleneck is distance-scaled selector calibration

Latest d5 selector ablation status on 2026-04-28:

- transition-prior, transition-top-k, BCE/group-balanced/pairwise
  candidate-compatibility, direct main-selector pairwise ranking,
  motif-evidence merge, and motif-only candidate-pool restriction have all
  been tested as d5 ablations
- `--selector-policy-candidate-mode none` removes raw threshold/top-k policy
  candidates from the selector pool, but selected `stage_c_corr` remains
  `0.888671875 -> 0.888671875`
- the motif-only candidate oracle is still high
  (`stage_c_corr` oracle `0.999023438` with `33.0` candidates/shot), so the
  next bottleneck is candidate representation rather than candidate
  availability
- geometry/placement-aware candidate features are now also implemented with
  `--selector-candidate-geometry-features`, but the d5 motif-only run still
  selects no edit on held-out `stage_c_corr`
- local motif pattern-id / anchor-pattern-aware candidate features are also
  implemented with `--selector-candidate-pattern-features`, but the d5
  motif-only run still selects no edit on held-out `stage_c_corr`
- anchor-local syndrome/evidence candidate features are now implemented with
  `--selector-candidate-local-evidence-features`; this creates sparse nonzero
  emission and improves `stage_b_local` by one shot, but harms held-out
  `stage_c_corr` by one shot, so selected mode remains global no-edit
- next implementation target: learned candidate-conditioned local patch scorer
  for the d5 router1k pre-decoder selector

### Phase 2: Representation Audit

Timebox:

- same session after Phase 1, or next short session

Tasks:

- verify that the rectangular layout matches the paper-format assumptions:
  - invalid cells use the configured incoherent fill value
  - valid mask is available to models that need it
  - row/column/time indices match detector geometry
  - no label or target leakage appears in input channels
- inspect dataset metadata for:
  - `rectangular_syndrome_layout`
  - recommended fill value
  - detector index volume shape
  - valid mask density

Acceptance:

- short audit note added to this document
- any mismatch becomes a concrete code issue before more model tuning

### Phase 3: RectCNN Reproducibility And Seed Check

Timebox:

- 1-2 focused sessions

Tasks:

- rerun RectCNN with at least 3 seeds on d3
- run at least 2 seeds on d5 if runtime is acceptable
- keep architecture close to the paper-format baseline
- avoid mixing this with FLFD/pre-decoder changes

Acceptance:

- mean/std for d3 and d5 RectCNN results
- clear answer to whether RectCNN is stable or seed-sensitive

### Phase 4: Main Extension Decision

Timebox:

- after Phases 1-3

Tasks:

- compare RectCNN against:
  - `factorized_logical_frame_decoder.py`
  - `research_noise_aware_3d.py`
  - existing `multiscale_factorized_decoder.py` results
- decide whether the best direct CNN extension should be:
  - factorized logical-frame head
  - stronger noise-aware 3D CNN
  - hybrid fallback on top of a direct CNN
  - or a return to pre-decoder as the main scaling fix

Acceptance:

- one explicit decision table:
  - paper-format baseline
  - best direct CNN extension
  - PyMatching
  - pre-decoder oracle and learned result

### Phase 5: Pre-Decoder Gate

Timebox:

- only after the paper-format baseline refresh is complete

Tasks:

- keep pre-decoder work gated by the following question:
  - does it produce learned held-out improvement over PyMatching or only oracle
    headroom?
- if continuing:
  - test seed reproducibility of the small d3 non-identity selected-policy gain
  - add benefit/harm calibration for nonzero edit selection

Acceptance:

- pre-decoder is either promoted back to an active main extension with learned
  held-out gains or kept as a documented secondary branch

## 5. Next Concrete Command-Level Task

The distance-scaled selector-calibration pass has now been tried on d5.
Nonzero-bias calibration, harm-margin loss, stronger hard-shot selector
weighting, and an `oracle_solvable` hard-shot router all preserve safety but do
not unlock d5 held-out improvement.

The first target-transition-prior selector is also now implemented and tested:

- artifact:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_transprior/experiment_summary.json`
- selected d5 behavior remains no-edit / `global_policy`
- held-out `stage_c_corr` remains
  `0.888671875 -> 0.888671875`
- forced margin-0 emission gives
  `0.888671875 -> 0.879882812`, improved `50`, harmed `59`

The next work item should therefore be an edit-validity or candidate-target
compatibility constraint, not another scalar threshold/margin/prior-weight
sweep.

The first hard transition-compatibility gate is also now implemented and
tested:

- artifact:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_compat_topk/experiment_summary.json`
- selected candidate-selector behavior remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- forced margin-0 top-k gate:
  - top-k `1/2/4`: suppresses all edits
  - top-k `8`: `0.888671875 -> 0.878906250`, improved `34`, harmed `44`

This shows that the current shot-level transition prior is too coarse. The next
work item should be candidate-level beneficial-vs-harmful transition
compatibility, not a wider top-k gate.

The first flat BCE candidate-compatibility head is also now implemented and
tested:

- artifact:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat/experiment_summary.json`
- selected candidate-selector behavior remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- forced threshold sweep shows thresholds `0.1..0.9` do not alter the harmful
  selected edits
- validation positive rate is only about `1-1.5%`, while the BCE head predicts
  about `23%` positives

This shows that flat BCE compatibility is not calibrated enough. The next work
item should be group-balanced or pairwise beneficial-vs-harmful compatibility.

The first group-balanced candidate-compatibility head is also now implemented
and tested:

- artifact:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_groupbal/experiment_summary.json`
- selected candidate-selector behavior remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- checkpoint diagnostic:
  true positive rate `~1-1.7%`, predicted positive rate `~0.1-0.2%`
- forced margin `0`, threshold `0`:
  `0.888671875 -> 0.880859375`, improved `48`, harmed `56`

This shows that group-balanced BCE overcorrects and becomes too conservative.
The next work item should be pairwise beneficial-vs-harmful ranking within each
shot group, not another absolute compatibility threshold.

The first auxiliary pairwise compatibility ranking head is also implemented and
tested:

- artifact:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_pairwise/experiment_summary.json`
- selected candidate-selector behavior remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- forced margin `0`, top-k `0/1/2/4/8` all give:
  `0.888671875 -> 0.879882812`, improved `52`, harmed `61`

This shows that a detached auxiliary ranking gate is still insufficient. The
next work item should merge beneficial-vs-harmful ranking directly into the
main selector objective.

The first direct main-selector pairwise benefit/harm term is also implemented
and tested:

- artifacts:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise/experiment_summary.json`
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise_margin15/experiment_summary.json`
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise_w16/experiment_summary.json`
- selected behavior remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- one forced sweep from the weight-`1` checkpoint showed a narrow non-selected
  positive band at margin `1.5`:
  `stage_c_corr 0.888671875 -> 0.889648438`
- the band did not reproduce as selected behavior after adding margin `1.5` to
  validation, and weight `16` still collapses to no-edit at selected margins

This shows that selector-side calibration is likely exhausted for d5. The next
work item should restrict or enrich the candidate set itself.

The first candidate-representation ablation is now implemented and tested:

- duplicate policy/motif candidates now merge motif evidence into the candidate
  feature row
- artifact:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifmerge_pairwise/experiment_summary.json`
- selected behavior remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- forced low-margin held-out result:
  `stage_c_corr 0.888671875 -> 0.881835938`, improved `52`, harmed `59`

This shows that motif provenance alone is insufficient. The next work item
should restrict raw policy candidates or use transition-conditioned motif
candidates.

The d3 reproducibility gate is now complete. New seeds `1,2,3` all select
`local_motif_selector` and improve held-out `stage_c_corr`; mean delta over
those seeds is `+0.010091146`.

Useful target manifests:

```text
artifacts/datasets/predecoder_targets_d3_2k_router1k/manifest.json
artifacts/datasets/predecoder_targets_d5_2k_router1k/manifest.json
artifacts/datasets/predecoder_targets_d7_2k_router1k/manifest.json
```

The concrete design target is:

- input: syndrome/noise-aware volume plus generated local-edit candidates
- output: choose identity or one small local edit whose edited PyMatching
  logical class passes an explicit target-compatibility / validity test
- objective: rank edits by actual PyMatching benefit minus harm while directly
  comparing beneficial and harmful nonzero candidates within hard-shot groups
- guardrail: selected mode must improve held-out `stage_c_corr` over raw
  PyMatching, not just improve an oracle or train-family metric

Do not spend the next iteration only changing:

- `selector_emit_margin`
- `selector_nonzero_bias`
- `selector_hard_shot_weight`
- `selector_harm_margin_loss_weight`
- `selector_transition_prior_weight`

Those knobs have already been checked on d5 and did not close the gap.

## 6. Stop Conditions

Do not spend the next iteration copying the exact target-paper CNN structure.
The reference hierarchy must stay clear:

- main paper contract: logical-frame output and PyMatching comparison
- optional baseline: RectCNN over rectangular syndrome lattice
- direct extension: stronger CNN logical-frame decoders, currently negative at
  d5/d7
- leading extension: neural pre-decoder plus PyMatching, because oracle
  headroom survives d3/d5/d7
