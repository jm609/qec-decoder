# Decoder Research Target

This document fixes the problem statement for the next decoder line.
It is intended to guide architecture design, dataset work, evaluation,
and milestone planning.

## Why This Document Exists

The project is no longer blocked on basic infrastructure:

- `sample_dataset.py` can now generate per-shot `logical_class4` labels via
  `--target-mode bell_pair_z_readout`.
- `decoders/baseline_pymatching.py` can evaluate the class4 path.
- `decoders/baseline_rectcnn.py` and `decoders/research_noise_aware_3d.py`
  can train and evaluate on `logical_class4`.
- `decoders/factorized_logical_frame_decoder.py` now implements the first
  mainline factorized logical-frame model with class4 supervision and optional
  auxiliary axis supervision.
- `ideal`, `stage_a_si1000`, `stage_b_local`, and `stage_c_corr` are already
  available for smoke experiments.

Because of that, the next blocker is not infrastructure. The next blocker is
the research target itself: what the new decoder should optimize for.

The detailed execution schedule is now fixed in:

- `RESEARCH_PLAN_PREDECODER_MAIN.md`
- `MAIN_TARGET_WORK_SCHEDULE.md`

`RESEARCH_PLAN_PREDECODER_MAIN.md` should be treated as the current primary
research plan. `MAIN_TARGET_WORK_SCHEDULE.md` remains the broader schedule and
context map.

## Current Context

The old project target was effectively:

- input: detector events
- output: one binary logical flip label
- success criterion: label accuracy

That target is now too weak.

The rebuilt mainline has already moved to a stronger supervision path:

- input: geometry-aware syndrome volume
- target: per-shot logical Pauli frame
- label space: `logical_class4 in {I, X, Z, Y}`

The repository now has the first implementation of the new decoder line.
However, the surrounding research problem is still larger than one model file:
calibration, fallback, recovery-level evaluation, and broader noise transfer
work remain open.

## Latest Empirical Lesson

The most important recent result is that the first FLFD stagnation was not yet
strong evidence that the factorized architecture should be discarded.

What changed:

- the repository now has a larger class4 dataset at
  `artifacts/datasets/dev_class4_2k`
- training families now contain nontrivial `X`, `Z`, and `Y` support instead of
  the earlier near-empty minority-class smoke regime
- a smaller FLFD variant trained on that dataset produces meaningful non-`I`
  predictions instead of collapsing to all-`I`

What that showed:

- mixed-test `macro_f1` and balanced accuracy become meaningfully nontrivial
- holdout `stage_c_corr` also improves in meaningful metrics
- threshold-based hybrid evaluation becomes informative instead of degenerating
  immediately into a smoke-level artifact

Operational conclusion:

- the immediate bottleneck is still data regime, class distribution, and stable
  model scale
- a full architecture reset is not yet justified
- the correct next comparisons should be made after larger class4 runs, using
  macro-F1, balanced accuracy, and holdout behavior rather than raw accuracy

Follow-up result:

- that conclusion holds at `d3`, but not automatically at larger distance
- the first `d5/r5` rerun with `2048` shots per family shows that the current
  FLFD family still collapses badly despite nontrivial minority-class support
  in the train families
- PyMatching remains strong there, with roughly `0.90` class4 accuracy while
  the current FLFD variants stay in the degenerate regime

Updated operational conclusion:

- `d3` established that smoke collapse was not enough evidence for a full reset
- `d5` now establishes that the current FLFD line does have a real
  distance-scaling limitation
- the next architecture work should therefore target scaling behavior on larger
  syndrome volumes, not just more loss-level tweaks on the same backbone

Further update:

- the first direct successor candidate, `multiscale_factorized_decoder.py`,
  has now been implemented and tested on the larger `d3` and `d5` class4
  benchmarks
- in its first large runs it did not beat the smaller original FLFD on `d3`
  and did not fix the `d5` collapse
- a stronger `d5` M3D-FLFD configuration collapsed even harder

New implication:

- the repository now has direct evidence that a simple multi-scale dense direct
  decoder is not, by itself, the missing step
- the next architecture move should therefore prioritize a qualitatively
  different system direction, most likely an Ising-inspired neural pre-decoder
  plus PyMatching final decoder

Current fixed research topic:

> Transition-aware neural pre-decoding for surface-code logical-frame inference.

NVIDIA Ising Decoding is now treated as related work that supports the broader
neural pre-decoding direction. It is not a blueprint to copy. The repo-specific
contribution is benefit/harm and transition-aware candidate selection under
Bell-pair `logical_class4` supervision and staged noise-family evaluation.

## Main Research Position

The new decoder should not be defined as:

- "predict the physical error chain"
- "beat PyMatching on binary label accuracy"
- "fit one fixed simulated noise model better than a baseline"

The new decoder should be defined as:

- a calibrated logical-frame inference model
- robust to noise-model shift
- compatible with latency constraints
- evaluated by final logical failure, not only by class accuracy

In one sentence:

> The new decoder should estimate the logical Pauli frame under realistic and
> shifted noise, with useful confidence, low latency, and lower final logical
> failure than strong classical baselines.

## Primary Objective

The primary objective is:

- learn `P(L | syndrome, noise_context)` where `L in {I, X, Z, Y}`

This means the model output is not just a hard class. The output should be a
posterior over the logical frame.

Why this is the right target:

- it matches the current per-shot class4 label path
- it is directly comparable with classical multi-observable decoding
- it supports confidence-aware fallback and hybrid decoding
- it is closer to the practical decoding decision than physical-chain
  reconstruction

## Secondary Objectives

The new decoder should also aim to provide:

- confidence or uncertainty estimates
- robustness to unseen noise families
- graceful degradation under mismatch between train and eval noise
- efficient inference for future streaming or real-time paths

These are not optional extras. They are part of the research target because
recent decoder work is no longer judged only by in-distribution accuracy.

## Final Evaluation Objective

The final objective is not:

- class accuracy alone
- cross-entropy alone
- AUROC alone

The final objective is:

- logical failure after applying a decoder decision

Near-term proxy:

- `logical_class4` accuracy
- macro-F1
- calibration error
- OOD gap across noise families

End-state metric:

- logical error rate per shot or per cycle after decoding
- threshold and scaling behavior
- latency under deployment constraints

This implies the project should eventually move from:

- "classifier evaluation"

to:

- "decoder decision quality"

and finally to:

- "recovery-system quality"

## Research-Worthy Problem Statement

The most valuable problem statement for this repository is:

> Design a neural decoder that predicts the logical Pauli frame from a
> space-time syndrome representation, remains reliable under unseen noise
> conditions, exposes confidence for selective fallback, and improves final
> logical error relative to strong classical baselines.

This is worth studying because it sits at the intersection of:

- realistic hardware noise
- distribution shift
- neural decoder calibration
- hybrid classical-neural decoding
- eventual real-time deployment pressure

## Recommended Objective Hierarchy

The work should be organized in the following order.

### Level 1: Logical Frame Posterior

Train the model to output class4 posterior probabilities.

Required outputs:

- class logits for `I`, `X`, `Z`, `Y`
- normalized posterior probabilities

Required questions:

- does the model learn more than the class prior?
- which classes collapse first under imbalance?
- how stable is the posterior under noise shift?

### Level 2: Confidence-Aware Decoding

The decoder should expose when it is uncertain.

Required outputs:

- max probability
- margin between top-2 classes
- entropy or another uncertainty score

Required questions:

- is confidence calibrated?
- can low-confidence shots be isolated reliably?
- how much performance is gained by abstention or fallback?

### Level 3: Hybrid Decoder

The project should test a selective hybrid:

- neural decoder on confident shots
- PyMatching or another classical decoder on uncertain shots

This is a high-value target because it aligns with practical deployment:

- classical baselines remain strong
- neural decoders may win selectively
- latency and accuracy can be traded explicitly

The repository now has a first frame-level hybrid evaluator:

- `tools/evaluate_hybrid_fallback.py`

It currently compares calibrated FLFD confidence against PyMatching fallback on
the same class4 manifests. This is still a decision-level evaluator, not yet a
recovery-level hybrid system.

The repository now also has a first learned post-hoc router:

- `tools/evaluate_learned_hybrid_router.py`

This tool keeps FLFD frozen, trains a small logistic correctness router on
frozen FLFD and PyMatching outputs from seen families, and then evaluates hybrid
routing on the same class4 manifests. Current smoke runs are encouraging in one
specific sense: they no longer collapse to fallback-all on seen families.
Instead, the learned router preserves seen-family accuracy while keeping a small
amount of neural coverage. However, the same smoke runs do not yet improve the
holdout `stage_c_corr` result beyond plain PyMatching, so this should be viewed
as a better hybrid research direction, not as a solved final policy.

The same learned-router tool now also supports a direct `prefer_neural` target
mode, where the router learns whether choosing the neural decoder is at least as
good as choosing PyMatching on a given shot. This is better aligned with the
current threshold-selection objective. However, current smoke runs still produce
the same practical outcome as the correctness-target version: partial neural
coverage on seen families, but no holdout improvement beyond PyMatching on
`stage_c_corr`.

The learned-router path now also includes explicit metadata and noise-context
features derived from the same family-level summaries used by the main
noise-aware decoder line. This makes the router structurally more consistent
with the project-wide noise-transfer objective. However, current smoke runs
still show the same limitation: the added context does not yet improve holdout
`stage_c_corr` routing beyond plain PyMatching.

The repository also now has a first tempered imbalance-aware FLFD training path:

- `decoders/factorized_logical_frame_decoder.py --imbalance-mode tempered`

This now adds tempered class4 reweighting, tempered main-task sampling, and a
lighter auxiliary-axis treatment so the model can start to address strong
logical-class imbalance without the collapse reversal caused by aggressive
balancing. The immediate research bottleneck is now confidence usefulness and
minority-class separation rather than infrastructure.

The FLFD line also now supports an optional focal-style class4 objective:

- `decoders/factorized_logical_frame_decoder.py --main-class4-loss focal`

Current smoke runs suggest that this objective alone does not materially improve
hybrid usefulness yet, so it should be treated as an available training option,
not as the solved final recipe.

The FLFD line also now supports an optional hierarchical non-identity auxiliary
objective:

- `decoders/factorized_logical_frame_decoder.py --non-identity-loss-weight ...`

This adds a dedicated `I vs non-I` head and associated loss/metrics. Current
smoke runs show that the first implementation learns the coarse non-identity
signal but can destabilize the fine class4 decision, so it should remain an
experimental option rather than a default setting.

The FLFD line also now supports an experimental joint confidence-loss head:

- `decoders/factorized_logical_frame_decoder.py --confidence-loss-weight ...`

Current smoke runs suggest that even small joint confidence-loss weights can
destabilize the main class4 task, so the preferred hybrid direction is now
post-hoc learned routing rather than joint correctness-head training.

### Level 4: Latency-Constrained Variant

A later model line should optimize for constrained inference.

Possible targets:

- short-window decoding
- streaming updates
- reduced-context inference
- small-model deployment version

This should not block the first new architecture, but it must shape the design.

## Explicit Non-Goals

The next decoder should not be optimized for the following as a primary goal:

- exact physical error reconstruction
- one-noise-family benchmark wins only
- binary `logical_label` accuracy only
- architecture novelty without evaluation novelty
- treating full paper reproduction of the original CNN decoder as the final
  research endpoint

The original CNN paper is the main format anchor for the first neural decoder
baseline:

- Jung, Ali, Ha, "Convolutional Neural Decoder for Surface Codes",
  IEEE Transactions on Quantum Engineering, 2024
- DOI: `10.1109/TQE.2024.3419773`

That is why the repository built:

- rectangular syndrome lattice tensors
- incoherent fill values for missing cells
- the `baseline_rectcnn.py` paper-style CNN baseline

It is not the final research endpoint because later `d5` and noise-shift
experiments pushed the project toward stronger factorized / pre-decoder
extensions.

## Why Recent Research Supports This Direction

The following trends motivate the objective above:

- modern decoder work focuses on final logical performance, not only label
  prediction
- realistic experiments expose mismatch between simulated and deployed noise
- confidence and prior calibration matter materially
- fast classical decoders remain strong, so hybrid systems are credible
- deployment latency is now a first-class constraint in hardware-facing work

For this repository, that means the meaningful research target is not "replace
PyMatching everywhere" but "build a neural decoding component that is useful
under realistic constraints."

## Project-Specific Implications

Given the current repository state, the new decoder should:

- consume the rectangular or volume-based syndrome representation
- optionally use explicit noise-context channels
- predict `logical_class4`
- expose uncertainty
- be evaluated on both seen and unseen noise families

The first new model does not need to solve everything at once, but it should be
designed so that these evaluation axes are natural.

## Recommended Experiment Axes

Every serious experiment should report at least the following:

- in-distribution class4 accuracy
- macro-F1 and confusion matrix
- held-out family performance
- calibration or confidence summary
- comparison against PyMatching on the same dataset

As the project advances, add:

- fallback policy results
- recovery-based logical failure
- latency measurements
- scaling across distance and rounds

## Concrete Research Questions

The next architecture should be built to answer these questions.

1. Can the model estimate the logical class4 posterior better than simple
   geometry-aware baselines?
2. Does explicit noise context improve generalization to unseen noise families?
3. Can uncertainty identify when the neural decoder should defer to a classical
   decoder?
4. Under a fixed latency budget, what is the best achievable logical failure?

If a proposed architecture does not help answer at least one of these, it is
probably not worth prioritizing.

## Phased Development Guidance

### Phase A: Stronger Class4 Neural Baseline

Goal:

- replace the current baseline-quality class4 model with a serious candidate

Must-have:

- class4 posterior output
- multiclass training objective
- seen vs unseen family reporting

### Phase B: Calibration And Hybrid Control

Goal:

- make model confidence operational

Must-have:

- calibration analysis
- fallback threshold experiments
- neural-only vs hybrid comparison

### Phase C: Recovery-Level Evaluation

Goal:

- move from label metrics to decoding metrics

Must-have:

- map posterior decision to a recovery action
- compare final logical failure against classical baselines

### Phase D: Latency-Oriented Variant

Goal:

- prepare the path toward deployable decoding

Must-have:

- reduced-compute inference path
- latency measurement
- accuracy-latency tradeoff curve

## Repository Rule Going Forward

From this point onward, any new decoder proposal should be justified using:

- logical-class4 relevance
- robustness to noise shift
- calibration or fallback usefulness
- a path toward recovery-level evaluation

If a proposal cannot be connected to those four points, it should not become
mainline work.

## Practical Summary

The decoder target for this repository is now:

- not binary logical-flip prediction
- not physical-chain reconstruction
- not pure benchmark recreation

It is:

- class4 logical-frame inference
- under noise shift
- with confidence
- compared against strong classical baselines
- on the path to final logical-failure evaluation

## Immediate Next Step

The next architecture-design document should start from this problem statement
and answer:

- what information the model sees
- what it predicts exactly
- how confidence is represented
- how fallback is triggered
- how evaluation progresses from class4 metrics to final recovery metrics

Status update:

- the first concrete architecture spec is now recorded in
  `DECODER_ARCHITECTURE_SPEC_V1.md`
- backbone option comparison and the recommended execution order are now
  recorded in `DECODER_ARCHITECTURE_OPTIONS.md`
- the next narrowed architecture decision, including the selected immediate
  successor to FLFD and the deferred Ising-inspired hybrid branch, is now
  recorded in `DECODER_ARCHITECTURE_SHORTLIST_V2.md`
- the first pre-decoder system spec is now recorded in
  `PREDECODER_ARCHITECTURE_SPEC_V1.md`
- the first derived-target builder for that branch now exists as
  `tools/build_pymatching_edit_targets.py`
- bounded `k<=2` pilot runs on the larger `d3` and `d5` class4 datasets now
  show strong local-edit oracle headroom over raw PyMatching, which is enough
  evidence to justify training the first syndrome-edit pre-decoder model
- that first pre-decoder model now exists in
  `decoders/syndrome_edit_predecoder.py`, but the initial pilot recipe is not
  yet competitive: with safe accuracy-first policy selection the `d3` pilot
  falls back to the identity no-edit policy, while the same recipe on `d5`
  still over-edits and harms baseline PyMatching
- the first hard-shot weighted-sampling follow-up also now exists in the same
  pre-decoder branch, but on current pilots it still converges to the same
  safe no-edit policy instead of realizing the local-edit oracle headroom
- the first decision-aware follow-up now also exists in the same pre-decoder
  branch as a candidate-edit selector layer over the current threshold / top-k
  candidate set; the first smoke run confirms that the selector plumbing,
  checkpointing, and eval outputs work
- the first real `d3` / `d5` pilot reruns of that selector path now also exist,
  and they still do not show a real holdout or system-level gain over the safe
  global-policy baseline: validation still selects `global_policy`, final eval
  stays at raw PyMatching accuracy, and the selector still chooses zero edits
  in practice
- the first in-training decision-aware follow-up now also exists as an
  identity-vs-target pairwise ranking loss on solved hard shots, and real
  `d3` / `d5` pilot reruns with that loss still remain in the same safe
  no-edit regime
- the first stronger selector-training follow-up now also exists as a per-shot
  group-ranking selector objective over the generated candidate set, and real
  `d3` / `d5` pilot reruns with that objective still remain in the same safe
  no-edit regime
- the first explicit edit-validity-structured follow-up now also exists as a
  motif-vocabulary selector over observed hard-shot edit masks, and real
  `d3` / `d5` pilot reruns with that path still remain in the same safe
  no-edit regime
- the first structured candidate-pool follow-up now also exists as a
  motif-augmented selector candidate set over observed hard-shot edit masks,
  and real `d3` / `d5` pilot reruns with that stronger pool still remain in
  the same safe no-edit regime
- the first explicit selector-side identity-vs-nonzero follow-up now also
  exists as a margin loss that pushes the best validated nonzero candidate
  above identity inside selector training, and real `d3` / `d5` pilot reruns
  with that loss still remain in the same safe no-edit regime
- the first action-path structured-action follow-up now also exists as a
  motif-action competition loss directly on `edit_logits + needs_edit_logits`,
  and real `d3` / `d5` pilot reruns with that loss still do not unlock
  holdout gains, though they do appear to stabilize the earlier `d5`
  over-editing failure back to baseline-level behavior
- the first action-motif inference follow-up now also exists as a direct
  structured action emit path with a validation-selected emit margin; real
  `d3` / `d5` pilot reruns show that this can finally emit nonzero actions and
  improve seen-family `d3` eval, but it still fails to improve holdout
  `stage_c_corr` and is suppressed to identity on `d5`
- the first local/generalizable action follow-up now also exists as
  `selection_mode = local_motif`; it builds relative `(dt, dr, dc)` hard-shot
  edit patterns, expands them over every valid detector-coordinate anchor, and
  applies a validation-selected `local_motif_min_bit_logit` edit-validity gate
- real gated `d3` / `d5` local-motif reruns still select `global_policy`;
  `d3` local inference improves `stage_b_local` by one shot
  (`0.89453125 -> 0.8984375`) but leaves holdout `stage_c_corr` unchanged,
  while `d5` stays unchanged under the selected local gate
- the first local-motif selector follow-up now also exists as
  `selection_mode = local_motif_selector`; it trains the existing
  `CandidateEditSelector` over top-k local placement candidates labeled by
  actual PyMatching correctness after applying each candidate edit
- real local-motif selector pilots show that candidate generation is no longer
  the bottleneck: the `d3` local placement candidate oracle reaches `1.0` on
  all eval families, and `d5 stage_c_corr` candidate oracle is `0.99609375`
- the learned selector still fails to calibrate safe emission: default `d3`
  remains identity-only, strong hard-shot `d3` over-emits and trades one
  `stage_b_local` improvement for three `stage_a_si1000` harms, guarded `d3`
  returns to identity-only, and guarded `d5` still selects `global_policy`
- the first factorized hard-shot router follow-up now also exists as
  `selection_mode = local_motif_router`; it trains a shot-level router from
  system-level local-candidate labels, then allows local selector nonzero
  actions only on routed shots
- real router pilots on `d3` and `d5` still select `global_policy`; the router
  collapses to all-route or no-route behavior in the current 256-shot pilot
  regime, with d3 validation route-positive rates only around `5-10%`
- a larger same-recipe target rerun is now also complete:
  `artifacts/datasets/predecoder_targets_d3_2k_router1k/manifest.json` and
  `artifacts/datasets/predecoder_targets_d5_2k_router1k/manifest.json`
  process `1024` shots per family with the same local edit search config
- those 1024-shot manifests preserve oracle headroom, including
  `d3 stage_c_corr 0.9287109375 -> 0.9921875` and
  `d5 stage_c_corr 0.888671875 -> 0.978515625`
- learned decoding on the matching router1k runs still does not improve:
  `artifacts/eval/nn/sedp_d3_router1k_localmotif_router_feat/experiment_summary.json`
  and
  `artifacts/eval/nn/sedp_d5_router1k_localmotif_router_feat/experiment_summary.json`
  both select `global_policy`, route no eval shots, and leave `stage_c_corr`
  unchanged at the raw PyMatching baselines
- the router branch now also supports baseline-failure / oracle-solvability
  pretraining and balanced route minibatches through:
  `--router-pretrain-target`, `--router-pretrain-epochs`,
  `--router-pretrain-pos-weight`, and `--router-negative-ratio`
- real balanced/pretrained router runs are complete:
  `artifacts/eval/nn/sedp_d3_router1k_router_pretrain_balanced/experiment_summary.json`
  and
  `artifacts/eval/nn/sedp_d5_router1k_router_pretrain_balanced/experiment_summary.json`
- those runs still select `global_policy`, route no eval shots, and leave
  selected `stage_c_corr` unchanged; however, the d3 action-motif eval path in
  that run improves `stage_c_corr` slightly from `0.9287109375` to
  `0.931640625`
- focused action-motif selected-mode runs are now also complete:
  `artifacts/eval/nn/sedp_d3_router1k_actionmotif_selected/experiment_summary.json`
  and
  `artifacts/eval/nn/sedp_d5_router1k_actionmotif_selected/experiment_summary.json`
- the d3 run selects a non-identity `global_policy` rather than direct
  `action_motif` inference and improves `stage_c_corr` from `0.9287109375` to
  `0.9306640625`; the d5 run remains identity/no-edit
- the next pre-decoder step should therefore not be more sampling-only work;
  it should now be a stronger change that affects edit generation itself or
  imposes action-level edit-validity structure plus, if needed, another
  decision-aware objective tied more directly to hard-shot routing and
  calibrated local nonzero emission inside that local/generalizable action path;
  the immediate practical next step is now to check whether the small d3
  non-identity selected-policy gain is reproducible across seeds, then move to
  benefit/harm-calibrated nonzero edit selection if it is not robust

## Current Mainline Pipeline

This section describes the current end-to-end repository pipeline so that
future decoder work can plug into the right stage instead of duplicating
infrastructure.

### 1. Circuit And Noise Definition

The pipeline starts from the circuit and noise generators:

- `circuits.py`
- `noise_si1000.py`
- `noise_willowcore.py`
- `config.py`

What they do:

- build the ideal rotated surface-code memory scaffold
- attach noise families such as `ideal`, `stage_a_si1000`,
  `stage_b_local`, and `stage_c_corr`
- expose detector metadata needed later for geometry-aware learning

Important limitation:

- the raw single-basis memory scaffold does not natively expose same-shot
  logical X and Z information together
- this is why the repository added the Bell-pair-based class4 supervision path

### 2. Logical-Frame Feasibility Audit

Before designing new labels, the repository can check whether a circuit path
structurally supports true same-shot logical-frame supervision.

Relevant files:

- `logical_frame.py`
- `tools/audit_logical_frame_support.py`

What this stage does:

- infers the logical support structure of the current scaffold
- checks whether both logical observables are deterministic in the same shot
- records why the raw single-basis scaffold is insufficient for true class4

This stage is important because it prevents dishonest supervision design.

### 3. Label Path Selection

The dataset pipeline currently supports two label modes.

#### Mode A: Legacy Single-Basis Supervision

Implemented through:

- `sample_dataset.py`
- `logical_targets.py`

What it produces:

- `logical_label`
- `logical_axis_flip`
- one measured logical observable per shot

Use case:

- axis-wise experiments
- dual-axis pairing of separate x-basis and z-basis datasets

Limitation:

- not true per-shot class4

#### Mode B: Bell-Pair Class4 Supervision

Implemented through:

- `logical_bell.py`
- `sample_dataset.py --target-mode bell_pair_z_readout`

What it produces in one shot:

- `logical_x_flip`
- `logical_z_flip`
- `logical_class4`

Mapping:

- `0 = I`
- `1 = X`
- `2 = Z`
- `3 = Y`

This is currently the strongest truthful supervision path in the repository.

### 4. Dataset Materialization

The central dataset writer is:

- `sample_dataset.py`

It saves, per family:

- `samples.npz`
- `metadata.json`
- `circuit.stim`
- `detector_error_model.dem`

Within `samples.npz`, the main arrays can include:

- `detector_events`
- `observable_flips`
- `logical_label`
- `logical_axis_flip`
- `logical_x_flip`
- `logical_z_flip`
- `logical_class4`

The manifest produced at the dataset root is the handoff point for all decoder
training and evaluation.

### 5. Geometry Conversion

The raw detector stream is not kept only as a flat vector.

Relevant files:

- `geometry/rotated_rect.py`
- `decoders/baseline_rectcnn.py`

What this stage does:

- maps detector coordinates onto a rectangular time-space grid
- stores rectangular layout metadata in the dataset
- supports incoherent fill values for missing lattice cells

This is the canonical geometry-aware input representation for the current
mainline.

### 6. Experimental Branching

After dataset generation, the pipeline splits into two experiment branches.

#### Branch A: Axis-Wise Branch

Relevant files:

- `dual_axis_manifest.py`
- `tools/build_dual_axis_manifest.py`
- `tools/run_dual_axis_experiment.py`
- `tools/run_dual_axis_pymatching.py`

How it works:

- generate separate x-basis and z-basis datasets
- pair them into one `dual_axis_manifest`
- train and evaluate logical-x and logical-z predictors separately
- compare neural and classical baselines on aligned splits

This branch remains useful for controlled axis-wise studies, but it is not the
end-state decoder problem.

#### Branch B: Per-Shot Class4 Branch

Relevant files:

- `sample_dataset.py --target-mode bell_pair_z_readout`
- `decoders/baseline_pymatching.py`
- `decoders/baseline_rectcnn.py --target-mode logical_class4`
- `decoders/research_noise_aware_3d.py --target-mode logical_class4`

How it works:

- generate one manifest where each shot already has a class4 logical-frame label
- evaluate classical and neural baselines directly on the same manifest
- report class4 metrics and cross-family generalization

This is the branch that future decoder design should target first.

### 7. Active Baselines And New Decoder Line

The current active comparison models are:

- `decoders/baseline_pymatching.py`
- `decoders/baseline_rectcnn.py`
- `decoders/research_noise_aware_3d.py`
- `decoders/factorized_logical_frame_decoder.py`
- `decoders/multiscale_factorized_decoder.py`

Their roles are different.

#### PyMatching

Role:

- main classical comparison baseline

Current capabilities:

- uses `detector_error_model.dem`
- supports single-observable and two-observable datasets
- reports class4-aware metrics on Bell-pair datasets

Current limitation:

- still reports frame and label quality, not final recovery-system quality

#### Factorized Logical-Frame Decoder

Role:

- first new mainline neural decoder architecture for this repository

Current capabilities:

- predicts `logical_class4`
- explicitly models `logical_x_flip` and `logical_z_flip` through factorized
  heads
- supports optional auxiliary supervision from the `single_basis` branch via a
  dual-axis manifest
- runs as single-family training, checkpoint evaluation, and multi-family
  manifest experiment

Current limitation:

- recovery-level evaluation is still missing
- the first larger run suggests that data scale and model size matter more than
  an immediate backbone reset
- the present mainline question is how far the smaller stable FLFD line can be
  pushed before a new architecture family is justified

#### Multi-Scale Factorized Decoder

Role:

- first direct successor architecture to the original FLFD line

Current capabilities:

- reuses the current truthful class4 training and evaluation stack
- replaces the shallow trunk with a multi-scale Dense3D encoder
- preserves the factorized logical-frame output structure
- already runs end-to-end through the existing train/eval/experiment CLI path

Current limitation:

- only smoke-level validation has been run so far
- it has not yet been benchmarked on the larger `d3` and `d5` class4 runs that
  now define the architecture-selection problem

#### RectCNN

Role:

- lightweight geometry-aware neural baseline

Current capabilities:

- consumes rectangular syndrome tensors
- supports binary and class4 target modes
- provides a low-cost baseline for fast iteration

Current limitation:

- still a baseline model, not a strong research candidate

#### NoiseAware3D

Role:

- current main research backend

Current capabilities:

- consumes 3-D syndrome volumes
- adds static geometry planes and noise-context planes
- supports binary and class4 targets
- supports multi-family train/eval experiments

Current limitation:

- still baseline-quality and not yet based on a new decoder idea

### 8. Evaluation Flow

The evaluation stack currently has three layers.

#### Layer 1: Label Metrics

Already implemented:

- accuracy
- label error rate
- macro-F1
- confusion matrix
- per-family evaluation

This is the current mainline for class4 experiments.

#### Layer 2: OOD And Cross-Noise Evaluation

Already implemented:

- train on selected families
- evaluate on held-out families
- compare seen vs unseen noise behavior

This is where `decoders/research_noise_aware_3d.py experiment` is currently
most useful.

#### Layer 3: Recovery-Level Evaluation

Not implemented yet:

- turn model decisions into a recovery action
- compare final logical failure after recovery
- compare hybrid fallback policies at the recovery level

This is the next major evaluation gap.

### 9. Artifact Outputs

The repository writes outputs mainly under `artifacts/`.

Typical locations:

- datasets: `artifacts/datasets/...`
- neural evals: `artifacts/eval/nn/...`
- classical evals: `artifacts/eval/pymatching/...`
- reports: `artifacts/reports/...`

This separation should be preserved. New decoder lines should not introduce a
parallel ad hoc output layout.

## Current Pipeline Summary

The current mainline is best read as:

1. define circuit and noise
2. verify logical-frame support assumptions if needed
3. choose supervision mode
4. generate datasets and manifests
5. convert detector data into geometry-aware tensors
6. run classical and neural baselines
7. compare seen/noise-shifted performance
8. use that as the starting point for a new decoder architecture

In compact form:

`circuits/noise -> sample_dataset -> manifest -> geometry-aware tensors ->
baseline comparison -> new decoder design`

For axis-wise work:

`sample_dataset(single_basis) -> dual_axis_manifest -> axis-wise baselines`

For class4 work:

`sample_dataset(bell_pair_z_readout) -> class4 manifest -> class4 baselines`

## What Is Already Stable Enough To Reuse

The following should be treated as stable reusable infrastructure:

- dataset generation
- Bell-pair class4 supervision path
- rectangular geometry representation
- active baseline runners
- class4-level hybrid fallback runner
- `ideal/A/B/C` smoke-level noise coverage
- artifact output conventions

These should be reused, not redesigned, unless a specific limitation is proven.

## What Is Still Missing In The Pipeline

The following are still open gaps:

- stronger decoder architectures beyond the first FLFD line
- recovery-action evaluation
- final logical failure benchmarking after recovery
- latency-aware deployment path
- new noise families beyond the current `ideal/A/B/C`
- native same-shot logical-frame circuits without Bell-pair scaffolding

## Rule For Future Work

Any future decoder implementation should enter the repository through the
current class4 branch unless there is a strong reason not to.

That means new work should usually plug into:

- `sample_dataset.py --target-mode bell_pair_z_readout`
- the existing manifest format
- the rectangular or volume-based input builders
- the current baseline comparison stack

This keeps architecture research comparable and avoids another project reset.

## Project Status And Roadmap

This section fixes the current repository state and the next execution order.
It should be treated as the operational summary for future work.

### A. What Has Already Been Completed

#### A1. Mainline Cleanup

Completed:

- the old flat and track-based experimental decoder files were moved out of the
  active mainline into `legacy_archive/`
- the active decoder surface is now limited to:
  - `decoders/baseline_pymatching.py`
  - `decoders/baseline_rectcnn.py`
  - `decoders/research_noise_aware_3d.py`
  - `decoders/factorized_logical_frame_decoder.py`

Why this matters:

- future decoder work now has a clean comparison surface
- new experiments do not need to depend on legacy track code

#### A2. Geometry-Aware Dataset Path

Completed:

- detector streams can be mapped into a rectangular space-time layout
- dataset metadata now stores layout information needed by geometry-aware models

Relevant files:

- `geometry/rotated_rect.py`
- `sample_dataset.py`
- `decoders/baseline_rectcnn.py`

Why this matters:

- the new decoder does not need to start from flat detector vectors

#### A3. Axis-Wise Supervision Branch

Completed:

- `single_basis` datasets remain supported
- x-basis and z-basis datasets can be paired via `dual_axis_manifest`
- aligned neural and classical axis-wise runners exist

Relevant files:

- `dual_axis_manifest.py`
- `tools/build_dual_axis_manifest.py`
- `tools/run_dual_axis_experiment.py`
- `tools/run_dual_axis_pymatching.py`

Role in the rebuilt project:

- not the final decoder target
- still useful for axis-wise analysis, pretraining, ablation, and debugging

#### A4. Logical-Frame Audit

Completed:

- the repository can explicitly test whether the raw scaffold supports same-shot
  logical-frame supervision

Relevant files:

- `logical_frame.py`
- `tools/audit_logical_frame_support.py`

Established conclusion:

- the raw single-basis scaffold does not truthfully provide same-shot X/Z
  logical frame labels

Why this matters:

- it justifies why class4 supervision cannot be fabricated from the raw memory
  scaffold alone

#### A5. True Per-Shot Class4 Label Path

Completed:

- a Bell-pair-assisted readout path was added
- one shot can now produce:
  - `logical_x_flip`
  - `logical_z_flip`
  - `logical_class4`

Relevant files:

- `logical_bell.py`
- `logical_targets.py`
- `sample_dataset.py`

Current class4 meaning:

- `0 = I`
- `1 = X`
- `2 = Z`
- `3 = Y`

Current truth status:

- this is the first truthful per-shot class4 supervision path in the repository
- it depends on an added Bell-pair readout and is not the raw scaffold itself

#### A6. Active Baselines Now Support Class4

Completed:

- `decoders/baseline_pymatching.py` can evaluate class4 datasets
- `decoders/baseline_rectcnn.py` can train and evaluate with
  `--target-mode logical_class4`
- `decoders/research_noise_aware_3d.py` can train, evaluate, and run manifest
  experiments with `--target-mode logical_class4`
- `decoders/factorized_logical_frame_decoder.py` can train, evaluate, and run
  manifest experiments while mixing class4 main supervision with optional
  auxiliary axis supervision

Why this matters:

- baseline/noise infrastructure is no longer the blocker for new decoder
  research

#### A7. Confidence-Aware Hybrid Fallback Exists

Completed:

- `tools/evaluate_hybrid_fallback.py` can sweep confidence thresholds for
  calibrated FLFD outputs
- the same tool can compare those thresholds against PyMatching fallback on the
  same class4 shots
- threshold selection can be based on seen training families and then reported
  on holdout families

Why this matters:

- calibration outputs are now tied to an actual decision policy
- the next bottleneck is improving confidence usefulness, not wiring fallback

#### A8. Current Noise Coverage

Completed:

- the repository can already generate and evaluate on:
  - `ideal`
  - `stage_a_si1000`
  - `stage_b_local`
  - `stage_c_corr`

Current status:

- this is enough for smoke experiments and first OOD comparisons
- it is not enough for a full robustness study

#### A8. Core Strategy Documents

Completed:

- this document now fixes:
  - the decoder research target
  - the current pipeline
  - the repository status
  - the next execution order

### B. What The Project Has Proven So Far

The repository has already proven the following points.

#### B1. Infrastructure Claim

Proven:

- the project can generate truthful class4 datasets and run both classical and
  neural baselines on them

Meaning:

- new decoder design can now start without another infrastructure rebuild

#### B2. Supervision Claim

Proven:

- truthful same-shot class4 supervision is possible in this repository

Important qualifier:

- it is currently achieved through the Bell-pair readout path, not through the
  raw scaffold alone

#### B3. Comparison Claim

Proven:

- the project can compare classical and neural baselines on both:
  - axis-wise supervision
  - per-shot class4 supervision

#### B4. Research Direction Claim

Proven:

- the right next problem is no longer data plumbing
- the right next problem is model design, confidence, hybridization, and final
  recovery-level evaluation

#### B5. Data-Regime Claim

Proven:

- the earlier smoke-level FLFD stagnation was largely a data-regime artifact
- once class4 data scale is increased and the FLFD is reduced to a smaller
  stable variant, the model stops defaulting to all-`I` predictions and begins
  to show meaningful macro-F1 and balanced-accuracy gains

Meaning:

- architecture reset should be a later decision, not the immediate reaction to
  smoke-level collapse
- the current mainline should first exhaust the larger-data, smaller-model
  regime

### C. What Is Still Not Completed

The following are still open and should not be overstated as solved.

#### C1. No Stronger Successor Beyond The First FLFD Line Yet

Not completed:

- the repository now has one concrete new neural architecture family:
  `factorized_logical_frame_decoder.py`
- however, there is still no clearly superior successor architecture beyond
  that first FLFD line
- the open question is whether a better next step is a stronger FLFD variant,
  a residual-to-classical design, or a new sparse/event-centric backbone

#### C2. No Recovery-Level Evaluation Yet

Not completed:

- the project still evaluates mostly at the label/frame level
- final logical failure after applying a recovery action is not yet the mainline
  metric

This is one of the largest remaining gaps.

#### C3. No Robust Hybrid Policy Yet

Not completed:

- the project now does have:
  - calibration evaluation
  - threshold-based fallback
  - learned routing against PyMatching
- however, it does not yet have a robust holdout-winning hybrid policy
- the current hybrid stack still needs to show stable improvement under larger
  runs and later at the recovery level

#### C4. No Native Same-Shot Logical-Frame Scaffold Yet

Not completed:

- the current truthful class4 path depends on Bell-pair readout augmentation
- the raw memory scaffold is still insufficient on its own

#### C5. Noise Breadth Is Still Limited

Not completed:

- there is no `stage_d` or `stage_e`
- there is no broader suite of custom shifted noise families
- there is no large-scale robustness benchmark yet

#### C6. Real Hardware-Oriented Path Is Still Missing

Not completed:

- no latency-aware decoding path
- no streaming decoder path
- no Willow-native xzzx schedule

### D. Working Repository Interpretation

From this point onward, the repository should be interpreted as follows.

#### D1. Main Task

Main task:

- build a new logical-class4 neural decoder

#### D2. Main Data Branch

Main data branch:

- `bell_pair_z_readout`

Reason:

- this is the only truthful per-shot class4 branch currently available

#### D3. Auxiliary Data Branch

Auxiliary data branch:

- `single_basis`

Reason:

- axis-wise analysis
- pretraining
- auxiliary losses
- debugging
- ablation against the raw scaffold

#### D4. Main Baseline Stack

Main baseline stack:

- PyMatching as the main classical comparison
- RectCNN as the lightweight neural baseline
- NoiseAware3D as the current stronger neural baseline

### E. Next Work Items In Correct Order

The remaining work should be done in the order below.

#### E1. Architecture Specification

Next mandatory task:

- write the first new architecture specification

The specification must define:

- the exact input tensor
- the exact model outputs
- whether uncertainty is explicit or derived
- whether `single_basis` is used as pretraining or auxiliary supervision
- the exact training objective

Acceptance condition:

- one document that fixes the first new model before implementation begins

#### E2. Unified Training Scheme

After architecture specification:

- implement the first training path for the new decoder

Recommended direction:

- shared encoder
- class4 main head
- optional axis-wise auxiliary heads or pretraining stage

Acceptance condition:

- one train/eval CLI that can run the new model on class4 manifests and, if
  needed, auxiliary single-basis data

#### E3. Calibration And Confidence Layer

After the first model works end-to-end:

- add confidence reporting and calibration analysis

Minimum targets:

- entropy or margin score
- reliability summary
- basic calibration metric

Acceptance condition:

- class4 experiments report both accuracy and confidence quality

#### E4. Hybrid Neural-Classical Evaluation

After confidence is available:

- implement selective fallback to PyMatching

Goal:

- use the neural model when confident
- use the classical decoder when uncertain

Acceptance condition:

- one experiment report that compares:
  - neural only
  - classical only
  - hybrid fallback

#### E5. Recovery-Level Evaluation

After hybrid comparison:

- move from label-level evaluation to recovery-level evaluation

Goal:

- compare final logical failure after a decoder action is applied

Acceptance condition:

- one reproducible evaluation path where decoder decisions map to a recovery or
  Pauli-frame action and are compared by final logical failure

#### E6. Noise Expansion

After the first recovery-level path exists:

- add more shifted or adversarial noise families

Goal:

- test whether the new decoder target really helps under noise mismatch

Acceptance condition:

- at least one new held-out noise family beyond the current `ideal/A/B/C`

#### E7. Native Scaffold Improvement

Longer-term task:

- investigate whether a more native same-shot logical-frame circuit path can be
  introduced without the current Bell-pair scaffolding

This is important but should not block the first decoder model.

### F. Practical Planning Rule

Until recovery-level evaluation exists, the project should use the following
planning rule:

- mainline model development happens on class4 datasets
- `single_basis` is auxiliary, not the end goal
- PyMatching remains the main classical baseline
- every new model must be testable on unseen noise families

### G. Short Version

If the whole repository must be summarized in a few lines:

- infrastructure rebuild is complete enough
- truthful per-shot class4 labels now exist
- active baselines can already use them
- the raw scaffold is still insufficient on its own
- the next true blocker is decoder design
- the correct order is:
  - architecture spec
  - unified training
  - calibration
  - hybrid fallback
  - recovery-level evaluation
  - broader noise robustness study
