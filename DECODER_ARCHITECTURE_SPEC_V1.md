# Decoder Architecture Spec V1

This document defines the first new neural decoder architecture that should be
implemented on top of the rebuilt repository.

Backbone option comparison and overall model sequencing are recorded separately
in `DECODER_ARCHITECTURE_OPTIONS.md`.

It is intentionally pragmatic:

- it uses the current mainline data path
- it reuses the current geometry-aware pipeline
- it can exploit both `bell_pair_z_readout` and `single_basis` data
- it is designed to support later calibration and hybrid fallback work

This is the first real model specification, not another baseline note.

## 1. Chosen Design

The first new decoder should be a:

> Factorized logical-frame decoder with a shared spatiotemporal encoder,
> auxiliary axis heads, and a residual class4 coupling head.

Short name:

- `FLFD-v1`

Expanded name:

- `Factorized Logical-Frame Decoder v1`

## 2. Why This Is The Right First Design

This design is preferred over a plain 4-class softmax model because:

- the project already has truthful per-shot class4 labels
- the project also has useful axis-wise supervision from `single_basis`
- the logical frame is naturally a 2-bit object, not just an arbitrary 4-class
  token
- we want a model that is interpretable and debuggable
- we want to reuse auxiliary supervision without inventing fake class4 labels

This design is preferred over a transformer-first or GNN-first model because:

- the repository already has a strong rectangular space-time tensor path
- the available data scale is still moderate
- a first model should test the right objective before testing expensive model
  families
- confidence, calibration, and hybrid evaluation matter more right now than
  maximum architecture novelty

## 3. Core Hypothesis

The key hypothesis is:

> A decoder that explicitly models the logical frame as two coupled logical bits
> will use the repository's data more efficiently and generalize better under
> noise shift than a direct black-box 4-class classifier.

This splits into two sub-hypotheses:

1. shared representation learning from axis-wise and class4 data will improve
   generalization
2. a factorized `X/Z + residual coupling` output structure will be more stable
   than a pure 4-class head

## 4. Model Objective

The model's main prediction target is:

- `P(L | syndrome, noise_context)` where `L in {I, X, Z, Y}`

But the model is not trained as a pure flat class4 predictor.

Instead it predicts:

- `logical_x_flip`
- `logical_z_flip`
- a residual class4 correction

The final class4 posterior is built from those parts.

## 5. Input Definition

The model should consume the existing geometry-aware volume pipeline.

### 5.1 Dynamic Input

Per-shot dynamic input:

- detector-event volume over `(time, height, width)`

From current mainline:

- built from `detector_events`
- using the rectangular layout path already implemented

### 5.2 Static Geometry Input

Static per-dataset planes:

- valid mask
- checkerboard class
- detector type
- boundary indicator
- final-round indicator

These are already available through current layout metadata.

### 5.3 Noise Context Input

Global context should be included, but not only as broadcast planes.

Use:

- stage identity
- distance
- rounds
- detector count
- detector occupancy summary
- noise parameters such as `p`, `p_cz`, `p_meas`, and other available metadata
- instruction histogram features already extracted in the repository

The noise context should be encoded by a small MLP and used to modulate the
main trunk.

## 6. Representation Strategy

The model should separate three kinds of information:

- dynamic syndrome evidence
- static code geometry
- global noise context

This matters because they play different roles:

- syndrome evidence carries shot-specific information
- geometry defines locality and meaning of detector positions
- noise context changes the decoder prior

The current baseline `research_noise_aware_3d.py` already mixes these, but the
new architecture should do it more deliberately.

## 7. Encoder Design

The encoder should be a multiblock spatiotemporal residual CNN with context
conditioning.

### 7.1 Why CNN First

Use a CNN first because:

- surface-code syndrome structure is highly local in space and time
- the current data path is already a rectangular volume
- CNNs are efficient enough for later latency work
- this keeps implementation risk low while testing the factorized objective

### 7.2 Recommended Trunk

The encoder should contain:

- one stem projection from input channels to hidden channels
- multiple residual blocks with factorized kernels
- masked or geometry-aware pooling at the end

Recommended block types:

- spatial block: `(1, 3, 3)`
- temporal block: `(3, 1, 1)`
- occasional mixed block: `(3, 3, 3)`

Reason:

- spatial and temporal correlations are related but not identical
- factorizing them keeps compute lower and makes the inductive bias clearer

### 7.3 Noise Conditioning

Noise context should modulate the trunk through FiLM-style conditioning.

That means:

- a context MLP produces per-block scale and shift terms
- each residual block uses those to modulate activations

Reason:

- global decoder priors should adapt to the noise family
- simply broadcasting context planes everywhere is weaker and less explicit

### 7.4 Pooling

Use masked global pooling over valid lattice cells.

Do not treat invalid rectangular filler cells as real lattice content during the
final aggregation.

## 8. Output Heads

The output structure is the most important design choice.

### 8.1 Axis Heads

Two binary heads:

- `head_x`: predicts `logical_x_flip`
- `head_z`: predicts `logical_z_flip`

These heads are directly trainable from:

- `single_basis` data
- Bell-pair class4 data

### 8.2 Residual Class4 Head

One additional head:

- `head_r`: predicts a residual class4 correction vector of size 4

This head is only fully supervised on class4 data.

### 8.3 Final Class4 Logits

The class4 logits should not be produced independently from scratch.

Instead:

1. produce two bit logits:
   - `u_x`
   - `u_z`
2. form a factorized base logical-frame score:
   - `I : 0`
   - `X : u_x`
   - `Z : u_z`
   - `Y : u_x + u_z`
3. add a learned residual class4 correction:
   - `logits4 = base_logits + residual_logits`

This gives the model two useful behaviors:

- if the bits are mostly independent, the model can act like a simple logical
  frame model
- if there are correlated effects, the residual head can learn the deviation

This is the main design idea of V1.

## 9. Why The Factorized Output Matters

This output design is valuable because it:

- uses `single_basis` supervision honestly
- makes X and Z behavior separately observable
- prevents the class4 task from becoming an opaque black box
- gives a natural bridge from axis-wise training to class4 training

It also makes debugging much easier:

- if class4 fails, we can ask whether:
  - the X bit is wrong
  - the Z bit is wrong
  - the coupling residual is wrong

## 10. Confidence Design

V1 should not introduce a separate learned confidence head yet.

Instead, confidence should be derived from the calibrated class4 posterior.

Use:

- posterior entropy
- top-1 minus top-2 margin
- max class probability

Reason:

- simpler
- easier to calibrate
- enough for the first hybrid fallback experiments

Later versions can add explicit confidence prediction if needed.

## 11. Training Data Usage

V1 should use both branches, but not symmetrically.

### 11.1 Main Supervision Branch

Main branch:

- `bell_pair_z_readout`

Why:

- it is the truthful per-shot class4 target path

### 11.2 Auxiliary Supervision Branch

Auxiliary branch:

- `single_basis`

Why:

- provides truthful axis-wise supervision
- helps pretraining
- helps regularization
- improves interpretability

Important rule:

- `single_basis` must never be reinterpreted as if it were direct class4 data

## 12. Training Schedule

The training schedule should be staged.

### Stage 1: Axis Pretraining

Train:

- shared encoder
- `head_x`
- `head_z`

Data:

- `single_basis` x-basis and z-basis datasets
- optionally class4 datasets decomposed into x/z bit labels

Loss:

- binary losses on available axes only

Purpose:

- learn robust logical-bit features before class4 coupling is introduced

### Stage 2: Joint Class4 Training

Train:

- shared encoder
- axis heads
- residual class4 head

Data:

- class4 Bell-pair datasets as the main stream
- optional continued auxiliary `single_basis` batches

Loss:

- class4 cross-entropy on class4 batches
- auxiliary bit losses on available labels

Purpose:

- align the learned representation to the true class4 task

### Stage 3: Calibration

Freeze the trained network and calibrate its class4 posterior.

Recommended first method:

- temperature scaling on validation data

Purpose:

- make entropy, margin, and confidence thresholds usable

### Stage 4: Hybrid Threshold Search

Using calibrated validation outputs:

- select a fallback threshold
- low-confidence shots defer to PyMatching

Purpose:

- create the first practical hybrid decoder path

## 13. Loss Design

The first loss should be simple and stable.

### 13.1 On Class4 Batches

Use:

- `CE(class4_logits, logical_class4)`
- auxiliary `BCE(logical_x_head, logical_x_flip)`
- auxiliary `BCE(logical_z_head, logical_z_flip)`

### 13.2 On Single-Basis Batches

Use only the available axis loss:

- z-basis single-basis data supervises `logical_x_flip`
- x-basis single-basis data supervises `logical_z_flip`

### 13.3 Regularization

Use:

- weight decay
- dropout
- small residual-head penalty if needed

Do not start with focal loss, label smoothing, or highly exotic objectives
unless imbalance clearly forces it.

## 14. Class Imbalance Strategy

Class imbalance is expected because the `I` class dominates.

The first strategy should be:

- class-weighted cross-entropy or logit adjustment
- careful validation by macro-F1, not accuracy only

Do not rely on raw accuracy for model selection.

## 15. Evaluation Protocol

The model should be judged in the following order.

### 15.1 Label-Level Evaluation

Report:

- class4 accuracy
- macro-F1
- confusion matrix
- bitwise X/Z accuracy

### 15.2 Generalization Evaluation

Report:

- seen-family performance
- unseen-family performance
- gap between the two

Suggested first split:

- train on `stage_a_si1000 + stage_b_local`
- evaluate on `ideal + stage_a_si1000 + stage_b_local + stage_c_corr`

### 15.3 Calibration Evaluation

Report:

- negative log-likelihood
- Brier score if feasible
- expected calibration error
- entropy or margin distribution by correctness

### 15.4 Hybrid Evaluation

Report:

- neural-only performance
- PyMatching-only performance
- fallback hybrid performance

### 15.5 Recovery-Level Evaluation

Later extension:

- compare final logical failure after decoder action

This is not blocked by the architecture, but it is outside the first
implementation step.

## 16. Baselines For Ablation

The first new model must be compared against:

- `decoders/baseline_pymatching.py`
- `decoders/baseline_rectcnn.py`
- `decoders/research_noise_aware_3d.py`

Internal ablations should include:

1. pure 4-class head without factorization
2. factorized X/Z heads without residual class4 correction
3. full factorized model with residual coupling
4. with and without `single_basis` pretraining
5. with and without noise-context conditioning

If the full model does not beat at least one simpler ablation, the design claim
is weak.

## 17. Why This Is Better Than A Plain 4-Class Head

A plain 4-class head wastes repository structure.

It ignores:

- the natural 2-bit meaning of the label
- the auxiliary supervision already available from `single_basis`
- the need for better debugging under OOD failure

The factorized design turns existing repository asymmetry into a strength.

## 18. Why This Is Still A First-Step Model

This is not meant to be the final decoder forever.

It is meant to be the first serious architecture because it is:

- grounded in the current repo
- scientifically testable
- implementable without another rebuild
- compatible with later calibration and hybrid fallback work

Later models may replace the encoder family entirely, but they should still be
compared against this factorized formulation.

## 19. Expected Failure Modes

The model may fail in predictable ways.

### 19.1 Collapse To `I`

Because the class prior is imbalanced, the class4 head may overpredict `I`.

Response:

- monitor macro-F1
- use class weighting
- inspect bit heads separately

### 19.2 Good Bits, Bad Coupling

The X and Z heads may work while the residual class4 head fails.

Response:

- compare the factorized base logits alone against the full model
- inspect whether the residual head adds value or only noise

### 19.3 Good Seen-Family Results, Poor Holdout Results

The model may overfit the noise context.

Response:

- ablate FiLM conditioning
- simplify context features
- compare with and without single-basis pretraining

## 20. Implementation Plan

The implementation should introduce a new decoder file rather than overwrite the
existing baselines.

Recommended new module:

- `decoders/factorized_logical_frame_decoder.py`

Recommended implementation order:

1. reuse current volume builders and dataset loaders
2. implement the shared encoder and axis heads
3. implement class4 residual head and final logit composition
4. add mixed-branch training with masked losses
5. add calibration utilities
6. add hybrid fallback evaluation

## 21. Acceptance Criteria For V1

V1 is successful if all of the following are true.

### Must-Have

- trains on truthful class4 data
- can use `single_basis` as auxiliary supervision
- reports class4 and bitwise metrics
- runs seen vs unseen family experiments
- produces calibrated posteriors usable for fallback

### Strong Success

- beats `research_noise_aware_3d.py` on at least one meaningful class4
  generalization setting
- improves a hybrid neural+PyMatching policy relative to either alone

## 22. Short Practical Summary

The first new decoder should be:

- CNN-based, not transformer-first
- factorized into X/Z bits plus class4 residual coupling
- trained mainly on class4 Bell-pair data
- supported by auxiliary `single_basis` supervision
- calibrated after training
- evaluated for OOD and hybrid usefulness

If a future proposal differs from this, it should explain clearly why it is
better than this first factorized design.
