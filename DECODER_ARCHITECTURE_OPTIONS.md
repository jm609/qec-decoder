# Decoder Architecture Options

This document records the architecture option space for the new decoder and
fixes the recommended execution order.

It exists to prevent the project from branching into too many model ideas at
once before the objective, evaluation path, and engineering constraints are
stable.

## 1. Decision Summary

The most reasonable path for this repository is:

1. implement the first new model as a dense 3-D factorized logical-frame
   decoder
2. use that model to validate the output formulation, training scheme, and
   class4 evaluation path
3. after that baseline is stable, implement a sparse 3-D variant under the same
   logical-frame output structure
4. only after a strong full-precision model exists, consider quantized or
   binary deployment variants

In short:

- `Dense3D first`
- `Sparse3D second`
- `BNN/QAT later`

## 2. What Must Stay Fixed Across Options

Regardless of backbone choice, the following should stay fixed for fair
comparison:

- main target: `logical_class4`
- auxiliary supervision: truthful axis-wise `single_basis`
- output structure: factorized logical frame
- comparison against the same class4 baselines
- seen vs unseen noise-family evaluation
- later extension to calibration and hybrid fallback

The repository should not compare a new backbone while also changing the task
definition, label meaning, and evaluation protocol at the same time.

## 3. Candidate Backbone Families

The realistic candidate families are:

- dense 3-D CNN
- sparse 3-D CNN
- event-centric GNN
- quantized or binary neural network

Each one can be combined with the factorized logical-frame head defined in
`DECODER_ARCHITECTURE_SPEC_V1.md`.

## 4. Dense 3-D CNN

### 4.1 What It Means Here

This option uses:

- the current rectangular syndrome volume
- dense tensor processing over `(time, height, width)`
- the factorized logical-frame output head

This is the option described in:

- `DECODER_ARCHITECTURE_SPEC_V1.md`

### 4.2 Advantages

- lowest implementation risk
- directly compatible with the current repository data path
- no new external sparse or graph dependency is required
- easiest option for validating the factorized objective itself
- easiest option for debugging, profiling, and ablation

### 4.3 Disadvantages

- does not exploit the strong sparsity of detector events at low error rates
- still pays compute on incoherent filler cells
- weaker novelty claim if used alone as the final paper contribution

### 4.4 Why It Should Be First

The first new model should answer:

- is the factorized logical-frame formulation correct?
- does auxiliary `single_basis` supervision help?
- does explicit noise-context conditioning help?

Dense 3-D CNN is the safest way to answer those questions without adding
backbone-specific uncertainty.

## 5. Sparse 3-D CNN

### 5.1 What It Means Here

This option replaces dense rectangular processing with sparse event-centric
3-D convolution while keeping:

- the same supervision
- the same factorized output head
- the same evaluation protocol

### 5.2 Why It Is Attractive

This repository's data has the following property:

- detector activity is sparse, especially in the below-threshold regime

That makes sparse 3-D processing attractive because it can:

- reduce wasted computation on empty lattice cells
- align better with event sparsity
- create a stronger efficiency and latency story than dense CNNs

### 5.3 Research Value

Sparse 3-D CNN is the strongest near-term extension candidate because it can
support a meaningful claim:

- the decoder is not only factorized at the logical-frame level, but also
  sparsity-aware at the syndrome-processing level

This is a stronger extension over the reference CNN paper than simply changing
filter details in a dense network.

### 5.4 Risks

- higher implementation complexity
- library and platform friction
- sparse kernels may not deliver the expected wall-clock gain immediately
- dense-vs-sparse comparisons can become noisy if preprocessing differs too much

### 5.5 Why It Should Not Be First

If sparse 3-D is implemented before the dense factorized model is stabilized,
then a failure becomes ambiguous:

- did the factorized objective fail?
- did the sparse processing fail?
- did the implementation complexity introduce noise?

That is why sparse 3-D should be the second model, not the first one.

## 6. Event-Centric GNN

### 6.1 What It Means Here

This option would represent the syndrome as a graph over detector events or
detector sites, with edges derived from:

- spatial adjacency
- temporal adjacency
- stabilizer structure
- optionally DEM-informed relations

### 6.2 Advantages

- strong structural inductive bias
- natural fit for irregular geometries or future schedule changes
- easier extension to non-rectangular layouts

### 6.3 Disadvantages

- more complex batching and implementation
- higher engineering risk than CNNs
- less straightforward latency path
- more moving parts before the project has even validated the factorized target

### 6.4 Role In This Project

GNN is a good future research branch, but not the first one.

It becomes more attractive after:

- the factorized objective is validated
- a recovery-level evaluation path exists
- the repository needs stronger geometry generalization than rectangular CNNs

## 7. Quantized / Binary Neural Network

### 7.1 What It Means Here

This option refers to:

- INT8 or lower-precision quantized models
- quantization-aware training
- binary or ternary weight/activation variants

### 7.2 Why It Is Tempting

The motivation is clear:

- lower compute
- lower memory
- potential latency gains
- future real-time deployment relevance

### 7.3 Why It Is Not The First Research Model

At the current project stage, the main bottleneck is not:

- full-precision model speed

The main bottlenecks are:

- choosing the right decoder objective
- making class4 decoding robust
- using `single_basis` honestly
- dealing with noise shift
- enabling confidence and hybrid fallback

If a binary model performs poorly, it becomes unclear whether:

- the decoder idea is weak
- binary constraints are too harsh
- calibration collapsed
- rare logical classes became too unstable

### 7.4 Correct Role In The Roadmap

Quantized or binary models should be treated as:

- deployment variants
- compression studies
- latency-oriented successors to a strong full-precision teacher

Not as the first mainline research model.

## 8. Recommended Architecture Sequence

The recommended sequence is:

### Stage A: FLFD-Dense3D

First implementation target:

- dense 3-D factorized logical-frame decoder

Purpose:

- validate objective
- validate auxiliary supervision scheme
- validate noise-context conditioning
- establish the first real non-baseline model

### Stage B: FLFD-Sparse3D

Second implementation target:

- sparse 3-D version with the same logical-frame head

Purpose:

- test whether event sparsity improves efficiency and robustness
- strengthen the paper-positioning beyond the reference dense CNN

### Stage C: FLFD-Hybrid

Third implementation target:

- calibrated dense or sparse model plus PyMatching fallback

Purpose:

- turn class4 posterior quality into practical decoding behavior

### Stage D: FLFD-Quantized

Fourth implementation target:

- quantized or binary deployment-oriented version

Purpose:

- latency and deployment study once the core decoder idea is already strong

## 9. How To Compare These Options Fairly

Every option should be compared under:

- the same class4 datasets
- the same train/eval family splits
- the same factorized head
- the same calibration procedure where applicable

At minimum compare:

1. class4 accuracy
2. macro-F1
3. seen vs unseen family gap
4. calibration quality
5. hybrid usefulness
6. inference cost and latency

Only after those are stable should recovery-level comparison be added.

## 10. Strongest Paper Positioning

If the project wants to make a defensible "beyond the reference CNN" claim,
the strongest path is:

- first validate factorized logical-frame decoding on dense 3-D tensors
- then show that sparse 3-D processing retains or improves decoding quality
  while improving efficiency or latency

This is more compelling than jumping directly to BNN, because:

- it keeps the decoding objective central
- it uses the repository's strongest current advantage, namely sparse detector
  activity
- it builds a clearer bridge from the reference dense CNN literature to a more
  modern practical decoder

## 11. Final Recommendation

The official recommendation for this repository is:

- do not make BNN the first mainline decoder
- do not make GNN the first mainline decoder
- implement the dense factorized model first
- treat sparse 3-D CNN as the highest-priority alternative backbone
- treat quantization or binary networks as later deployment variants

In practical terms:

- `DECODER_ARCHITECTURE_SPEC_V1.md` remains the first implementation target
- the next architecture after that should be a sparse 3-D variant under the same
  logical-frame formulation

## 12. Implementation Guidance

The repository should reflect this sequence in file planning.

Recommended order:

1. `decoders/factorized_logical_frame_decoder.py`
2. `decoders/factorized_logical_frame_decoder_sparse.py`
3. calibration and fallback utilities
4. later quantized deployment variants

This order minimizes risk while preserving a strong research story.
