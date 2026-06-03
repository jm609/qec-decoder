# Decoder Architecture Shortlist V2

This document narrows the next decoder step to two serious candidates and fixes
which one should become the next implementation target.

It supersedes the earlier broad option discussion by incorporating the new
`d3` and `d5` empirical results from the rebuilt class4 pipeline.

## 1. Current Evidence

The repository now has two important results:

- on `d3/r3` with `2048` shots per family, a smaller FLFD variant breaks the
  smoke-level all-`I` collapse and reaches meaningful macro-F1 and balanced
  accuracy
- on `d5/r5` with `2048` shots per family, even though train families contain
  real `X`, `Z`, and `Y` support, the current FLFD line still collapses badly
  while PyMatching remains strong

This means:

- the old smoke failure was not enough evidence for a full architecture reset
- but the current FLFD backbone does have a real distance-scaling limitation

## 2. Design Requirements

The next model must satisfy all of the following better than the current FLFD:

- use the current truthful per-shot `logical_class4` supervision path
- consume the current geometry-aware space-time data without another dataset
  rebuild
- scale better from `d3` to `d5+`
- remain compatible with calibration and hybrid evaluation
- leave a path toward latency-sensitive decoding

## 3. Shortlist

The next step is narrowed to two candidates.

### Candidate A: Multi-Scale Dense3D Factorized Decoder

Definition:

- a stronger direct class4 decoder
- 3-D multi-scale backbone, preferably U-Net-like
- factorized logical-frame output retained:
  - `logical_x_flip`
  - `logical_z_flip`
  - class4 residual or class4 fusion head

Core idea:

- the current FLFD fails partly because its trunk is too shallow and too local
  for larger volumes
- the next direct decoder should expand receptive field and combine local and
  global context without abandoning the current class4 supervision path

Why it fits the current repository:

- uses the existing `bell_pair_z_readout` labels directly
- uses the current rectangular/volume input pipeline directly
- needs no new supervision target construction
- can be compared against FLFD with a clean architectural delta

Expected benefit:

- better scaling from `d3` to `d5`
- stronger minority-class separation on larger syndrome volumes
- keeps the current class4 calibration and hybrid stack intact

Main risk:

- still a dense model
- may improve scaling but not solve latency or PyMatching-gap issues by itself

### Candidate B: Ising-Inspired Neural Pre-Decoder Plus PyMatching

Definition:

- neural network predicts local corrections, syndrome reduction, erasures, or
  matchability aids
- PyMatching remains the final logical decision layer

Reference motivation:

- NVIDIA Ising Decoding publicly positions its decoder as a neural pre-decoder
  feeding a standard decoder, and emphasizes latency plus logical-performance
  gains rather than neural-only replacement
- sources:
  - https://github.com/NVIDIA/Ising-Decoding
  - https://developer.nvidia.com/ising

Why it fits the project objective:

- our `d5` result already shows PyMatching remains very strong
- the project objective is practical decoding under noise shift and latency, not
  neural-only purity
- this direction aligns naturally with hybrid deployment

Why it does not fit the immediate next implementation as well:

- the current truthful labels are class4 frame labels, not local pre-decoder
  targets
- a serious pre-decoder branch needs new supervision construction and later
  recovery-level evaluation
- it is a larger system redesign than the next direct-decoder replacement

Expected benefit:

- best long-term system direction if direct class4 models keep losing to strong
  classical baselines at larger distance

Main risk:

- higher implementation and evaluation overhead
- requires new target design before fair training can start

## 4. Deprioritized Options

These are not the right next main step.

- `RectCNN` / paper-style compact CNN:
  too weak as a next mainline model; keep only as a baseline
- more loss tweaks on current FLFD:
  tempered imbalance was useful, but focal, hierarchical non-identity, and
  confidence-loss variants did not fix the core scaling issue
- BNN / quantized-first:
  too early while accuracy and calibration remain unresolved
- GNN-first:
  still attractive long-term, but it increases engineering risk before the
  multi-scale dense baseline question is settled
- learned router as the main model:
  routing only becomes useful once the primary neural decoder is stronger

## 5. Selection

The next implementation target should be:

> Candidate A: a multi-scale Dense3D factorized logical-frame decoder.

Short name recommendation:

- `M3D-FLFD`

Expanded name:

- `Multi-Scale 3D Factorized Logical-Frame Decoder`

## 6. Why Candidate A Wins The Immediate Slot

Candidate A is the best immediate choice because it balances:

- current-label compatibility
- current-input compatibility
- scientific clarity
- manageable implementation cost
- direct comparability with the current FLFD

Most importantly:

- it directly targets the newly exposed `d5` scaling failure
- it does so without introducing a new supervision problem

This makes it the cleanest next test of whether the repository's main problem
is the current FLFD trunk or the direct class4 formulation itself.

## 7. Why Candidate B Still Matters

Candidate B should not be dropped.

It should become the next parallel research branch after Candidate A is tested,
because it may ultimately be the more practical system architecture if:

- PyMatching continues to dominate at larger distances
- latency becomes central
- recovery-level evaluation favors neural assistance over neural replacement

In short:

- Candidate A is the next model
- Candidate B is the next branch

## 8. Cross-Cutting Training Idea To Borrow From Ising

One idea should be adopted regardless of which candidate is implemented first:

- denser-training curriculum or noise upscaling

Reason:

- the NVIDIA Ising material explicitly treats sparse-syndrome training as a
  learning bottleneck
- our own experiments already showed that data regime strongly changes decoder
  behavior

That means the next direct class4 model should be trained with at least one of:

- mixed-noise-density curriculum
- elevated-noise pretraining followed by target-noise finetuning
- class-aware or difficulty-aware sampling

## 9. Immediate Implementation Rule

The next coding step should therefore be:

1. implement `M3D-FLFD`
2. keep the current factorized logical-frame output idea
3. replace the shallow FLFD trunk with a multi-scale 3-D encoder
4. evaluate first on:
   - `d3/r3` class4
   - `d5/r5` class4
5. compare against:
   - current FLFD
   - PyMatching

Success condition:

- meaningful `d5` macro-F1 and balanced-accuracy improvement over the current
  FLFD
- hybrid fallback no longer degenerates immediately to fallback-all

## 10. Short Practical Summary

The decision is:

- next implementation: `M3D-FLFD`
- next parallel branch after that: Ising-style pre-decoder plus PyMatching

This is the best match between:

- the current project objective
- the current truthful labels
- the current geometry-aware input pipeline
- the new `d3` vs `d5` evidence
