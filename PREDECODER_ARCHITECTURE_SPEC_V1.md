# Pre-Decoder Architecture Spec V1

This document defines the first serious pre-decoder branch for the repository.

It is written after the direct class4 decoder line reached a clear limit:

- larger `d3` data showed that direct class4 learning was not completely
  misguided
- but larger `d5` runs showed that both the original FLFD line and the first
  multi-scale Dense3D successor still collapse badly relative to PyMatching

Because of that, the next model should no longer be another direct dense class4
decoder tweak.

The next model should be a neural pre-decoder feeding a strong classical final
decoder.

## Current Empirical Status

This document is no longer only a design note. The first branch has now been
implemented and tested.

What now exists:

- `tools/build_pymatching_edit_targets.py`
- `decoders/syndrome_edit_predecoder.py`

What the pilot results currently say:

- bounded local-edit oracle search shows strong headroom over raw PyMatching on
  both larger `d3` and larger `d5` pilots
- the first SEDP-v1 training recipe does **not** realize that headroom yet
- follow-up hard-shot weighted sampling and hard-shot-only edit supervision do
  not fix the problem by themselves
- the first decision-aware follow-up now also exists as a candidate-edit
  selector layer inside `decoders/syndrome_edit_predecoder.py`, but it has only
  been partially validated so far
- the first real `d3` / `d5` pilot reruns of that selector path still do not
  show a real system-level gain over the safe baseline policy: both distances
  still select `global_policy`, and the selector still chooses no edits in
  practice on final eval
- the first in-training decision-aware ranking-loss follow-up now also exists,
  but real `d3` / `d5` pilot reruns still do not move the branch out of the
  same safe no-edit regime
- the first stronger selector-training follow-up now also exists as a per-shot
  group-ranking objective over the generated candidate set, but real `d3` /
  `d5` pilot reruns still do not move the branch out of the same safe no-edit
  regime

Current interpretation:

- the main bottleneck is likely no longer data plumbing
- it is more likely the mismatch between:
  - detector-level BCE supervision on one chosen edit mask
  - and the real system objective: improving final PyMatching

## 1. Chosen System Direction

The first pre-decoder should be a:

> Syndrome-edit pre-decoder followed by unchanged PyMatching final decoding.

Short name:

- `SEDP-v1`

Expanded name:

- `Syndrome Edit Pre-Decoder v1`

## 2. Why This Is The Right Next Move

The repository now has direct evidence that:

- PyMatching remains strong at larger distance
- direct dense class4 decoders do not currently scale well enough
- the project objective is practical decoding quality under noise shift, not
  neural-only purity

This makes a pre-decoder branch an extension beyond the original CNN baseline.
The original main target paper / format anchor remains:

- Jung, Ali, Ha, "Convolutional Neural Decoder for Surface Codes",
  IEEE Transactions on Quantum Engineering, 2024
- DOI: `10.1109/TQE.2024.3419773`

The repository already followed that format in its first geometry-aware neural
baseline:

- rectangular lattice syndrome representation
- incoherent fill value for invalid rectangle cells
- CNN processing aligned with the surface-code lattice
- logical-error / decoded-correctness evaluation against classical baselines

The current pre-decoder branch is aligned with both:

- the empirical evidence from the current repository
- the later system-level direction suggested by NVIDIA's Ising-Decoding
  materials

Secondary pre-decoder reference:

- https://github.com/NVIDIA/Ising-Decoding
- https://developer.nvidia.com/ising
- https://research.nvidia.com/publication/2026-04_fast-ai-based-pre-decoders-surface-codes

Important takeaways from the secondary pre-decoder reference:

- the neural model is a local, parallel pre-decoder, not the final logical
  decision maker
- the pre-decoder should reduce residual syndrome difficulty before a global
  decoder handles the remaining correction
- performance should be judged by end-to-end logical decoding, not by matching
  an arbitrary detector-level label
- the repo's current `neural pre-decoder -> edited syndrome -> PyMatching`
  direction is aligned with this, but the current learned route/emit decision
  is not yet calibrated enough to realize the oracle headroom

## 3. Main Design Principle

The pre-decoder should not try to replace the final logical decision layer.

Instead, it should:

1. inspect the raw syndrome
2. make a small structured edit or assistance prediction
3. produce a modified syndrome for the final classical decoder
4. let PyMatching make the final logical-frame decision

In one sentence:

> The neural model should learn how to minimally rewrite hard shots into
> easier-to-decode shots, while leaving the final logical decision to
> PyMatching.

## 4. Why A Syndrome-Edit Pre-Decoder

Among pre-decoder choices, the first one should be a syndrome-edit model rather
than:

- direct local Pauli recovery on data qubits
- edge-weight reparameterization only
- detector-graph GNN routing only

The reason is current supervision availability.

The repository already has:

- `detector_events`
- truthful per-shot `logical_class4`
- detector geometry metadata
- `detector_error_model.dem`
- PyMatching evaluation on the same shots

The repository does not yet have:

- truthful per-shot physical fault labels
- truthful per-shot optimal local recovery labels on data qubits

Therefore, a syndrome-edit target can be constructed from current artifacts
without pretending that hidden physical-fault labels are already available.

## 5. System Definition

The full system is:

1. input raw syndrome `s`
2. neural pre-decoder predicts edit mask `e_hat`
3. construct edited syndrome `s' = s xor e_hat`
4. run PyMatching on `s'`
5. take the resulting logical-frame prediction as final output

This keeps the final decoder unchanged and isolates the neural contribution to a
clear system role.

## 6. Input Definition

The neural pre-decoder should consume the same geometry-aware space-time input
stack already used by the direct decoder line.

### 6.1 Dynamic Input

- detector-event volume over `(time, height, width)`

### 6.2 Static Geometry Input

- valid mask
- checkerboard class
- detector type
- boundary flag
- final-round flag

### 6.3 Global Context Input

- stage identity
- distance
- rounds
- detector count
- noise metadata already used by the current volume builders

This keeps the pre-decoder compatible with the current data path and future
noise-transfer work.

## 7. Output Definition

`SEDP-v1` should output two things.

### 7.1 Detector Edit Logits

One logit per detector slot in the rectangular volume:

- probability that this detector bit should be toggled before final decoding

After masking invalid lattice cells, this gives an edit mask candidate.

### 7.2 Shot-Level Edit Head

One auxiliary binary head:

- `needs_edit`

Meaning:

- whether this shot appears to require any syndrome edit before PyMatching

This head is not the final product, but it helps:

- calibration
- abstention
- edit sparsity control
- later routing

## 8. Target Construction

This is the most important section.

The target should be constructed offline from current artifacts.

### 8.1 Baseline Classification

For each shot:

1. run PyMatching on the original syndrome
2. compare the result against truthful `logical_class4`

This gives:

- `baseline_pymatching_correct`

### 8.2 Easy Shots

If PyMatching is already correct:

- target edit mask is all zeros
- `needs_edit = 0`

This is critical because the pre-decoder must learn not to "fix" shots that are
already solved.

### 8.3 Hard Shots

If PyMatching is wrong:

search for a small detector edit mask that makes PyMatching correct.

The target should be:

- the smallest found edit mask under a local search budget
- `needs_edit = 1` if such a mask is found

If no mask is found within budget:

- mark the shot as `unsolved_by_local_edit`
- optionally exclude it from edit-mask supervision while still using it as a
  hard negative for `needs_edit`

### 8.4 Search Rule

The search should be constrained.

Recommended first search:

- only toggle currently active detectors or detectors in a small local
  neighborhood of active detectors
- limit edit weight to a small budget, for example `k <= 2` or `k <= 3`
- prefer masks with:
  1. fewer edits
  2. edits on valid active detector neighborhoods
  3. earlier success under PyMatching

This makes target generation computationally bounded and physically plausible.

## 9. Why This Target Is Honest Enough

This target is not "ground-truth physical error."

That is acceptable, because the pre-decoder is not being asked to reconstruct
the hidden physical error. It is being asked to assist a final decoder.

So the correct supervision question is:

> What small edit would make the final decoder recover the correct logical frame?

That is exactly the system-level role we want.

## 10. Required New Artifact Layer

This branch needs a new derived artifact layer built from existing class4
datasets.

Recommended tool:

- `tools/build_pymatching_edit_targets.py`

Recommended output root:

- `artifacts/datasets/predecoder_targets/...`

Per family, store:

- original family reference
- detector edit target mask
- `baseline_pymatching_correct`
- `needs_edit`
- `solved_by_local_edit`
- search budget metadata
- edit-weight histogram

This should not replace the original dataset. It should be a derived layer.

## 11. First Model Architecture

The first pre-decoder model should be simpler than the failed direct decoder
stack.

Recommended first model:

- shallow-to-medium 3-D encoder
- U-Net-like skip structure or two-scale feature fusion
- per-detector edit head
- global `needs_edit` head

This is not the place to start with an extremely large backbone.

Why:

- the hard part is the target and system coupling, not another heavy trunk
- the final decoder already provides strong inductive bias

## 12. Loss Design

The first loss should be:

- masked BCE on detector edit logits
- auxiliary BCE on `needs_edit`
- sparsity penalty on predicted edit rate

Possible form:

- `L = L_edit + lambda_need * L_needs_edit + lambda_sparse * mean(edit_prob)`

The sparsity term is important because:

- the model should learn minimal corrections
- not arbitrary syndrome rewrites

## 13. Inference Rule

The first inference policy should be simple.

1. compute per-detector edit probabilities
2. restrict to valid detector positions
3. either:
   - threshold probabilities, or
   - take top-k edits
4. apply XOR edit to detector events
5. run unchanged PyMatching

The first version should expose both:

- threshold policy
- top-k policy

## 14. Evaluation Protocol

The evaluation must be system-level.

### 14.1 Required Baselines

Compare:

- raw PyMatching
- direct neural-only FLFD
- direct hybrid threshold policy
- pre-decoder + PyMatching

### 14.2 Required Metrics

Report:

- final class4 accuracy after PyMatching on edited syndrome
- macro-F1 after final decoding
- fraction of shots edited
- average edit weight
- solved-hard-shot rate
- holdout-family improvement over raw PyMatching

### 14.3 Oracle Metrics

Also report:

- oracle local-edit success rate from the offline target builder

This matters because it separates two cases:

- target-generation branch has real headroom
- or local syndrome edits simply do not help enough

Without the oracle, pre-decoder failure is hard to interpret.

## 15. Why This Is Better Than Another Direct Decoder Right Now

Another direct decoder would still be trying to win against PyMatching
head-on.

This branch instead asks the more practical question:

> Can neural assistance make PyMatching better on the shots it currently
> mishandles?

That is better aligned with:

- the repository's current empirical evidence
- realistic deployment logic
- the strength of the current classical baseline

## 16. Immediate Risks

The first serious risks are:

### 16.1 Target Search May Be Too Weak

If local search rarely finds a useful edit:

- the branch may have little headroom

This is why oracle local-edit statistics must be computed first.

### 16.2 Target Search May Be Too Expensive

If edit search is too expensive:

- restrict search to active-detector neighborhoods
- lower edit budget
- precompute candidate neighborhoods from detector geometry

### 16.3 Model May Learn Overediting

If the model edits too often:

- strengthen sparsity penalty
- calibrate the `needs_edit` head
- add a strict top-k edit budget

### 16.4 Model May Collapse To Identity No-Edit

This risk is now observed in practice on current pilots.

What this means:

- once the policy selection is made accuracy-safe, the current learned model
  often prefers the identity edit
- therefore sampling and detector-level BCE alone are not enough to force the
  model to realize the oracle headroom

Most likely implication:

- the next change should target the system objective more directly, not just
  the sampling distribution

### 16.5 First Decision-Aware Follow-Up May Still Default To Identity

This risk also now exists in a more explicit form.

What now exists:

- `decoders/syndrome_edit_predecoder.py` can now build multiple candidate edits
  from the current threshold / top-k policy grid
- it can train a `CandidateEditSelector` over those candidates
- it can save both the raw global policy and selector-based metrics in the
  checkpoint and eval outputs

What the first smoke run says:

- the selector plumbing works end-to-end
- validation guardrails can still choose the original `global_policy` mode
- therefore simply adding a selector layer is not, by itself, evidence that
  the branch now unlocks the oracle headroom

What the first real pilot reruns add:

- the same conclusion survives on the actual `d3` / `d5` pilot manifests
- final eval still stays at raw PyMatching accuracy
- the selector still chooses identity `no-edit` in practice
- however, candidate-oracle accuracy remains above baseline, so there is still
  unrealized headroom inside the current candidate pool

What the first ranking-loss reruns add:

- even when the training loop is modified to prefer the oracle edit target over
  identity `no-edit` on solved hard shots, the final selected behavior can
  still remain unchanged
- in practice, that means this first pairwise identity-vs-target margin loss is
  too weak or too indirect to change the safe policy outcome by itself

What the first group-rank selector reruns add:

- even when selector training is changed from per-candidate BCE to the actual
  within-shot ranking problem, the final selected behavior can still remain
  unchanged
- in practice, that means the next bottleneck is probably not only "how to fit
  the selector better" but "how to make the model generate useful safe nonzero
  candidates in the first place"

What the first motif-vocabulary reruns add:

- even when the selector output is constrained to a small vocabulary of
  observed hard-shot edit masks, the final selected behavior can still remain
  unchanged
- in practice, that means a static whole-mask vocabulary is still too weak or
  too indirect to surface useful safe nonzero edits by itself

What the first motif-augmented candidate-pool reruns add:

- even when the selector candidate set itself is expanded with observed motif
  actions, the selector can still remain stuck on identity `no-edit`
- in practice, that means a stronger candidate pool alone is still not enough;
  the training objective still does not make a beneficial nonzero action
  attractive enough relative to identity

What the first explicit selector identity-margin reruns add:

- even when selector training explicitly compares identity against the best
  available nonzero candidate on the shots where that nonzero candidate is
  actually better, the final selected behavior can still remain unchanged
- in practice, that means the bottleneck is probably now deeper than the
  selector head itself
- the identity-vs-nonzero preference likely has to be pushed into the
  edit-logit / action-generation path, not only into downstream selector
  fitting

What the first action-path structured motif-competition reruns add:

- even when identity-vs-nonzero competition is pushed into the
  `edit_logits + needs_edit_logits` path through a structured action-class loss,
  the current system still does not unlock holdout gains or select nonzero
  edits in practice
- however, that deeper loss does appear to stabilize the earlier `d5`
  over-editing failure back to baseline-level behavior
- in practice, that means the current bottleneck is now likely the action
  parameterization / inference path itself, not only the loss placement

What the first action-motif emit reruns add:

- when the structured motif-action path is allowed to emit actions directly,
  it can finally produce nonzero edits instead of remaining a purely auxiliary
  training signal
- on `d3`, this improves seen-family eval but slightly harms holdout
  `stage_c_corr`
- on `d5`, the validation guardrail chooses an emit margin that suppresses all
  action emission
- in practice, that means static whole-mask motif actions can overfit seen
  families; the next action space needs to be more local and generalizable

What the first local-motif action reruns add:

- the action space is no longer tied to exact whole-detector masks
- observed hard-shot oracle edits are converted to relative `(dt, dr, dc)`
  patterns, then expanded over all valid detector-coordinate anchors at
  inference
- a first edit-validity gate, `local_motif_min_bit_logit`, now lets validation
  suppress local placements whose selected detector logits are too weak
- real `d3` / `d5` gated pilot reruns still select `global_policy`
- `d3` shows only a tiny seen-family movement:
  - local `stage_b_local`: `0.89453125 -> 0.8984375`
  - holdout `stage_c_corr`: unchanged at `0.9609375`
- `d5` stays fully unchanged under the selected local gate
- in practice, this confirms that locality/generalization plumbing is now in
  place, but the scorer still needs system-level decision awareness over the
  generated local placements

What the first local-motif selector reruns add:

- the local placement action space can now be used as a decision-aware
  candidate pool for `CandidateEditSelector`
- candidate labels are produced by actually applying each local placement and
  running PyMatching, so the candidate oracle is system-level rather than
  detector-BCE-level
- on `d3`, the local placement candidate oracle reaches `1.0` on all eval
  families, but the default selector still chooses identity everywhere
- a stronger hard-shot selector can emit nonzero actions, but it over-emits:
  it improves `stage_b_local` by one shot while harming `stage_a_si1000` by
  three shots
- a validation-selected `selector_emit_margin` can suppress that over-emission
  on `d3`, but then the result returns to identity/no-gain behavior
- on `d5`, the local placement candidate oracle remains very high, but the
  selector remains uncalibrated and the selected system still falls back to
  `global_policy`
- in practice, this moves the bottleneck from candidate generation to
  hard-shot routing / calibrated emit decisions

What the first hard-shot router reruns add:

- the branch now has a factorized `local_motif_router` mode:
  - a shot-level `HardShotRouter` predicts whether any edit should be attempted
  - only routed shots allow the local selector to choose a nonzero placement
  - non-routed shots are forced to identity
- router labels are built from the same system-level local candidate bundle:
  target `1` when the best nonzero local candidate beats identity after actual
  PyMatching decode, else target `0`
- router input was extended from pooled trunk features to pooled features plus
  local candidate confidence summaries
- real `d3` / `d5` pilot reruns still select `global_policy`
- in the current 256-shot pilot regime, router positives are too sparse:
  validation positive rates are only about `5-10%` on the d3 train-family
  splits
- the learned router collapses to all-negative or all-positive regimes; system
  selection then either routes no shots or suppresses edits with the selector
  emit margin
- a larger 1024-shot target rerun was also completed:
  - `artifacts/datasets/predecoder_targets_d3_2k_router1k/manifest.json`
  - `artifacts/datasets/predecoder_targets_d5_2k_router1k/manifest.json`
  - `artifacts/eval/nn/sedp_d3_router1k_localmotif_router_feat/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_localmotif_router_feat/experiment_summary.json`
- that rerun preserves strong oracle headroom:
  `d3 stage_c_corr 0.9287109375 -> 0.9921875` and
  `d5 stage_c_corr 0.888671875 -> 0.978515625`
- but actual learned decoding still selects identity/no-edit:
  `d3 stage_c_corr 0.9287109375 -> 0.9287109375` and
  `d5 stage_c_corr 0.888671875 -> 0.888671875`
- the local-motif-router candidate oracle remains saturated/high
  (`d3` all eval families `1.0`; `d5` stage_a/stage_c `0.9990234375`), but
  router route fraction is `0.0` on every eval family
- a first router-supervision follow-up now exists in code:
  - router label modes: `identity_vs_nonzero`, `baseline_failure`,
    `oracle_solvable`
  - optional `baseline_failure` / `oracle_solvable` pretraining
  - optional balanced negative sampling via `router_negative_ratio`
- real balanced/pretrained router reruns are complete:
  - `artifacts/eval/nn/sedp_d3_router1k_router_pretrain_balanced/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_router_pretrain_balanced/experiment_summary.json`
- those runs still select `global_policy` and route no eval shots; selected
  decoding remains `d3 stage_c_corr 0.9287109375 -> 0.9287109375` and
  `d5 stage_c_corr 0.888671875 -> 0.888671875`
- side observation: in the d3 balanced/pretrained run, the action-motif eval
  path improves `stage_c_corr` slightly to `0.931640625`, with seen-family
  gains on `stage_a_si1000` and `stage_b_local`; this should be treated as a
  new action-emission signal, not as a solved router result
- a focused action-motif selected-mode rerun is also complete:
  - `artifacts/eval/nn/sedp_d3_router1k_actionmotif_selected/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_actionmotif_selected/experiment_summary.json`
- in that rerun, `d3` selects a non-identity `global_policy` rather than the
  action-motif inference path and improves
  `stage_c_corr 0.9287109375 -> 0.9306640625`; seen-family gains are larger,
  but the holdout gain is only two net shots (`17` improved, `15` harmed)
- the matching `d5` run stays identity/no-edit and does not improve
- in practice, this means the next bottleneck is router supervision/data
  regime, not local candidate generation

Practical implication:

- the next useful work is not more selector plumbing alone
- it is also not just a moderately larger same-recipe target manifest
- it is not baseline-failure pretraining plus simple balanced router batches by
  itself
- the local/generalizable action space is good enough to expose oracle headroom,
  so the next signal should keep that action space fixed and make the router
  trainable/calibrated first
- concrete next options:
  - repeat the d3 action-motif/global-edit signal over seeds to test
    reproducibility
  - use benefit/harm labels directly for route calibration instead of only
    baseline-failure or identity-vs-nonzero BCE
  - if router work continues, train it against actual selected-action
    benefit/harm, not only whether any nonzero oracle candidate exists

## 17. Immediate Implementation Order

The early tasks above are now complete.

Current next tasks should be executed in this order.

1. keep `tools/build_pymatching_edit_targets.py` as the oracle / derived-target
   layer
2. keep `decoders/syndrome_edit_predecoder.py` as the current SEDP-v1 baseline
   plus first candidate-selector follow-up
3. treat the completed selector pilot reruns as a negative result for a purely
   post-hoc selector layer
4. treat the completed identity-vs-target ranking-loss pilot reruns as a
   negative result for that first in-training decision-aware recipe as well
5. treat the completed group-rank selector pilot reruns as a negative result
   for stronger selector-only fitting on the same candidate pool
6. treat the first local-motif placement/gating rerun as a completed partial
   result: locality exists, but safe held-out improvement does not
7. treat the first local-motif selector rerun as a completed negative/partial
   result: local placement candidate oracle is high, but learned emit decisions
   are not calibrated
8. treat the first hard-shot-router rerun as a completed negative/partial
   result: the architecture is wired, but the 256-shot pilot has too few
   positive route labels for a calibrated router
9. treat the first 1024-shot-per-family router rerun as a completed negative
   result for simply increasing same-recipe target size: oracle headroom
   remains, but the learned router still routes no eval shots
10. treat baseline-failure router pretraining plus balanced route batches as a
   completed negative/partial result: diagnostics change, but selected routing
   still routes no eval shots
11. treat the focused action-motif selected-mode rerun as a completed
   partial result: `d3` gets a small non-identity selected global-policy
   improvement, while `d5` remains identity/no-edit
12. next, run seed/reproducibility checks for that d3 signal or add
   benefit/harm calibration for selected nonzero edits
13. if that fails, move to benefit/harm-calibrated routing or a different
   PyMatching-assist mechanism before
   scaling out

## 18. Acceptance Criteria

`SEDP-v1` is worth keeping only if:

- oracle local-edit search shows real headroom over raw PyMatching
- learned edits improve held-out family decoding over raw PyMatching
- the model does not achieve gains only by applying large indiscriminate edits

Current status against those criteria:

- criterion 1: satisfied
- criterion 2: not yet satisfied
- criterion 3: not yet satisfied robustly

If those conditions fail, the next pre-decoder branch should move to:

- edge-weight modification
- erasure prediction
- or another PyMatching-assist mechanism

## 19. Short Practical Summary

The next mainline decoder branch should be:

- not another direct dense class4 decoder
- but a syndrome-edit neural pre-decoder plus unchanged PyMatching final decode

Why:

- it matches the current evidence
- it uses currently available truthful labels and artifacts
- it frames the neural model as an assistive system, which is where the current
  repository evidence is strongest

Next-session recommendation:

- continue this branch
- do **not** return to direct dense class4 tuning
- do **not** spend the next session on more sampling-only changes
- move to decision-aware objectives or edit-validity-constrained edits
