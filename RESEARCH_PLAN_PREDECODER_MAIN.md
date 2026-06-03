# Research Plan: Transition-Aware Neural Pre-Decoding For Surface-Code Logical Frames

This document fixes the current research topic, title candidates, contribution
claim, and execution plan.

## 0. Current Snapshot As Of 2026-05-02

Research topic:

> Transition-aware neural pre-decoding for surface-code logical-frame
> inference.

Primary goal:

- build a neural pre-decoder that edits or preserves the detector syndrome
  before PyMatching, so final `logical_class4` decoding beats raw PyMatching
  on held-out staged noise families
- treat Jung/Ali/Ha, "Convolutional Neural Decoder for Surface Codes", as the
  output/evaluation-format anchor, not as an exact architecture constraint

Current best system:

```text
syndrome volume
  -> neural candidate-conditioned patch-head pre-decoder / local edit selector
  -> edited or unchanged syndrome
  -> PyMatching
  -> logical_class4 prediction
```

Main empirical status:

- direct single-model `logical_class4` decoders are completed negative /
  secondary baselines: d3 partially learns, but d5/d7 collapse
- local-edit oracle headroom remains very high across d3/d5/d7, so candidate
  generation is not the main bottleneck
- d3 and d5 are unlocked by the learned patch-head selector plus
  non-inferiority selected-mode adoption:
  - d3 mean held-out `stage_c_corr` delta over seeds `0..3`:
    `+0.009521484`
  - d5 mean held-out `stage_c_corr` delta over seeds `0..3`:
    `+0.010253906`
- d7 is now guarded rather than solved:
  - selected no-edit guardrail with margin `0.005` changes d7 mean held-out
    `stage_c_corr` delta from `-0.004394531` to `+0.002197266`
  - d7 candidate oracle remains high, so this is still selected-path
    calibration rather than candidate availability
- 2026-05-03 reproducibility correction:
  - predecoder training now seeds numpy/torch at the training entry points;
    previous d7 seed sweeps fixed splits but did not fully fix model
    initialization, sampler order, or selector group shuffle
  - seed-fixed d7 seed `3` / seed `5` pilots both select `raw_no_edit`; seed
    `3` has only a one-shot post-hoc margin-`1.25` held-out gain that does not
    clear validation guardrail criteria
  - the old seed `3` d7 gain remains evidence that a high-gap local cluster can
    help, but it is not yet deterministic seed-stability evidence
- deterministic seed-fixed d7 `0..7` is now complete:
  - selected modes are `7/8` raw no-edit and `1/8` local selector
  - seed `2` is a validation false positive:
    `+0.012998091` validation delta but `-0.004882812` held-out
    `stage_c_corr` delta
  - seed `2` harm is low-margin over-editing; margin `1.0+` suppresses it to
    no-edit
  - validation delta over no-edit is useful but not sufficient as a d7 adoption
    rule under seed-controlled training
- d7 margin-floor recipe check:
  - rerunning seed `2` with selector emit margins restricted to `>=1.0`
    switches selected mode to `raw_no_edit`
  - held-out `stage_c_corr` delta becomes `+0.000000000`, confirming the
    margin floor blocks the false positive but does not recover learned d7 gain
- seed-fixed d5 revalidation is now complete:
  - no-guard selected-mode mean held-out `stage_c_corr` delta is
    `+0.001220703`, much weaker than the previous non-seed-fixed
    `+0.010253906`
  - seed `1` is a harmful global-policy false positive
    (`-0.018554688` held-out delta)
  - post-hoc no-edit guard margin `0.005` would raise d5 mean selected delta
    to `+0.005859375`
  - candidate branch mean remains `+0.011230469`, so d5 still has learned
    signal but selected-mode adoption is not stable enough
- seed-fixed d3 revalidation is now complete:
  - all four seeds select `local_motif_selector`
  - held-out `stage_c_corr` deltas are `+0.010742188`,
    `+0.006835938`, `+0.008789062`, and `+0.003906250`
  - mean selected/candidate delta is `+0.007568359`
- current seed-fixed distance ladder:
  - d3 selected: `+0.007568359`
  - d5 selected: `+0.001220703`
  - d5 post-hoc no-edit guard `0.005`: `+0.005859375`
  - d5 candidate branch: `+0.011230469`
  - d7 selected: `-0.000610352`
  - d7 margin-floor/no-edit safe policy: `+0.000000000`
- candidate-first adoption policy simulation is complete:
  - post-hoc policy preserves d3 at `+0.007568359`
  - raises d5 to `+0.011230469` by adopting seed `2`'s candidate branch and
    blocking harmful seed `1`
  - keeps d7 safe at `+0.000000000` by blocking the seed `2` false positive
  - this is not yet an integrated decoder result
- the current bottleneck is recovering d7 nonzero gains under the no-edit
  guardrail and improving d5 selected-mode adoption, not more feature addition
- 2026-05-04 d7 recovery update:
  - optional `--selector-epoch-selection-mode diagnostic_system` is now
    implemented while default epoch selection remains `proxy`
  - d7 identity-margin `0.5` + diagnostic epoch selection + unchanged
    `candidate_first_safety` adopts local selector on seeds `0` and `2`
  - mean held-out `stage_c_corr` selected delta over d7 seeds `0..7` is
    `+0.000854492`, versus `+0.000000000` for the canonical candidate-first
    no-gain sweep
  - this is the first seed-controlled d7 selected-mode learned gain, but it is
    still sparse and should be treated as a recovery signal rather than a
    solved d7 mechanism

Immediate next work:

- keep patch-head as the active representation
- keep selected no-edit guardrail active and consider it for d5 as well as d7
- treat low-margin d7 adoption as unsafe and test or keep a d7 margin floor
  (`>=1.0`) before selected-mode adoption
- improve selected-mode adoption/calibration enough to capture d5 seed `2`
  candidate gains while blocking harmful d5 seed `1` and d7 seed `2`
- use integrated `candidate_first_safety` as the active selected-mode policy
- continue d7 recovery by stabilizing high-margin positive selector clusters
  without relaxing adoption thresholds
- do not add more scalar feature branches until calibration is addressed

## 1. Research Decision

The main research direction is:

> A transition-aware neural pre-decoder that assists PyMatching for
> surface-code logical-frame decoding under staged circuit-level noise.

This is the mainline because current project evidence is asymmetric:

- direct single-model class4 decoders partially learn at `d3` but collapse at
  `d5` and `d7`
- PyMatching remains a strong baseline across `d3/d5/d7`
- bounded local detector-edit oracle search shows large headroom over raw
  PyMatching across `d3/d5/d7`
- the learned pre-decoder branch already produced the first selected held-out
  `d3 stage_c_corr` gain over raw PyMatching

The project should not treat the exact CNN architecture from the target paper as
mandatory. The target-paper role is the output/evaluation contract:

- rectangular or volume syndrome representation
- invalid lattice cells filled by an incoherent value
- final logical-frame output
- honest comparison against PyMatching

## 2. Working Paper Title

Primary title:

> Transition-Aware Neural Pre-Decoding for Surface-Code Logical Frame Inference

Alternative titles:

- Benefit-Harm Calibrated Neural Pre-Decoding for Surface Codes
- Learning Local Syndrome Edits for PyMatching-Assisted Surface-Code Decoding
- Logical-Transition-Aware Pre-Decoders for Robust Surface-Code Decoding

The primary title is preferred because it reflects the current technical
bottleneck: the model must distinguish beneficial logical transitions from
harmful ones, not merely emit more local edits.

## 3. Main Claim

The intended contribution is not "replace PyMatching with a neural network."

The intended contribution is:

> Learn a compact neural pre-decoder that proposes or selects local detector
> edits only when they are expected to improve final PyMatching logical-frame
> decoding, with explicit benefit/harm and logical-transition awareness.

The final decoder system is modular:

```text
syndrome volume
  -> neural pre-decoder / transition-aware edit selector
  -> edited or unchanged syndrome
  -> PyMatching
  -> logical_class4 evaluation
```

## 4. Novelty And Positioning

### Relation To Jung/Ali/Ha

Jung, Ali, Ha, "Convolutional Neural Decoder for Surface Codes" remains the
paper-format anchor:

- surface-code syndrome represented on a rectangular lattice
- invalid positions filled with an incoherent value
- CNN locality matched to surface-code geometry
- logical decoding performance compared against classical baselines

This project differs by studying a PyMatching-assist pre-decoder instead of
using a CNN as the final standalone decoder.

### Relation To NVIDIA Ising Decoding

NVIDIA Ising Decoding is a relevant recent reference because it validates the
practical value of neural pre-decoding for surface-code syndromes.

This project must not copy NVIDIA's model, wording, figures, or released
checkpoints. NVIDIA should be cited as related work, while this project keeps
its own contribution centered on:

- Bell-pair `logical_class4` supervision in this repository
- staged noise families: `ideal`, `stage_a_si1000`, `stage_b_local`,
  `stage_c_corr`
- explicit local-edit oracle headroom measurement
- benefit/harm scoring of candidate edits
- transition-aware candidate selection before PyMatching
- held-out noise-family evaluation under `stage_c_corr`

The safe positioning is:

> Inspired by the broader neural pre-decoding direction, we study a
> transition-aware candidate-selection problem for logical-frame decoding under
> staged noise families, and quantify how much local-edit oracle headroom can
> be recovered by a learned pre-decoder.

## 5. Current Empirical Baseline

Current PyMatching class4 accuracy on the 2k class4 manifests:

| dataset | ideal | stage_a_si1000 | stage_b_local | stage_c_corr |
| --- | ---: | ---: | ---: | ---: |
| d3/r3 2k | 1.000000000 | 0.937011719 | 0.917968750 | 0.925292969 |
| d5/r5 2k | 1.000000000 | 0.907226562 | 0.904296875 | 0.899902344 |
| d7/r7 2k | 1.000000000 | 0.891113281 | 0.868652344 | 0.874511719 |

Direct neural class4 decoder status:

- `FLFD`: learns some non-identity behavior at `d3`, collapses at `d5`
- `M3D-FLFD`: does not fix the `d5` collapse
- `FLFD-small d7`: collapses to degenerate all-`X` behavior

Local-edit oracle status on router1k target manifests:

- `d3 stage_c_corr`: raw PyMatching `0.928710938`, oracle near `0.992187500`
- `d5 stage_c_corr`: raw PyMatching `0.888671875`, oracle near `0.978515625`
- `d7 stage_c_corr`: raw PyMatching `0.873046875`, oracle near `0.984375000`

Learned pre-decoder status:

- benefit/harm selector with logical-transition features gives the first
  selected held-out `d3 stage_c_corr` gain:
  `0.928710938 -> 0.939453125`
- `d5/d7` still select no-edit / `global_policy` under the safe guardrail
- scalar calibration attempts on `d5` did not unlock the oracle headroom

## 6. Research Questions

Primary question:

> Can a neural pre-decoder recover a meaningful fraction of local-edit oracle
> headroom over PyMatching under distance scaling and noise-family shift?

Subquestions:

- Can the model improve held-out `stage_c_corr` on `d3`, `d5`, and `d7`?
- Is transition-aware candidate selection better than scalar emit-margin or
  nonzero-bias calibration?
- Which errors are improved and which are harmed relative to raw PyMatching?
- Does the method remain safe, meaning it does not over-edit already-correct
  shots?
- How much of the oracle gap remains after learned selection?

## 7. Success Criteria

Minimum publishable technical signal:

- reproduce the current `d3 stage_c_corr` selected gain across multiple seeds
- obtain at least one selected held-out `d5 stage_c_corr` improvement over raw
  PyMatching
- show that the selected method improves or preserves seen-family results under
  a no-harm guardrail

Strong result:

- selected improvement over raw PyMatching on `d3/d5/d7 stage_c_corr`
- clear oracle-gap recovery table
- robust benefit/harm analysis showing improved shots exceed harmed shots by a
  meaningful margin
- ablation showing transition-aware selection beats scalar calibration-only
  selector variants

Failure result worth documenting:

- direct neural decoders fail under distance scaling
- local-edit oracle headroom remains high
- learned transition-aware pre-decoding still cannot recover d5/d7 headroom

Even this negative result is scientifically useful if the experiments are
cleanly controlled.

## 8. Immediate Technical Plan

### Phase 1: Freeze Baselines

Status: mostly complete.

Tasks:

- keep PyMatching `d3/d5/d7` refresh artifacts as fixed baselines
- keep FLFD/M3D negative results in model-selection docs
- keep current benefit/harm d3/d5/d7 runs as pre-decoder baseline

Artifacts:

- `artifacts/eval/pymatching/d3_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d5_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d7_2k_class4_refresh.json`
- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_trans/experiment_summary.json`
- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_trans/experiment_summary.json`

### Phase 2: Reproduce d3 Gain

Goal:

- test whether the small selected `d3 stage_c_corr` gain is stable

Tasks:

- run the current benefit/harm transition-feature selector for at least 3 seeds
- report mean/std for `stage_c_corr`
- track improved vs harmed counts

Stop condition:

- if the d3 gain is not reproducible, do not claim learned improvement yet

Status as of 2026-04-27: complete for seeds `1,2,3`.

Artifacts:

- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans_seed1/experiment_summary.json`
- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans_seed2/experiment_summary.json`
- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans_seed3/experiment_summary.json`

Held-out `stage_c_corr` result:

| run | selected mode | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| existing run | local_motif_selector | 0.928710938 | 0.939453125 | +0.010742188 | 30 | 19 |
| seed 1 | local_motif_selector | 0.928710938 | 0.939453125 | +0.010742188 | 30 | 19 |
| seed 2 | local_motif_selector | 0.928710938 | 0.939453125 | +0.010742188 | 29 | 18 |
| seed 3 | local_motif_selector | 0.928710938 | 0.937500000 | +0.008789062 | 27 | 18 |

Mean over new seeds `1,2,3`:

- edited accuracy: `0.938802083`
- delta over PyMatching: `+0.010091146`
- population std of delta: `0.000920712`

Interpretation:

- the d3 selected gain is reproducible under this recipe
- the next blocker is no longer d3 reproducibility
- proceed to Phase 3: target-class-aware transition selector for d5

### Phase 3: Implement Target-Class-Aware Transition Selector

Goal:

- replace scalar calibration-only d5 work with a selector that explicitly
  models which logical transition is plausible for a shot

Design:

- keep the existing local candidate pool
- add shot-level prediction heads for:
  - target `logical_class4`, or
  - baseline-to-target transition class
- use candidate transition features already available in benefit/harm mode
- select nonzero edits only when candidate transition agrees with the learned
  shot-level transition signal

Training signal:

- supervised target from `logical_class4`
- auxiliary transition target from baseline PyMatching class to true class
- candidate benefit/harm target remains the final selection objective

Acceptance:

- d5 `stage_c_corr` selected mode beats raw PyMatching
- harms remain below improvements on held-out family

Status as of 2026-04-27: first implementation complete, acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- optional shot-level baseline-to-target transition-prior head
- target class: `baseline_class4 * 4 + logical_class4`
- CLI:
  - `--selector-transition-prior-weight-grid`
  - `--transition-prior-hidden-dim`
  - `--transition-prior-epochs`
  - `--transition-prior-lr`
- checkpoint save/load support for the transition-prior head
- selector grid search over emit margin, nonzero bias, and transition-prior
  weight

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_transprior/experiment_summary.json`

Held-out eval result:

| family | selected behavior | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |

Selected transition-prior settings:

- `selected_selector_transition_prior_weight = 0.0`
- `selected_selector_emit_margin = 2.0`
- selected inference mode remains `global_policy`

Post-hoc forced-emission check on the saved d5 checkpoint:

| forced setting | stage_c baseline | stage_c edited | delta | improved | harmed |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior `0`, margin `0` | 0.888671875 | 0.879882812 | -0.008789062 | 50 | 59 |
| prior `0.25`, margin `0` | 0.888671875 | 0.879882812 | -0.008789062 | 50 | 59 |
| prior `0.5`, margin `0` | 0.888671875 | 0.879882812 | -0.008789062 | 50 | 59 |
| prior `1`, margin `0` | 0.888671875 | 0.879882812 | -0.008789062 | 50 | 59 |
| prior `2`, margin `0` | 0.888671875 | 0.879882812 | -0.008789062 | 50 | 59 |

Interpretation:

- the transition-prior head trains but does not change the selected candidate
  decisions in this d5 recipe
- validation correctly rejects nonzero emission because forced edits improve
  some hard shots but harm more shots
- this does not recover the d5 local-edit oracle headroom
- the next model change should move from a separate transition prior to a
  stricter edit-validity / candidate-target compatibility constraint inside
  the selector objective or candidate set

### Phase 3b: Hard Transition-Compatibility Gate

Status as of 2026-04-27: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- optional hard compatibility gate:
  `--selector-transition-compat-top-k-grid`
- nonzero selector candidates are only selectable when their
  baseline-to-edited transition class appears in the shot-level transition
  prior's top-k predictions
- identity remains always selectable as the safety fallback

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_compat_topk/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- selected compatibility setting is `top_k = 0`, so validation rejects the hard
  gate as a useful improvement

Forced held-out `stage_c_corr` sweep on the saved checkpoint:

| forced setting | edited accuracy | delta | improved | harmed | edit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| margin `0`, top-k `0` | 0.879882812 | -0.008789062 | 50 | 59 | 0.106445 |
| margin `0`, top-k `1` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |
| margin `0`, top-k `2` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |
| margin `0`, top-k `4` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |
| margin `0`, top-k `8` | 0.878906250 | -0.009765625 | 34 | 44 | 0.076172 |
| margin `0`, top-k `16` | 0.879882812 | -0.008789062 | 50 | 59 | 0.106445 |

Interpretation:

- hard compatibility top-k gating is safe when narrow, but it becomes equivalent
  to no-edit
- wider top-k allows edits again but still harms more than it helps
- the current shot-level transition prior does not isolate the beneficial d5
  hard-shot transitions
- the next useful change should train compatibility directly on candidate-level
  beneficial-vs-harmful transitions, rather than relying on a global
  shot-level transition classifier

### Phase 3c: Candidate-Level Compatibility Head

Status as of 2026-04-27: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- optional candidate-level compatibility head:
  `--selector-candidate-compat-threshold-grid`
- training target is nonzero candidate beneficial-vs-harmful:
  `target_score > 0`
- identity remains always selectable
- nonzero candidates below the compatibility threshold are masked before final
  selector choice

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- selected compatibility threshold is effectively `0.0`

Compatibility diagnostics from the saved checkpoint:

- validation positive fraction is very small:
  - `stage_a_si1000`: `0.014706`
  - `stage_b_local`: `0.011268`
- predicted positive fraction is too large:
  - `stage_a_si1000`: `0.230328`
  - `stage_b_local`: `0.227464`

Forced held-out `stage_c_corr` sweep:

- thresholds `0.1` through `0.9` do not change the selected nonzero edits
- margin `0`: `0.888671875 -> 0.879882812`, improved `50`, harmed `59`
- margin `1`: `0.888671875 -> 0.878906250`, improved `34`, harmed `44`
- margin `2`: no edit, no change

Interpretation:

- the candidate-level compatibility head is wired but not calibrated
- positive beneficial nonzero candidates are too rare, and the head
  over-predicts compatibility
- simple BCE compatibility is therefore not enough
- the next useful change should rebalance or rank compatibility within each
  shot group, for example by training a pairwise beneficial-vs-harmful
  candidate loss only on hard-shot groups with at least one beneficial nonzero
  candidate

### Phase 3d: Group-Balanced Candidate Compatibility

Status as of 2026-04-27: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- candidate compatibility objective option:
  `--candidate-compat-objective group_balanced`
- balanced training samples:
  - keep beneficial nonzero candidates
  - sample harmful nonzero candidates around them with
    `--candidate-compat-negative-ratio`
  - keep a small number of negative-only groups via
    `--candidate-compat-no-positive-negative-count`

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_groupbal/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- selected emit margin is `1.0`

Compatibility diagnostic from the saved checkpoint:

| validation family | true positive fraction | predicted positive fraction |
| --- | ---: | ---: |
| stage_a_si1000 | 0.016619 | 0.002292 |
| stage_b_local | 0.010626 | 0.001377 |

Forced held-out `stage_c_corr` sweep:

| forced setting | edited accuracy | delta | improved | harmed | edit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| margin `0`, threshold `0` | 0.880859375 | -0.007812500 | 48 | 56 | 0.101562 |
| margin `0`, threshold `0.5` | 0.878906250 | -0.009765625 | 32 | 42 | 0.072266 |
| margin `1`, threshold `0` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |

Interpretation:

- group-balanced BCE reverses the flat-BCE failure mode: it becomes too
  conservative rather than too permissive
- it reduces emitted edits under stricter thresholds, but does not improve
  the improve/harm balance enough to beat PyMatching
- the next attempt should not be another absolute threshold classifier
- the next useful change is pairwise candidate ranking within the same shot:
  beneficial nonzero candidates must outrank harmful nonzero candidates, and
  selection should use relative compatibility rank rather than only sigmoid
  threshold

### Phase 3e: Pairwise Candidate Compatibility Ranking

Status as of 2026-04-27: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- candidate compatibility objective option:
  `--candidate-compat-objective pairwise_rank`
- inference option:
  `--selector-candidate-compat-top-k-grid`
- pairwise training loss:
  beneficial nonzero candidates should score above harmful nonzero candidates
  within the same shot group
- inference keeps only the top-k nonzero candidates by compatibility score
  before the final selector choice

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_pairwise/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- selected candidate compatibility top-k is `0`

Forced held-out `stage_c_corr` sweep:

| forced setting | edited accuracy | delta | improved | harmed | edit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| margin `0`, top-k `0` | 0.879882812 | -0.008789062 | 52 | 61 | 0.110352 |
| margin `0`, top-k `1` | 0.879882812 | -0.008789062 | 52 | 61 | 0.110352 |
| margin `0`, top-k `2` | 0.879882812 | -0.008789062 | 52 | 61 | 0.110352 |
| margin `0`, top-k `4` | 0.879882812 | -0.008789062 | 52 | 61 | 0.110352 |
| margin `2`, any top-k | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |

Interpretation:

- pairwise compatibility ranking is wired but does not change the harmful
  selected candidates
- the harmful candidates chosen by the main selector are already ranked high by
  the auxiliary compatibility head
- a detached auxiliary compatibility gate is therefore not enough
- the next useful change should merge compatibility into the main selector
  objective directly, so the selector score itself is trained to separate
  beneficial nonzero edits from harmful nonzero edits under the exact
  system-level selection rule

### Phase 3f: Main Selector Pairwise Benefit/Harm Ranking

Status as of 2026-04-27: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- direct main-selector pairwise term:
  `--selector-benefit-harm-pairwise-loss-weight`
- optional pairwise margin:
  `--selector-benefit-harm-pairwise-margin`
- loss is added inside the main group-rank selector objective:
  beneficial nonzero candidates should outrank harmful nonzero candidates
  within the same shot group

d5 router1k artifacts:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise_margin15/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise_w16/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch remains no-edit on the selected policies:
  `stage_c_corr 0.888671875 -> 0.888671875`

Important forced-sweep observation:

- one checkpoint with loss weight `1.0` showed a narrow full-eval emission band
  at margin `1.5`:
  - `stage_a_si1000`: `0.900390625 -> 0.901367188`
  - `stage_b_local`: `0.904296875 -> 0.907226562`
  - `stage_c_corr`: `0.888671875 -> 0.889648438`
- this did not reproduce when margin `1.5` was included in the validation grid
- with stronger pairwise weight `16.0`, emission still harms at low margin and
  collapses to no-edit at selected margins:
  - `stage_c_corr`, margin `0`: `0.888671875 -> 0.879882812`
  - `stage_c_corr`, margin `1.25+`: no edit

Interpretation:

- merging pairwise benefit/harm ranking into the main selector is wired
- it can create a narrow positive band in one checkpoint, but the signal is not
  stable enough for selected-mode d5 improvement
- increasing pairwise weight does not solve the problem; the pairwise loss
  rapidly becomes small while system-level harmful edits remain
- the next useful change is likely not another selector-side calibration term
  but a candidate-set restriction or richer candidate representation that
  changes which harmful edits are even available to the selector

### Phase 3g: Motif Evidence Merge For Duplicate Candidates

Status as of 2026-04-28: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- when a policy-generated candidate duplicates an oracle-derived motif/local
  motif candidate, keep the same edit mask but merge motif evidence into its
  candidate features
- specifically, duplicated candidates now preserve the policy flag while also
  setting the motif flag and motif-count feature
- rationale: d5 harmful selected edits were mostly high-confidence single-bit
  policy candidates; if some are also oracle motif candidates, selector should
  be able to see that provenance

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifmerge_pairwise/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch remains no-edit:
  `stage_c_corr 0.888671875 -> 0.888671875`

Forced held-out sweep:

| family | setting | edited accuracy | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: |
| stage_a_si1000 | margin `0` | 0.901367188 | +0.000976562 | 44 | 43 |
| stage_b_local | margin `0` | 0.903320312 | -0.000976562 | 48 | 49 |
| stage_c_corr | margin `0` | 0.881835938 | -0.006835938 | 52 | 59 |
| stage_c_corr | margin `2` | 0.888671875 | +0.000000000 | 0 | 0 |

Interpretation:

- motif evidence merging improves candidate representation but does not solve
  held-out d5 `stage_c_corr`
- the main harmful candidate issue is not simply missing motif provenance
- the next candidate-set step should be stronger: either disable/restrict raw
  policy candidates in the selector candidate pool, or generate candidates from
  oracle-derived motif/transition classes only

### Phase 3h: Motif-Only Candidate Pool

Status as of 2026-04-28: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- selector policy candidate mode:
  `--selector-policy-candidate-mode {all,none}`
- mode `none` keeps identity and motif/local-motif candidates but disables raw
  threshold/top-k policy candidates in the selector candidate pool

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifonly_pairwise/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`

Candidate pool diagnostic:

| family | candidate oracle accuracy | mean candidates/shot | selected behavior |
| --- | ---: | ---: | --- |
| stage_a_si1000 | 0.999023438 | 33.0 | no edit |
| stage_b_local | 0.996093750 | 33.0 | no edit |
| stage_c_corr | 0.999023438 | 33.0 | no edit |

Interpretation:

- disabling raw policy candidates removes the harmful policy emission path, but
  the selector still cannot identify which motif/local-motif placement to use
- oracle headroom remains very high, so candidate availability is not the
  blocker
- the likely blocker is candidate representation: current candidate features
  summarize edit probability statistics and motif count, but do not encode the
  selected detector location, relative placement, or motif pattern identity
  strongly enough
- next useful change: add geometry/placement-aware candidate features
  `(time,row,col)` summaries and/or local motif pattern id embeddings/features

### Phase 3i: Geometry/Placement-Aware Candidate Features

Status as of 2026-04-28: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- selector candidate geometry flag:
  `--selector-candidate-geometry-features`
- candidate features append normalized coordinate summaries for the selected
  edit indices:
  - mean `(time,row,col)`
  - std `(time,row,col)`
  - span `(time,row,col)`
- benefit/harm transition offsets now account for the longer candidate feature
  row, preserving compatibility with non-geometry checkpoints

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_geom_motifonly_pairwise/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`

Candidate pool diagnostic:

| family | candidate oracle accuracy | mean candidates/shot | selected behavior |
| --- | ---: | ---: | --- |
| stage_a_si1000 | 0.999023438 | 33.0 | no edit |
| stage_b_local | 0.996093750 | 33.0 | no edit |
| stage_c_corr | 0.999023438 | 33.0 | no edit |

Interpretation:

- absolute detector placement summaries are wired but do not unlock d5
- the selector still cannot choose useful motif/local-motif edits despite
  near-saturated oracle headroom
- the next representation step should encode local motif pattern identity and
  anchor-pattern structure, not only coordinate summaries

### Phase 3j: Local Motif Pattern/Anchor Candidate Features

Status as of 2026-04-28: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- selector candidate pattern flag:
  `--selector-candidate-pattern-features`
- candidate features append:
  - local-pattern-present flag
  - normalized local motif pattern id
  - log local motif pattern count
  - normalized anchor `(time,row,col)`

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patterngeom_motifonly_pairwise/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch selects no edit:
  `stage_c_corr 0.888671875 -> 0.888671875`
- selected selector emit margin is `2.0`

Candidate pool diagnostic:

| family | candidate oracle accuracy | mean candidates/shot | selected behavior |
| --- | ---: | ---: | --- |
| stage_a_si1000 | 0.999023438 | 33.0 | no edit |
| stage_b_local | 0.996093750 | 33.0 | no edit |
| stage_c_corr | 0.999023438 | 33.0 | no edit |

Interpretation:

- pattern identity and anchor metadata made the selector more willing to emit
  on validation, but the emitted candidates had negative mean selected target
  score
- the final margin guardrail correctly suppresses those harmful/non-beneficial
  emissions
- candidate metadata alone is not enough; the next representation step should
  include anchor-local syndrome/evidence features, such as selected detector
  events and small-neighborhood event/probability summaries

### Phase 3k: Anchor-Local Syndrome/Evidence Candidate Features

Status as of 2026-04-29: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- selector candidate local-evidence flag:
  `--selector-candidate-local-evidence-features`
- candidate features append:
  - selected-detector event/probability summaries
  - radius-1 anchor-neighborhood event/probability summaries

d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_localevidence_patterngeom_motifonly_pairwise/experiment_summary.json`

Selected result:

- selected inference mode remains `global_policy`
- candidate-selector branch emits sparse edits at margin `1.5`, but does not
  pass the final guardrail

Candidate-selector full-eval result:

| family | baseline PyMatching | edited PyMatching | improved | harmed | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | 0.900390625 | 0.900390625 | 1 | 1 | 0.999023438 |
| stage_b_local | 0.904296875 | 0.905273438 | 1 | 0 | 0.996093750 |
| stage_c_corr | 0.888671875 | 0.887695312 | 0 | 1 | 0.999023438 |

Interpretation:

- handcrafted local evidence changes selector behavior, unlike pure metadata
  features
- however, the generalization signal is still wrong: it slightly helps
  `stage_b_local` but harms held-out `stage_c_corr`
- the next useful step should be a learned candidate-conditioned local patch
  scorer that can read local syndrome/probability structure around the anchor,
  not another small scalar feature append

### Phase 3l: Candidate-Conditioned Local Patch Features

Status as of 2026-04-30: implemented and tested; d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- selector candidate local-patch flag:
  `--selector-candidate-local-patch-features`
- candidate features append a radius-1 anchor patch:
  - `3x3x3` relative detector-coordinate cells
  - per cell: raw detector event and pre-decoder edit probability
  - missing cells are zero-filled

Validation:

- `python -m py_compile decoders\syndrome_edit_predecoder.py project_status.py`

Short d5 smoke artifact:

- `artifacts/eval/nn/sedp_d5_smoke_localpatch/experiment_summary.json`

Smoke setting:

- `--max-shots 128`
- `--epochs 1`
- `--selector-epochs 1`
- `--selector-local-motif-max-classes 8`
- `--selector-local-motif-top-k 8`

Smoke result:

| family | baseline PyMatching | edited PyMatching | improved | harmed | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | 0.929687500 | 0.929687500 | 0 | 0 | 0.968750000 |
| stage_b_local | 0.890625000 | 0.890625000 | 0 | 0 | 0.937500000 |
| stage_c_corr | 0.921875000 | 0.921875000 | 0 | 0 | 0.921875000 |

Modest d5 artifact:

- `artifacts/eval/nn/sedp_d5_modest_localpatch/experiment_summary.json`

Modest setting:

- `--max-shots 512`
- `--epochs 4`
- `--selector-epochs 3`
- `--selector-local-motif-top-k 16`

Modest candidate-selector full-eval result:

| family | baseline PyMatching | edited PyMatching | improved | harmed | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ideal | 1.000000000 | 1.000000000 | 0 | 0 | 1.000000000 |
| stage_a_si1000 | 0.902343750 | 0.902343750 | 0 | 0 | 0.998046875 |
| stage_b_local | 0.912109375 | 0.914062500 | 1 | 0 | 0.996093750 |
| stage_c_corr | 0.892578125 | 0.892578125 | 0 | 0 | 0.998046875 |

Full d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_localpatch_patterngeom_motifonly_pairwise/experiment_summary.json`

Full selected result:

- selected inference mode remains `global_policy`
- validation chose selector emit margin `2.0`, suppressing all selector edits

Full candidate-selector full-eval result:

| family | baseline PyMatching | edited PyMatching | improved | harmed | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ideal | 1.000000000 | 1.000000000 | 0 | 0 | 1.000000000 |
| stage_a_si1000 | 0.900390625 | 0.900390625 | 0 | 0 | 0.999023438 |
| stage_b_local | 0.904296875 | 0.904296875 | 0 | 0 | 0.996093750 |
| stage_c_corr | 0.888671875 | 0.888671875 | 0 | 0 | 0.999023438 |

Interpretation:

- this is still represented as candidate features consumed by the existing
  selector MLP, not yet a separate convolutional patch scorer
- it is the minimal next step after handcrafted local-evidence summaries:
  preserve local spatial arrangement first, then only add a separate learned
  patch module if this representation is insufficient
- the smoke and modest runs prove the CLI/training/eval plumbing
- the modest run shows a tiny seen-family improvement without held-out harm,
  but the full router1k run reverts to no-edit under the validation guardrail
- appended local-patch features are therefore a completed negative/neutral
  d5 ablation
- next step should be a true learned candidate-conditioned patch scorer/head
  or a selector objective change, not another small feature append

### Phase 3m: Patch-Head Candidate Selector

Status as of 2026-05-01: implemented and tested; full d5 acceptance not met.

Implemented in `decoders/syndrome_edit_predecoder.py`:

- `CandidateEditSelector` now supports an optional patch branch
- new CLI:
  - `--selector-patch-head`
  - `--selector-patch-hidden-dim`
- when enabled, the selector:
  - extracts the local-patch slice from the flat candidate feature vector
  - encodes that patch slice with a small MLP
  - concatenates the patch embedding with shot features and non-patch
    candidate features

Smoke artifact:

- `artifacts/eval/nn/sedp_d5_smoke_patchhead_v3/experiment_summary.json`

Smoke setting:

- `--max-shots 128`
- `--epochs 1`
- `--selector-epochs 1`
- `--selector-candidate-local-patch-features`
- `--selector-patch-head`
- `--selector-patch-hidden-dim 32`

Smoke result:

- selected inference mode remains `global_policy`
- no selector edits were emitted on the tiny eval split
- summary metadata records `selector_patch_head=True` and
  `selector_patch_hidden_dim=32`

Modest d5 artifact:

- `artifacts/eval/nn/sedp_d5_modest_patchhead/experiment_summary.json`

Modest selected result:

- selected inference mode becomes `local_motif_selector`

| family | baseline PyMatching | edited PyMatching | improved | harmed | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ideal | 1.000000000 | 1.000000000 | 0 | 0 | 1.000000000 |
| stage_a_si1000 | 0.902343750 | 0.914062500 | 8 | 2 | 0.998046875 |
| stage_b_local | 0.912109375 | 0.933593750 | 11 | 0 | 0.996093750 |
| stage_c_corr | 0.892578125 | 0.898437500 | 11 | 8 | 0.998046875 |

Full d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_patterngeom_motifonly_pairwise/experiment_summary.json`

Full selected result:

- selected inference mode returns to `global_policy`
- selected selector emit margin is `4.0`
- candidate-selector branch emits no edits on eval families

| family | baseline PyMatching | edited PyMatching | improved | harmed | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ideal | 1.000000000 | 1.000000000 | 0 | 0 | 1.000000000 |
| stage_a_si1000 | 0.900390625 | 0.900390625 | 0 | 0 | 0.999023438 |
| stage_b_local | 0.904296875 | 0.904296875 | 0 | 0 | 0.996093750 |
| stage_c_corr | 0.888671875 | 0.888671875 | 0 | 0 | 0.999023438 |

Interpretation:

- this is the first actual patch-head mechanism, not just another flat feature
  append
- it is also the first d5 representation/objective variant in this sequence to
  produce a selected positive held-out signal at modest scale
- however, the full router1k run still falls back to no-edit under the
  validation guardrail
- next step should change selector objective/calibration for rare beneficial
  nonzero edits, not add another feature branch

### Phase 3n: Patch-Head Seed Sweep and PyMatching Comparison

Status as of 2026-05-01: completed; first full d5 PyMatching-beating selected
runs found, but not yet seed-stable.

Full d5 patch-head artifacts:

- seed 0:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_patterngeom_motifonly_pairwise/experiment_summary.json`
- seed 1:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed1/experiment_summary.json`
- seed 2:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed2/experiment_summary.json`
- seed 3:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed3/experiment_summary.json`

Selected-policy held-out `stage_c_corr` comparison:

| seed | selected mode | PyMatching | selected decoder | delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | global_policy | 0.888671875 | 0.888671875 | +0.000000000 |
| 1 | local_motif_selector | 0.888671875 | 0.895507813 | +0.006835938 |
| 2 | global_policy | 0.888671875 | 0.888671875 | +0.000000000 |
| 3 | local_motif_selector | 0.888671875 | 0.896484375 | +0.007812500 |

Mean selected delta over seeds `0..3`:

- `+0.003662109`

Candidate-selector branch held-out `stage_c_corr` comparison:

| seed | PyMatching | candidate branch | delta | improved | harmed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |
| 1 | 0.888671875 | 0.895507813 | +0.006835938 | 26 | 19 |
| 2 | 0.888671875 | 0.893554688 | +0.004882813 | 5 | 0 |
| 3 | 0.888671875 | 0.896484375 | +0.007812500 | 15 | 7 |

Mean candidate-branch delta over seeds `0..3`:

- `+0.004882813`

Additional calibration checks:

- `artifacts/eval/nn/sedp_d5_modest_patchhead_idmargin/experiment_summary.json`
  over-emits and harms held-out `stage_c_corr`
- `artifacts/eval/nn/sedp_d5_modest_patchhead_harmmargin/experiment_summary.json`
  improves held-out `stage_c_corr`, but
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_harmmargin/experiment_summary.json`
  returns to no-edit under the full validation guardrail

Interpretation:

- patch-head is now the first full d5 route that produces selected
  PyMatching-beating runs
- the performance signal is real but not stable enough to call final
- the next work should stabilize selector calibration / selected-mode adoption
  across seeds, using patch-head as the active representation

### Phase 3o: Selector Adoption Non-Inferiority

Status as of 2026-05-01: implemented; full d5 seed sweep completed.

Code change:

- `decoders/syndrome_edit_predecoder.py` supports
  `--selector-adoption-min-delta`
- default `0.0` adopts the requested selector when validation system metric is
  non-inferior to global policy, instead of requiring strict improvement
- positive values recover a more conservative adoption guardrail

Evidence:

- existing seed `2` patch-head summary had validation selector/global tie, so
  the previous strict `>` guardrail selected `global_policy`
- its held-out candidate branch was already positive:
  `stage_c_corr 0.888671875 -> 0.893554688`, improved `5`, harmed `0`
- fresh full d5 seed sweep artifacts:
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed1_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed2_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed3_adopt_noninferior/experiment_summary.json`

Selected-policy held-out `stage_c_corr` comparison:

| sweep | adopted seeds | mean selected delta | total improved | total harmed |
| --- | ---: | ---: | ---: | ---: |
| old strict adoption | 2/4 | +0.003662109 | 41 | 26 |
| new non-inferiority adoption | 4/4 | +0.010253906 | 66 | 24 |

New non-inferiority seed details:

| seed | selected mode | PyMatching | selected decoder | delta | improved | harmed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | local_motif_selector | 0.888671875 | 0.888671875 | +0.000000000 | 1 | 1 |
| 1 | local_motif_selector | 0.888671875 | 0.898437500 | +0.009765625 | 17 | 7 |
| 2 | local_motif_selector | 0.888671875 | 0.912109375 | +0.023437500 | 30 | 6 |
| 3 | local_motif_selector | 0.888671875 | 0.896484375 | +0.007812500 | 18 | 10 |

New per-family selected mean deltas over seeds `0..3`:

- `stage_a_si1000`: `+0.010986328`
- `stage_b_local`: `+0.009521484`
- `stage_c_corr`: `+0.010253906`

Interpretation:

- the immediate adoption instability was partly a strict-tie guardrail problem
- the fresh non-inferiority sweep selects the patch-head local selector in
  all four d5 seeds and improves the mean held-out `stage_c_corr` delta
- this is now the active selected-mode policy for the distance ladder
- next step should compare the active selected recipe on d3/d5/d7 against raw
  PyMatching and the local-edit oracle

### Phase 4: Distance Ladder

Status as of 2026-05-02: completed for active patch-head +
non-inferiority selected-mode adoption.

Artifacts:

- d3:
  - `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed0_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed1_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed2_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed3_adopt_noninferior/experiment_summary.json`
- d5:
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed1_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed2_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed3_adopt_noninferior/experiment_summary.json`
- d7 first active-recipe sweep:
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed1_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed3_adopt_noninferior/experiment_summary.json`
- d7 selected no-edit guardrail, margin `0.005`:
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_adopt_noninferior_noeditguard_margin005/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed1_adopt_noninferior_noeditguard_margin005/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_adopt_noninferior_noeditguard_margin005/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed3_adopt_noninferior_noeditguard_margin005/experiment_summary.json`

Held-out `stage_c_corr` mean over seeds `0..3`:

| distance | adopted seeds | PyMatching | learned pre-decoder | delta | oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | 4/4 | 0.928710938 | 0.938232422 | +0.009521484 | 1.000000000 |
| d5 | 4/4 | 0.888671875 | 0.898925781 | +0.010253906 | 0.999023438 |
| d7 no guard | 2/4 | 0.873046875 | 0.868652344 | -0.004394531 | 0.961181641 |
| d7 no-edit guard 0.005 | 1/4 selector, 3/4 raw | 0.873046875 | 0.875244141 | +0.002197266 | 0.988037109 |

Per-family mean deltas:

| distance | stage_a_si1000 | stage_b_local | stage_c_corr |
| --- | ---: | ---: | ---: |
| d3 | +0.022460938 | +0.039794922 | +0.009521484 |
| d5 | +0.010986328 | +0.009521484 | +0.010253906 |
| d7 no guard | -0.004394531 | -0.007324219 | -0.004394531 |
| d7 no-edit guard 0.005 | -0.000488281 | +0.001220703 | +0.002197266 |

Interpretation:

- d3 and d5 now show selected PyMatching-beating behavior under the same
  active recipe
- d7 no-edit guardrail fixes the harmful `global_policy` fallback failure mode
- with margin `0.005`, d7 becomes slightly positive on held-out `stage_c_corr`
  but mostly selects raw no-edit
- seed/checkpoint calibration matters: absolute selected validation metric
  chooses d7 seed `0` (`raw_no_edit`, held-out delta `+0.000000000`), while
  validation improvement over no-edit chooses seed `3` (`local_motif_selector`,
  held-out `stage_c_corr 0.873046875 -> 0.881835938`, delta `+0.008789062`)
- next useful change should stabilize this d7 nonzero local-selector recovery
  under the guardrail, not add another feature branch

Selection-calibration artifact:

- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_noeditguard_margin005_selection_compare.json`
- extended seed comparison:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_noeditguard_margin005_selection_compare_seed0_7.json`

Selection-calibration tool:

- `tools/compare_predecoder_seed_sweep.py`

Extended d7 guard005 seed check, seeds `0..7`:

| criterion | chosen seed | mode | held-out `stage_c_corr` delta |
| --- | ---: | --- | ---: |
| absolute selected validation metric | 4 | raw_no_edit | +0.000000000 |
| selected validation delta over no-edit | 3 | local_motif_selector | +0.008789062 |
| candidate validation delta over no-edit | 3 | local_motif_selector | +0.008789062 |

8-seed interpretation:

- selected-mode count is `raw_no_edit` `7/8`, `local_motif_selector` `1/8`
- mean selected held-out `stage_c_corr` delta is `+0.001098633`
- seed `5` has candidate validation delta `+0.003249608` but held-out
  candidate delta `-0.009765625`, so the `0.005` no-edit guard margin is
  justified
- the current d7 recipe is therefore safe but not stable; the next d7 work
  should explain the seed `3` / seed `5` separation before adding features

Seed `3` / seed `5` per-shot diagnostic:

- tool: `tools/diagnose_predecoder_selection.py`
- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed3_vs_seed5_stagec_selection_diagnostic.json`
- seed `3`: selector margin `1.25`, selected `17` nonzero edits on held-out
  `stage_c_corr`, with `13` positive-target-score edits and `4` negative ones
- seed `5`: selector margin `0.0`, candidate branch selected `68` nonzero
  edits, with `29` positive-target-score edits and `39` negative ones
- seed `5` best-nonzero logit-gap max is `1.209157`, below seed `3`'s adopted
  margin `1.25`; the harmful seed is a broad low-gap over-edit case
- seed `5` held-out harms concentrate on non-I classes:
  `Z->I|target=Z` `26` times and `Y->X|target=Y` `9` times

Post-hoc margin floor check:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_7_stagec_margin125_diagnostic.json`
- selector emit margin `1.25` preserves seed `3` and suppresses seed `5`
- all other d7 guarded seeds also emit no nonzero edits at margin `1.25`
- this confirms the margin-scale diagnosis, but it is still only a safety
  calibration; it does not recover additional d7 nonzero seeds

Margin-sweep calibration:

- summary artifact:
  `artifacts/eval/nn/sedp_d7_seed3_seed5_margin_sweep_summary.json`
- seed `3` validation chooses margin `1.25` with mean delta
  `+0.006493506`; the same margin gives held-out `stage_c_corr`
  `+0.008789062`
- seed `5` validation best is margin `0.0`, but its mean delta is only
  `+0.003246753`, below the active no-edit guard margin `0.005`; the held-out
  `stage_c_corr` candidate delta at that margin is `-0.009765625`
- therefore the active `0.005` guard margin is empirically justified for this
  seed-pair separation; lowering it would admit a harmful held-out seed

8-seed fixed-margin profile:

- artifact:
  `artifacts/eval/nn/sedp_d7_seed0_7_margin125_validation_heldout_profile.json`
- fixed selector margin `1.25` leaves only seed `3` with validation nonzero
  edits (`+0.006493506`, `4/2` improved/harmed)
- the same seed is the only held-out `stage_c_corr` nonzero seed at this
  margin (`+0.008789062`, `13/4` improved/harmed)
- seed `5` held-out max best-nonzero gap is `1.209157`, below the floor;
  seed `3` reaches `1.923424`
- current interpretation: the guard calibration is defensible, but the
  unresolved problem is training/score-scale stability, not candidate
  availability

## Integrated Candidate-First Adoption Policy

The candidate-first safety policy is now implemented in
`decoders/syndrome_edit_predecoder.py` as
`--selector-adoption-policy candidate_first_safety`. The default policy remains
`global_noninferiority`, so old runs are not changed unless the new policy is
explicitly selected.

Actual integrated d5 seed `0..3` policy sweep:

- artifacts:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise/experiment_summary.json`
  through seed `3`
- comparison:
  `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_3.json`
- selected held-out `stage_c_corr` mean delta:
  `+0.011230469`
- selected modes:
  raw no-edit for seeds `0,1`; local selector for seeds `2,3`

The d5 result matches the earlier post-hoc candidate-first simulation and
recovers the candidate-branch mean as a real selected-mode result.

d7 seed `2` integrated-policy smoke:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_smoke_pairwise/experiment_summary.json`
- selected mode: `raw_no_edit`
- selected held-out `stage_c_corr` delta: `+0.000000000`
- candidate branch delta: `-0.004882812`

This confirms the integrated policy blocks the known seed-fixed d7
low-margin false positive.

Full canonical d7 integrated-policy sweep:

- artifacts:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise_seq/experiment_summary.json`
  through seed `7`
- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json`
- all seeds select `raw_no_edit`
- mean selected held-out `stage_c_corr` delta: `+0.000000000`
- mean candidate-branch held-out `stage_c_corr` delta: `-0.000854492`

This completes the d7 safety check for the integrated policy. The result is
safe, but it is still not a learned d7 improvement.

Canonical d3 integrated-policy regression:

- artifacts:
  `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise_seq/experiment_summary.json`
  through seed `3`
- comparison:
  `artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_3.json`
- all seeds select `local_motif_selector`
- selected held-out `stage_c_corr` mean delta: `+0.007568359`
- distance ladder summary:
  `artifacts/eval/nn/sedp_candidatefirst_distance_ladder_summary.json`

D7 recovery epoch diagnostics:

- tool:
  `tools/summarize_selector_epoch_diagnostics.py`
- comparison artifact:
  `artifacts/eval/nn/sedp_d7_recovery_epoch_diagnostic_comparison.json`
- d7 has only `6` positive nonzero epoch/margin rows, versus d5 `14` and d3
  `66`
- d7 has only `3` margin `>=1` positive rows, and none meet the high-margin
  tied-evidence rule with at least `6` validation nonzero edits
- best d7 validation row is seed `2`, epoch `4`, margin `0.0`, delta
  `+0.012987013`; this is low-margin evidence and corresponds to the known
  harmful held-out seed `2` candidate branch
- conclusion: d7 recovery now needs score/training stability that creates
  robust high-margin positive clusters; relaxing adoption thresholds is not
  justified by the current diagnostics

D7 diagnostic epoch-selection recovery check:

- code:
  `--selector-epoch-selection-mode diagnostic_system`
- artifacts:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_idmargin05_diagselect_pairwise_seq/experiment_summary.json`
  through seed `7`
- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_7.json`
- epoch diagnostic summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_epoch_diagnostic_summary_seed0_7.json`
- result:
  - selected modes: `2/8` local selector, `6/8` raw no-edit
  - mean held-out `stage_c_corr` selected delta: `+0.000854492`
  - mean held-out candidate-branch delta: `+0.000488281`
  - adopted seeds are `0` (`+0.001953125`) and `2` (`+0.004882812`)
- interpretation:
  this creates the first seed-controlled d7 selected-mode positive signal, but
  it is still sparse; the next step should compare conservative diagnostic
  epoch-selection / identity-margin variants while keeping candidate-first
  adoption thresholds fixed

D7 identity-margin weight sentinel ablation:

- compared identity-margin loss weights `0.25`, `0.5`, and `1.0` on sentinel
  seeds `0,2,5`
- artifacts:
  - `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin025_diagselect_selection_compare_seed0_2_5.json`
  - `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_2_5.json`
  - `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin10_diagselect_selection_compare_seed0_2_5.json`
- held-out `stage_c_corr` mean selected deltas on the sentinel seeds:
  - weight `0.25`: `+0.000651042`, but seed `0` is a harmful selected false
    positive at `-0.002929688`
  - weight `0.5`: `+0.002278646`, no harmful selected sentinel seed
  - weight `1.0`: `+0.000000000`, all seeds raw no-edit
- conclusion:
  identity-margin weight `0.5` remains the active setting; full sweeps of
  `0.25` or `1.0` are not justified by the sentinel result

D7 small-volume epoch-selection probe:

- support-aware diagnostic tie-break was checked post-hoc on existing
  seed `0,2,5` epoch records and did not materially change choices
- seed `2` was rerun with epoch diagnostic grid
  `0.0 1.0 1.25 1.5`
- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_diagselect_grid015_pairwise_seq/experiment_summary.json`
- result:
  selector epoch `6`, adoption margin `1.25`, validation nonzero `5`, and
  held-out `stage_c_corr` selected delta `+0.004882812`, unchanged from the
  current best seed `2` run
- conclusion:
  do not expand the wider diagnostic-grid probe; usage should be saved for a
  different low-cost calibration axis

D7 selector-epoch count probe:

- seed `2` was rerun with `--selector-epochs 8` because the active seed `2`
  recipe selected the final selector epoch `6`
- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_diagselect_epochs8_pairwise_seq/experiment_summary.json`
- result:
  selected epoch `6`, margin `1.25`, validation nonzero `5`, and held-out
  `stage_c_corr` selected delta `+0.004882812`, unchanged from default
  6-epoch training
- diagnostic detail:
  epoch `7` has a validation tie with more support, but epoch `8` shows
  over-edit risk at margin `1.25`
- conclusion:
  keep default selector epochs `6`; do not expand this probe

### Phase 5: Paper-Style Writeup Assets

Tasks:

- method diagram
- dataset/noise table
- direct-neural negative-result table
- oracle headroom table
- learned pre-decoder result table
- ablation table:
  - no transition features
  - benefit/harm only
  - nonzero-bias calibration
  - harm-margin calibration
  - transition-aware selector

## 9. Single-Model Branch Position

Single-model decoding is not deleted. It is a secondary, timeboxed branch.

Candidate directions:

- graph neural logical-frame decoder
- spatio-temporal transformer
- stronger 3D CNN with two observable heads

However, it should not replace the mainline unless it clears a strict bar:

- no degenerate collapse at `d5`
- meaningful balanced accuracy / macro-F1 over FLFD/M3D
- plausible path to beating PyMatching on held-out `stage_c_corr`

Until then, single-model work remains exploratory.

## 10. Next Action

Recommended immediate next step:

1. treat integrated `candidate_first_safety` as the active d5 selected-mode
   policy
2. treat integrated `candidate_first_safety` as preserving the stable d3
   positive result
3. keep the d7 identity-margin `0.5` + diagnostic epoch-selection recipe with
   `positive_max_harmed=2` and `positive_max_margin=1.5`
4. treat d7 nonzero local-selector recovery as the current bottleneck because
   seed `8` and seed `13` showed unsafe false-positive modes
5. keep identity-margin weight `0.5`; the sentinel ablation rejected `0.25` as
   unsafe and `1.0` as too conservative
6. compare any d7 calibration change against both guarded no-edit
   (`+0.000000000`) and the new diagnostic epoch-selection result
   (`+0.000854492`), not only raw PyMatching or oracle headroom

Next execution protocol:

- continue d7 robustness extension with
  `--selector-candidate-first-positive-max-harmed 2` and
  `--selector-candidate-first-positive-max-margin 1.5`
- do not continue new out-of-sample seeds `18..23` yet; seed `17` hit the
  harmful selected stop condition
- stop immediately if any selected seed is harmful on held-out `stage_c_corr`
- do not add another guardrail before a new failure case is observed
- do not continue the old no-harm-cap or cap1-only d7 extension

2026-05-05 update:

- the delta-first high-margin post-hoc rule did not change the existing d7
  epoch choices, so it was not implemented
- the old d7 recovery recipe was extended to seed `8`
- seed `8` exposed a harmful selected false positive:
  validation delta `+0.006481003`, validation improved/harmed `6/4`, selected
  margin `2.0`, held-out `stage_c_corr` delta `-0.019531250`
- candidate-first adoption added a positive-delta harmed-shot cap:
  `--selector-candidate-first-positive-max-harmed`; it was introduced with
  cap `1` and later calibrated to default `2`
- the guard preserves d7 seed `0` and seed `2`, blocks seed `8`, and is
  post-hoc compatible with d3/d5 candidate-first results
- actual d7 sentinel comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap1_selection_compare_seed0_2_8.json`
- over seeds `0,2,8`, mean selected held-out delta is `+0.002278646`
- seeds `9..11` with harm cap were selected-mode safe, but seed `11` shows the
  guard can block a held-out positive candidate branch
- seed `13` exposed a second false-positive type: harm cap only selected a
  margin-`1.75` candidate that was held-out harmful by one shot
- candidate-first adoption now also has a positive-delta max-margin cap:
  `--selector-candidate-first-positive-max-margin`, default `1.5`
- seed `13` rerun with this guard selects raw no-edit via
  `candidate_positive_delta_margin_guard`
- seeds `14..15` with both guards are raw no-edit and safe
- final mixed 0..15 guarded result:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_guarded_mixed_selection_compare_seed0_15.json`
- final mixed 0..15 guarded metrics:
  local selector `2/16`, mean selected held-out `stage_c_corr` delta
  `+0.000427246`, candidate-branch mean `-0.000854492`, harmful selected
  seed count `0`
- seed `11` versus seed `13` diagnostics show seed `11` is a true positive at
  margin `1.5`, while seed `13` is a sparse high-margin false positive at
  margin `1.75`
- the positive harmed-shot cap is now calibrated to default `2`, while the
  positive max-margin cap remains `1.5`
- seed `11` cap2 rerun:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed11_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
- current mixed 0..15 cap2 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_15.json`
- current cap2 metrics:
  local selector `3/16`, selected local seeds `0,2,11`, mean selected
  held-out `stage_c_corr` delta `+0.000671387`, candidate-branch mean
  `-0.000854492`, harmful selected seed count `0`
- seed `16` out-of-sample cap2 rerun:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed16_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
- seed `16` selected raw no-edit with validation candidate delta
  `0.000000000` and held-out selected delta `0.000000000`
- current mixed 0..16 cap2 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_16.json`
- current cap2 0..16 metrics:
  local selector `3/17`, mean selected held-out `stage_c_corr` delta
  `+0.000631893`, candidate-branch mean `-0.000804228`, harmful selected
  seed count `0`
- seed `17` out-of-sample cap2 rerun:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed17_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
- seed `17` selected local selector with validation delta `+0.009746037`,
  margin `1.25`, nonzero `5`, validation improved/harmed `4/1`, but held-out
  selected delta `-0.004882812` (`8/13`)
- current mixed 0..17 cap2 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_17.json`
- current cap2 0..17 metrics:
  local selector `4/18`, mean selected held-out `stage_c_corr` delta
  `+0.000325521`, candidate-branch mean `-0.001030816`, harmful selected
  seed count `1`
- margin-profile artifact:
  `artifacts/eval/nn/sedp_d7_margin_profile_seed0_2_8_11_13_17.json`
- plateau-guard post-hoc simulation:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_posthoc_seed0_17.json`
- plateau-guard hypothesis over seeds `0..17`: block positive-delta local
  selection if a higher emit margin still has validation delta
  `>= positive_delta`
- post-hoc result: blocks seed `17`, local selector `3/18`, mean selected
  delta `+0.000596788`, harmful selected seed count `0`
- d5 seed `3` compatibility:
  `artifacts/eval/nn/sedp_d5_margin_profile_seed3.json`
- d5 seed `3` is preserved by the plateau hypothesis; no higher margin has
  aggregate validation delta `>= positive_delta`
- d7 seed `18`: raw no-edit, validation candidate delta `+0.003250404`,
  held-out candidate delta `-0.001953125`
- d7 seed `19`: raw no-edit, validation and held-out candidate deltas `0`
- current cap2 0..19 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_19.json`
- plateau post-hoc 0..19:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_posthoc_seed0_19.json`
- cap2 current 0..19: local selector `4/20`, mean selected delta
  `+0.000292969`, harmful selected seed count `1`
- plateau post-hoc 0..19: local selector `3/20`, mean selected delta
  `+0.000537109`, harmful selected seed count `0`
- optional integrated plateau guard:
  `--selector-candidate-first-positive-plateau-guard`
- integrated seed `17` run:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed17_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_pairwise_seq/experiment_summary.json`
- integrated seed `17` result: selected mode `raw_no_edit`, reason
  `candidate_positive_delta_plateau_guard`, held-out selected delta
  `+0.000000000`
- integrated plateau 0..19:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_19.json`
- integrated plateau 0..19 metrics: local selector `3/20`, mean selected
  delta `+0.000537109`, candidate-branch mean `-0.001025391`, harmful
  selected seed count `0`
- compatibility verification:
  - d7 seed `11` with integrated plateau guard stays local selector, held-out
    delta `+0.003906250`
  - d5 seed `3` with integrated plateau guard stays local selector, held-out
    delta `+0.023437500`
- d7 seed `20` and seed `21` extension with integrated plateau guard:
  both select `raw_no_edit` and have held-out selected delta `0`
- integrated plateau 0..21:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_21.json`
- integrated plateau 0..21 metrics: local selector `3/22`, mean selected
  delta `+0.000488281`, candidate-branch mean `-0.000932173`, harmful
  selected seed count `0`
- d7 seed `22` and seed `23` extension with integrated plateau guard:
  both select `raw_no_edit` and have held-out selected delta `0`
- integrated plateau 0..23:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_23.json`
- integrated plateau 0..23 metrics: local selector `3/24`, selected local
  seeds `0,2,11`, mean selected delta `+0.000447591`, candidate-branch mean
  `-0.000854492`, harmful selected seed count `0`
- d7 seed `24` and seed `25` extension with integrated plateau guard:
  both select `raw_no_edit` and have held-out selected delta `0`
- integrated plateau 0..25:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_25.json`
- integrated plateau 0..25 metrics: local selector `3/26`, selected local
  seeds `0,2,11`, mean selected delta `+0.000413161`, candidate-branch mean
  `-0.000788762`, harmful selected seed count `0`
- d7 seed `26` and seed `27` extension with integrated plateau guard:
  seed `26` selects `raw_no_edit` but its candidate branch is harmful
  (`-0.011718750`); seed `27` selects `raw_no_edit` with candidate delta `0`
- integrated plateau 0..27:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_27.json`
- integrated plateau 0..27 metrics: local selector `3/28`, selected local
  seeds `0,2,11`, mean selected delta `+0.000383650`, candidate-branch mean
  `-0.001150949`, harmful selected seed count `0`
- d7 seed `28` and seed `29` extension with integrated plateau guard:
  seed `28` selects `raw_no_edit` but its candidate branch is held-out
  positive by one shot (`+0.000976562`); seed `29` selects `raw_no_edit`
- integrated plateau 0..29:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_29.json`
- integrated plateau 0..29 metrics: local selector `3/30`, selected local
  seeds `0,2,11`, mean selected delta `+0.000358073`, candidate-branch mean
  `-0.001041667`, harmful selected seed count `0`
- d7 seed `30` and seed `31` extension with integrated plateau guard:
  both select `raw_no_edit` and have held-out selected/candidate deltas `0`
- integrated plateau 0..31:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_31.json`
- integrated plateau 0..31 metrics: local selector `3/32`, selected local
  seeds `0,2,11`, mean selected delta `+0.000335693`, candidate-branch mean
  `-0.000976562`, harmful selected seed count `0`
- d7 seed `32` and seed `33` extension with integrated plateau guard:
  both select `raw_no_edit` by harm guard; candidate branches are harmful
  (`-0.010742188`, `-0.016601562`)
- integrated plateau 0..33:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_33.json`
- integrated plateau 0..33 metrics: local selector `3/34`, selected local
  seeds `0,2,11`, mean selected delta `+0.000315947`, candidate-branch mean
  `-0.001723346`, harmful selected seed count `0`
- d7 seed `34` and seed `35` extension with integrated plateau guard:
  seed `34` selects `raw_no_edit` but its candidate branch is harmful
  (`-0.002929688`); seed `35` selects `raw_no_edit`
- integrated plateau 0..35:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_35.json`
- integrated plateau 0..35 metrics: local selector `3/36`, selected local
  seeds `0,2,11`, mean selected delta `+0.000298394`, candidate-branch mean
  `-0.001708984`, harmful selected seed count `0`
- d7 seed `36` and seed `37` extension with integrated plateau guard:
  seed `36` selects `raw_no_edit` but its candidate branch is harmful
  (`-0.001953125`); seed `37` selects `raw_no_edit`
- integrated plateau 0..37:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_37.json`
- integrated plateau 0..37 metrics: local selector `3/38`, selected local
  seeds `0,2,11`, mean selected delta `+0.000282689`, candidate-branch mean
  `-0.001670436`, harmful selected seed count `0`
- d7 seed `38` and seed `39` extension with integrated plateau guard:
  seed `38` selects `raw_no_edit` but its candidate branch is harmful
  (`-0.001953125`); seed `39` selects `raw_no_edit`
- integrated plateau 0..39:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_39.json`
- integrated plateau 0..39 metrics: local selector `3/40`, selected local
  seeds `0,2,11`, mean selected delta `+0.000268555`, candidate-branch mean
  `-0.001635742`, harmful selected seed count `0`
- d7 seed `40` and seed `41` extension with integrated plateau guard:
  seed `40` selects `raw_no_edit`; seed `41` selects `raw_no_edit` but its
  candidate branch is harmful (`-0.000976562`)
- integrated plateau 0..41:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_41.json`
- integrated plateau 0..41 metrics: local selector `3/42`, selected local
  seeds `0,2,11`, mean selected delta `+0.000255766`, candidate-branch mean
  `-0.001581101`, harmful selected seed count `0`
- d7 seed `42` and seed `43` extension with integrated plateau guard:
  seed `42` selects `raw_no_edit`; seed `43` selects `raw_no_edit` while its
  candidate branch is positive (`+0.000976562`)
- integrated plateau 0..43:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_43.json`
- integrated plateau 0..43 metrics: local selector `3/44`, selected local
  seeds `0,2,11`, mean selected delta `+0.000244141`, candidate-branch mean
  `-0.001487038`, harmful selected seed count `0`
- d7 seed `44` and seed `45` extension with integrated plateau guard:
  seed `44` selects `raw_no_edit`; seed `45` selects `raw_no_edit` while its
  candidate branch is positive (`+0.000976562`)
- integrated plateau 0..45:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_45.json`
- integrated plateau 0..45 metrics: local selector `3/46`, selected local
  seeds `0,2,11`, mean selected delta `+0.000233526`, candidate-branch mean
  `-0.001401155`, harmful selected seed count `0`
- d7 seed `46` and seed `47` extension with integrated plateau guard:
  both select `raw_no_edit` with candidate delta `0`
- integrated plateau 0..47:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_47.json`
- integrated plateau 0..47 metrics: local selector `3/48`, selected local
  seeds `0,2,11`, mean selected delta `+0.000223796`, candidate-branch mean
  `-0.001342773`, harmful selected seed count `0`
- d7 seed `48` and seed `49` extension with integrated plateau guard:
  both select `raw_no_edit` with candidate delta `0`
- integrated plateau 0..49:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_49.json`
- integrated plateau 0..49 metrics: local selector `3/50`, selected local
  seeds `0,2,11`, mean selected delta `+0.000214844`, candidate-branch mean
  `-0.001289063`, harmful selected seed count `0`
- d7 seed `50` and seed `51` extension with integrated plateau guard:
  both select `raw_no_edit` with candidate delta `0`
- integrated plateau 0..51:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_51.json`
- integrated plateau 0..51 metrics: local selector `3/52`, selected local
  seeds `0,2,11`, mean selected delta `+0.000206581`, candidate-branch mean
  `-0.001239483`, harmful selected seed count `0`
- d7 seed `52` and seed `53` extension with integrated plateau guard:
  seed `52` selects `raw_no_edit`; seed `53` selects `raw_no_edit` by harm
  guard while its candidate branch is harmful (`-0.010742188`)
- integrated plateau 0..53:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_53.json`
- integrated plateau 0..53 metrics: local selector `3/54`, selected local
  seeds `0,2,11`, mean selected delta `+0.000198929`, candidate-branch mean
  `-0.001392506`, harmful selected seed count `0`
- d7 seed `54` extension with integrated plateau guard failed selected safety:
  seed `54` selects `local_motif_selector` at margin `1.25`; validation delta
  is `+0.006508300`, but held-out selected delta is `-0.006835938`; validation
  support is stage_a-only at the selected margin
- integrated plateau 0..54 failed:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_54_failed.json`
- integrated plateau 0..54 failed metrics: local selector `4/55`, selected
  local seeds `0,2,11,54`, mean selected delta `+0.000071023`,
  candidate-branch mean `-0.001491477`, harmful selected seed count `1`
- support-guard calibration:
  - implemented `--selector-candidate-first-positive-min-nonzero`
  - post-hoc min-nonzero `5` over 0..54 selects seeds `2,11`, mean selected
    delta `+0.000159801`, harmful selected seed count `0`
  - actual seed54 sentinel with min-nonzero `5` selects `raw_no_edit` by
    `candidate_positive_delta_support_guard`; held-out selected delta `0`
  - actual seed2/11/54 support-guard comparison:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
  - seed `2` and seed `11` remain local selector with held-out deltas
    `+0.004882812` and `+0.003906250`; seed `54` is blocked
  - support-guard mixed 0..54 sentinel:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_54_sentinel.json`
  - support-guard mixed 0..54 metrics: local selector `2/55`, selected seeds
    `2,11`, mean selected delta `+0.000159801`, harmful selected seed count `0`
  - seed `55` with support guard selects `raw_no_edit`; held-out selected
    delta `0`, candidate delta `-0.004882812`
  - support-guard mixed 0..55:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_55.json`
  - support-guard mixed 0..55 metrics: local selector `2/56`, selected seeds
    `2,11`, mean selected delta `+0.000156948`, harmful selected seed count `0`
  - seed `56` with support guard selects `raw_no_edit`; held-out selected
    delta `0`, candidate delta `-0.000976562`
  - support-guard mixed 0..56:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_56.json`
  - support-guard mixed 0..56 metrics: local selector `2/57`, selected seeds
    `2,11`, mean selected delta `+0.000154194`, candidate-branch mean
    `-0.001541941`, harmful selected seed count `0`, harmful candidate seed
    count `17`
  - seed `57` with support guard selects `raw_no_edit`; held-out selected
    delta `0`, candidate delta `0`
  - support-guard mixed 0..57:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json`
  - support-guard mixed 0..57 metrics: local selector `2/58`, selected seeds
    `2,11`, mean selected delta `+0.000151536`, candidate-branch mean
    `-0.001515356`, harmful selected seed count `0`, harmful candidate seed
    count `17`
- d7 support-guard candidate-oracle analysis:
  `artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json`
  - all checked seeds have positive candidate-oracle headroom
  - mean candidate-oracle delta `+0.096679688`
  - actual candidate mean delta `-0.001515356`
  - candidate outcomes: `6` positive, `35` neutral, `17` harmful
- true/false selected-shot diagnostic:
  `artifacts/eval/nn/sedp_d7_support_guard_true_false_selection_diagnostic_seed2_11_54_stagec.json`
  - seed `54` remains harmful despite high oracle headroom, so the d7
    bottleneck is selector ranking/generalization rather than candidate-set
    coverage
- oracle/harm ranking diagnostic:
  `artifacts/eval/nn/sedp_d7_support_guard_oracle_harm_ranking_diagnostic_seed2_11_54_55_stagec.json`
  - seed `2` margin `1.25`: oracle above margin `6`, negative above margin
    `1`, selected delta `+0.004882812`
  - seed `11` margin `1.5`: oracle above margin `10`, negative above margin
    `6`, selected delta `+0.003906250`
  - seed `54` margin `1.25`: oracle above margin `6`, negative above margin
    `13`, selected delta `-0.006835938`
  - seed `55` margin `1.75`: oracle above margin `8`, negative above margin
    `13`, candidate delta `-0.004882812`
- optional hard-negative identity-margin loss was implemented and tested:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_negidmargin10_m15_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
  - weight `1.0`, margin `1.5` reduces seed `54` harm from `-0.006835938`
    to `-0.001953125`
  - the same setting destroys seed `2`, changing candidate delta from
    `+0.004882812` to `-0.003906250`
  - verdict: this hard-negative setting is too strong
- weaker hard-negative identity-margin setting was tested:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_negidmargin025_m10_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
  - weight `0.25`, margin `1.0` lets seed `54` through as
    `local_motif_selector`
  - held-out selected/candidate delta `-0.001953125`
  - ranking diagnostic:
    `artifacts/eval/nn/sedp_d7_negidmargin025_m10_oracle_harm_ranking_diagnostic_seed54_stagec.json`
  - at margin `1.0`, oracle-positive above margin `6`, negative-target above
    margin `9`
  - verdict: this hard-negative setting is unsafe
- correct-split validation ranking-guard check:
  `artifacts/eval/nn/sedp_d7_support_guard_validation_ranking_guard_summary_seed2_11_54_55.json`
  - validation diagnostics must use the training split convention:
    `stage_a_si1000` split seed `seed`, `stage_b_local` split seed `seed + 1`;
    default split seed `0` should not be used for adoption-policy conclusions
  - tested statistic: block adoption if validation negative-target above-margin
    count exceeds oracle-positive above-margin count at the candidate adoption
    margin
  - support-guard seed `2` and seed `11` are preserved with validation
    oracle/negative above-margin counts `4/1` and `3/2`
  - support-guard seed `54` is not blocked by the statistic: at margin `1.25`,
    validation oracle/negative above-margin counts are `2/0`, but held-out
    candidate delta is `-0.006835938`
  - support-guard seed `55` is also not blocked: validation counts are `2/1`,
    but held-out candidate delta is `-0.004882812`
  - weak hard-negative seed `54` also passes this statistic at margin `1.0`
    with combined validation counts `3/1`, despite held-out delta
    `-0.001953125`
  - verdict: simple validation negative-over-margin excess is rejected as an
    adoption guard
- hard positive-vs-negative ranking sentinel:
  - implemented optional selector loss flags:
    `--selector-positive-negative-hard-loss-weight` and
    `--selector-positive-negative-hard-margin`
  - the loss applies only when a shot group has both positive nonzero and
    negative nonzero candidates; it compares best-positive logit against
    hardest-negative logit
  - sentinel setting weight `1.0`, margin `0.5`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
  - seed `2` remains local selector but weakens from `+0.004882812` to
    `+0.003906250`
  - seed `11` candidate branch improves to `+0.004882812`, but selected mode
    becomes no-edit, so selected delta becomes `0`
  - seed `54` selected mode remains safe no-edit; candidate harm improves from
    `-0.006835938` to `-0.003906250`, but the candidate branch is still
    harmful
  - seed `54` stage_c diagnostic:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed54_stagec.json`
    - margin `1.5`: oracle-positive above margin `6`, negative-target above
      margin `10`
  - verdict: partial ranking improvement, not a final selected-mode recipe
- adoption recalibration for the `1.0/0.5` hard positive-vs-negative sentinel:
  - `tools/simulate_predecoder_adoption_policy.py` now supports
    `--positive-plateau-guard`
  - calibrated adoption artifact:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54.json`
  - with `positive_delta=0.003`, `positive_min_nonzero=1`, and plateau guard,
    seeds `2` and `11` are selected while seed `54` stays blocked
  - mean selected delta is `+0.002929688`, matching the old support-guard
    sentinel over seeds `2,11,54`
  - mean candidate delta improves from the old support-guard sentinel's
    `+0.000651042` to `+0.001627604`
- weaker hard positive-vs-negative sentinel:
  - weight `0.5`, margin `0.5` comparison:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard05_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
  - current adoption selects all no-edit; mean selected delta `0`
  - calibrated adoption:
    `artifacts/eval/nn/sedp_d7_posneghard05_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54.json`
  - only seed `11` is recovered; mean policy delta `+0.002278646`, and seed
    `2` candidate is harmful (`-0.003906250`)
  - verdict: reject `0.5/0.5`
- `1.0/0.5` extension to seeds `55,56,57`:
  - comparison artifact:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54_57.json`
  - calibrated adoption artifact:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54_57.json`
  - old support-guard over seeds `2,11,54,55,56,57`: mean selected
    `+0.001464844`, mean candidate `-0.000651042`
  - `1.0/0.5` calibrated adoption over the same seeds: mean selected
    `+0.001464844`, mean candidate `-0.002278646`
  - seed `55` and seed `57` are harmful candidate regressions:
    `-0.006835938` and `-0.011718750`
  - diagnostics:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed55_stagec.json`
    and
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed57_stagec.json`
  - at margin `1.5`, seed `55` has improved/harmed `19/26`, and seed `57`
    has improved/harmed `7/10`
  - verdict: reject broad extension of `1.0/0.5`
- simple family-level stage-consistency adoption check:
  - `tools/simulate_predecoder_adoption_policy.py` now records validation
    family-level candidate deltas/nonzero/improved/harmed counts and supports
    `--positive-family-min-delta`, `--positive-min-family-count`, and
    `--positive-max-family-harmed`
  - all-family nonnegative guard on `1.0/0.5` calibrated adoption:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_familymin0_count2_seed2_11_54_57.json`
  - result: seed `2` is blocked despite being a held-out true positive because
    validation is mixed (`stage_a=-0.006493506`,
    `stage_b=+0.025974026`); mean policy delta falls to `+0.000813802`
  - family harmed-cap `2`:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_familymaxharm2_seed2_11_54_57.json`
  - result: no additional discrimination beyond calibrated adoption
  - all-family nonnegative guard on the original support-guard recipe:
    `artifacts/eval/nn/sedp_d7_support_adoption_sim_posdelta003_posminnz1_plateau_familymin0_count2_seed2_11_54_57.json`
  - result: also blocks seed `2` and lowers mean policy delta to
    `+0.000651042`
  - verdict: simple family-level post-hoc adoption guards are rejected
- cross-family hard positive-vs-negative selector objective:
  - implemented default-off flags:
    `--selector-cross-family-positive-negative-loss-weight` and
    `--selector-cross-family-positive-negative-margin`
  - intent: move stage consistency into training by forcing a positive
    nonzero candidate from one train family to outrank a hard negative
    nonzero candidate sampled from another train family
  - weak seed `54` sentinel `0.25/0.5`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_crossfam025_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
    leaves the candidate branch unchanged at `-0.006835938`
  - strong seed `54` sentinel `1.0/0.5`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_crossfam10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
    worsens the candidate branch to `-0.009765625`
  - verdict: reject this simple cross-family hard-negative objective; it
    failed the false-positive seed54 gate
- next action is still target/ranking redesign, not another seed extension:
  hard-negative identity-margin alone is rejected, and simple validation
  negative-over-margin excess is also rejected; hard positive-vs-negative
  `1.0/0.5` and `0.5/0.5` are both rejected as broad d7 recipes. If
  continuing d7, the next attempt must introduce a stage-consistency-aware
  selector objective rather than another scalar ranking/adoption threshold,
  all-family validation guard, or this first cross-family hard-negative
  variant. Otherwise, consolidate the d3/d5 success and treat d7 as a
  documented scaling limitation.
- consolidation artifact now exists:
  `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`
  plus `PREDECODER_CONSOLIDATED_EVIDENCE.md`,
  `PREDECODER_FINAL_RESULT_TABLES.md`, and
  `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
  - d3 selected mean delta `+0.007568359` over seeds `0..3`
  - d5 selected mean delta `+0.011230469` over seeds `0..3`
  - d7 support-guard selected mean delta `+0.000151536` over seeds `0..57`
  - d7 mean candidate-oracle delta `+0.096679688`, but actual candidate mean
    delta `-0.001515356`
  - current claim boundary: positive selected d3/d5 result plus documented d7
    selector-ranking limitation
- d7 targeted bottleneck analysis:
  `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
  plus `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`
  - preserve true positives `2,11`
  - recover missed positives `0,28,43,45`
  - block validation false positives `13,17,33,54,53,32,8,18`
- d7 adoption-grid diagnostic:
  `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json`
  - checked `183040` simple policies
  - found `0` policies passing preserve/recover/block
  - best recovery opens harmful false positives `13,17,18,54`
- d7 candidate-compatibility pairwise top-k sentinel:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
  - blocks seed54
  - destroys seed2 candidate branch with candidate delta `-0.136718750`
  - reject this recipe; do not expand to seed11 or missed-positive recovery
    seeds
- remaining work is now tracked in `PREDECODER_REMAINING_WORK.md`:
  - freeze the final d3/d5/d7 result tables
  - finalize the method description and d3/d5 success structure
  - finish d7 as a selector-ranking/generalization limitation unless a new
    objective passes the sentinel gate
  - prepare the final reproducibility package
