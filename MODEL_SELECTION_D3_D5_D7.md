# Model Selection Across d3/d5/d7

This note records the 2026-04-24 decision after extending the current class4
noise scope to d7.

## Objective Contract

The main paper, Jung/Ali/Ha, "Convolutional Neural Decoder for Surface Codes",
is no longer treated as an architectural constraint. The required match is the
decoder task format:

- input: syndrome-derived surface-code information from the current noise
  environment
- output: `logical_class4 in {I, X, Z, Y}`
- comparison: PyMatching on the same shots
- judgement: distance scaling and held-out noise-family behavior

`baseline_rectcnn.py` remains a useful paper-style neural baseline, but future
research models do not need to copy that architecture.

## Current Class4 Distance Scope

Current datasets:

- `artifacts/datasets/dev_class4_2k/manifest.json`
- `artifacts/datasets/dev_class4_d5_2k/manifest.json`
- `artifacts/datasets/dev_class4_d7_2k/manifest.json`

Current PyMatching artifacts:

- `artifacts/eval/pymatching/d3_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d5_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d7_2k_class4_refresh.json`

## PyMatching Baseline

| dataset | ideal | stage_a_si1000 | stage_b_local | stage_c_corr |
| --- | ---: | ---: | ---: | ---: |
| d3/r3 2k | 1.000000000 | 0.937011719 | 0.917968750 | 0.925292969 |
| d5/r5 2k | 1.000000000 | 0.907226562 | 0.904296875 | 0.899902344 |
| d7/r7 2k | 1.000000000 | 0.891113281 | 0.868652344 | 0.874511719 |

This is now the current classical baseline table for the constructed class4
noise environment.

## Direct Neural Class4 Results

The strongest direct class4 neural line so far is the small FLFD variant. It is
useful on d3 but does not scale:

| model | dataset | ideal | stage_a_si1000 | stage_b_local | stage_c_corr | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FLFD-small | d3/r3 2k | 1.000000000 | 0.814453125 | 0.788574219 | 0.792968750 | learns non-I classes but remains below PyMatching |
| FLFD-small | d5/r5 2k | 1.000000000 | 0.783203125 | 0.772460938 | 0.761230469 | all-I collapse on noisy families |
| FLFD-small | d7/r7 2k | 0.000000000 | 0.183105469 | 0.181640625 | 0.195312500 | all-X collapse |

The M3D-FLFD successor did not improve this:

| model | dataset | ideal | stage_a_si1000 | stage_b_local | stage_c_corr | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M3D-FLFD | d3/r3 2k | 1.000000000 | 0.752929688 | 0.751464844 | 0.731933594 | worse than FLFD-small |
| M3D-FLFD | d5/r5 2k | 1.000000000 | 0.783203125 | 0.772460938 | 0.761230469 | all-I collapse |
| M3D-FLFD-strong | d5/r5 2k | 0.000000000 | 0.088378906 | 0.081054688 | 0.077148438 | all-Z collapse |

Current conclusion:

- direct class4 CNN replacement is not the most promising next direction
- increasing dense/multiscale trunk capacity did not fix scaling
- d7 confirms the collapse is a distance-scaling problem, not only a d5 artifact

## PyMatching-Assist Oracle Headroom

Local detector-edit target search with `max_edit_weight=2`,
`time_radius=1`, `space_radius=1`, and `max_candidates=20` remains strong across
d3/d5/d7:

| target set | family | baseline PyMatching | local-edit oracle | solved hard shots |
| --- | --- | ---: | ---: | ---: |
| d3/r3 1k | stage_a_si1000 | 0.928710938 | 0.996093750 | 69/73 |
| d3/r3 1k | stage_b_local | 0.906250000 | 0.996093750 | 92/96 |
| d3/r3 1k | stage_c_corr | 0.928710938 | 0.992187500 | 65/73 |
| d5/r5 1k | stage_a_si1000 | 0.900390625 | 0.984375000 | 86/102 |
| d5/r5 1k | stage_b_local | 0.904296875 | 0.990234375 | 88/98 |
| d5/r5 1k | stage_c_corr | 0.888671875 | 0.978515625 | 92/114 |
| d7/r7 1k | stage_a_si1000 | 0.889648438 | 0.977539062 | 90/113 |
| d7/r7 1k | stage_b_local | 0.868164062 | 0.978515625 | 113/135 |
| d7/r7 1k | stage_c_corr | 0.873046875 | 0.984375000 | 114/130 |

Current target artifacts:

- `artifacts/datasets/predecoder_targets_d3_2k_router1k/manifest.json`
- `artifacts/datasets/predecoder_targets_d5_2k_router1k/manifest.json`
- `artifacts/datasets/predecoder_targets_d7_2k_router1k/manifest.json`

## Model Choice

The most likely next model family is not a standalone direct neural replacement
for PyMatching. It is a PyMatching-assist model:

> Neural syndrome-edit / benefit-calibrated pre-decoder that proposes small,
> local detector edits and falls back to raw PyMatching unless expected benefit
> beats expected harm.

Reason:

- PyMatching is already strong and scales to d7 in the current noise scope.
- Direct neural class4 models lose badly to PyMatching and collapse with
  distance.
- Local-edit oracle headroom is consistent across d3/d5/d7.
- The missing piece is learned benefit/harm calibration, not another dense
  trunk or another sampling-only pre-decoder rerun.

## Benefit/Harm Selector Follow-Up

Implemented on 2026-04-24 in `decoders/syndrome_edit_predecoder.py`:

- `--selector-target-mode benefit_harm`
- candidate scores are now relative to raw PyMatching:
  - beneficial nonzero edit: positive utility
  - harmful edit when identity is already correct: negative utility
  - non-beneficial nonzero edit: weak negative utility
- benefit/harm mode also augments candidate features with logical-transition
  information available at inference:
  - baseline predicted observable delta
  - baseline class one-hot
  - edited class one-hot
  - baseline-to-edited class transition one-hot

Router1k pilot outputs:

- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_trans/experiment_summary.json`
- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_trans/experiment_summary.json`

Selected-mode results:

| target set | selected mode | family | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| d3/r3 1k | local_motif_selector | stage_a_si1000 | 0.928710938 | 0.951171875 | +0.022460938 | 39 | 16 |
| d3/r3 1k | local_motif_selector | stage_b_local | 0.906250000 | 0.949218750 | +0.042968750 | 60 | 16 |
| d3/r3 1k | local_motif_selector | stage_c_corr | 0.928710938 | 0.939453125 | +0.010742188 | 30 | 19 |
| d5/r5 1k | global_policy | stage_a_si1000 | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| d5/r5 1k | global_policy | stage_b_local | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| d5/r5 1k | global_policy | stage_c_corr | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |
| d7/r7 1k | global_policy | stage_a_si1000 | 0.889648438 | 0.889648438 | +0.000000000 | 0 | 0 |
| d7/r7 1k | global_policy | stage_b_local | 0.868164062 | 0.868164062 | +0.000000000 | 0 | 0 |
| d7/r7 1k | global_policy | stage_c_corr | 0.873046875 | 0.873046875 | +0.000000000 | 0 | 0 |

Interpretation:

- The transition features are necessary: without them, benefit/harm targets
  still selected identity-only on d3/d5/d7.
- With transition features, d3 finally produces a selected held-out
  `stage_c_corr` gain over raw PyMatching.
- d5/d7 still select identity under the validation guardrail, despite candidate
  oracle accuracy near `0.996-0.999`.
- The remaining bottleneck is therefore not candidate generation and not the
  benefit/harm label definition. It is distance-scaled selector calibration.

## Distance-Scaled Selector Calibration Follow-Up

Implemented and tested on 2026-04-24:

- selector nonzero-bias grid:
  `--selector-nonzero-bias-grid`
- selector harm-margin loss:
  `--selector-harm-margin-loss-weight`
  and `--selector-harm-margin`
- router label correction for benefit/harm runs:
  `baseline_failure` and `oracle_solvable` router targets now use actual
  candidate correctness instead of benefit/harm target-score thresholds

d5 router1k follow-up artifacts:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_biascal/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_harmmargin/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_hardw16/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_router_oracle/experiment_summary.json`

Key d5 `stage_c_corr` result:

| run | selected behavior | baseline PyMatching | edited PyMatching | improved | harmed | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| nonzero-bias calibration | validation rejects useful emission | 0.888671875 | 0.888671875 | 0 | 0 | forcing nonzero edits helps some shots but harms more |
| harm-margin `1.0` | no-edit selector | 0.888671875 | 0.888671875 | 0 | 0 | safe but too conservative |
| hard-shot weight `16` | no-edit selector | 0.888671875 | 0.888671875 | 0 | 0 | hard-shot upweighting alone does not calibrate |
| router `oracle_solvable` | no-edit selector | 0.888671875 | 0.888671875 | 0 | 0 | router probabilities collapse to an almost constant value |

Post-hoc sweep on the router checkpoint confirms that this is not only a
validation-threshold issue. The best accuracy remains raw PyMatching
`0.888671875`; forcing broad nonzero emission can improve `103` shots but harms
`856`, collapsing accuracy to `0.1533203125`.

Updated interpretation:

- d3 is learnable with the current transition-feature selector.
- d5 is not unlocked by scalar calibration knobs, hard-shot weighting, or the
  current factorized hard-shot router.
- The next useful model change should make candidate selection target-class or
  logical-transition aware, rather than only tuning emit thresholds.

## D3 Seed Reproducibility

Completed on 2026-04-27 for the existing benefit/harm transition-feature
selector.

Artifacts:

- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans_seed1/experiment_summary.json`
- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans_seed2/experiment_summary.json`
- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_trans_seed3/experiment_summary.json`

Held-out `stage_c_corr`:

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

- the d3 selected gain is reproducible enough to treat as a real signal
- the next bottleneck is d5 transition-aware candidate selection

## D5 Transition-Prior Selector Attempt

Implemented and tested on 2026-04-27 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-transition-prior-weight-grid`
- `--transition-prior-hidden-dim`
- `--transition-prior-epochs`
- `--transition-prior-lr`
- transition-prior checkpoint save/load during evaluation

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_transprior/experiment_summary.json`

Design:

- train a shot-level baseline-to-target transition prior with target
  `baseline_class4 * 4 + logical_class4`
- add the log transition prior to each selector candidate according to the
  candidate's baseline-to-edited transition class
- grid search transition-prior weight together with selector emit margin

d5 held-out result:

| family | selected behavior | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |

Selected settings:

- `selected_inference_mode = global_policy`
- `selected_selector_transition_prior_weight = 0.0`
- `selected_selector_emit_margin = 2.0`

Forced emission check on the saved checkpoint:

- prior weights `0, 0.25, 0.5, 1, 2` with margin `0` all produce the same
  d5 `stage_c_corr` result:
  `0.888671875 -> 0.879882812`, improved `50`, harmed `59`

Interpretation:

- the separate transition-prior head is wired and trains, but it does not
  change the selected d5 candidate decisions enough to improve decoding
- the safe selector is right to reject nonzero emission in this recipe
- the next useful step is not more prior-weight tuning; it is an explicit
  edit-validity or candidate-target compatibility constraint that prevents
  locally plausible but logically harmful edits

## D5 Hard Transition-Compatibility Gate

Implemented and tested on 2026-04-27 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-transition-compat-top-k-grid`

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_compat_topk/experiment_summary.json`

Design:

- keep identity always selectable
- for nonzero candidates, require the candidate's baseline-to-edited transition
  class to be in the shot-level transition prior's top-k predictions

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |

Selected compatibility setting:

- `selected_selector_transition_compat_top_k = 0`
- `selected_selector_emit_margin = 2.0`

Forced held-out `stage_c_corr` sweep:

| forced setting | edited accuracy | delta | improved | harmed | edit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| margin `0`, top-k `0` | 0.879882812 | -0.008789062 | 50 | 59 | 0.106445 |
| margin `0`, top-k `1` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |
| margin `0`, top-k `2` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |
| margin `0`, top-k `4` | 0.888671875 | +0.000000000 | 0 | 0 | 0.000000 |
| margin `0`, top-k `8` | 0.878906250 | -0.009765625 | 34 | 44 | 0.076172 |
| margin `0`, top-k `16` | 0.879882812 | -0.008789062 | 50 | 59 | 0.106445 |

Interpretation:

- hard top-k compatibility can suppress harmful edits, but in the useful narrow
  regime it suppresses all edits
- widening the gate reintroduces harmful edits
- the current shot-level transition classifier is too coarse for d5
- the next candidate should learn candidate-level transition compatibility or
  beneficial-vs-harmful nonzero classification directly

## D5 Candidate-Level BCE Compatibility

Implemented and tested on 2026-04-27 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-candidate-compat-threshold-grid`
- `--candidate-compat-hidden-dim`
- `--candidate-compat-epochs`
- `--candidate-compat-lr`

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat/experiment_summary.json`

Design:

- train a separate candidate-level head over `(shot_features, candidate_features)`
- supervise only nonzero candidates
- positive target is `target_score > 0`, i.e. a nonzero candidate that actually
  fixes the PyMatching logical class under the benefit/harm objective
- use the head as a threshold gate before final selector choice

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |

Compatibility diagnostic:

| validation family | true positive fraction | predicted positive fraction |
| --- | ---: | ---: |
| stage_a_si1000 | 0.014706 | 0.230328 |
| stage_b_local | 0.011268 | 0.227464 |

Forced held-out `stage_c_corr` check:

- thresholds `0.1` through `0.9` do not change the harmful selected edits
- margin `0`: `0.888671875 -> 0.879882812`, improved `50`, harmed `59`
- margin `1`: `0.888671875 -> 0.878906250`, improved `34`, harmed `44`
- margin `2`: no edit, no change

Interpretation:

- the candidate-level BCE head is connected but badly calibrated under extreme
  positive sparsity
- it predicts too many nonzero candidates as compatible, so thresholding does
  not remove the harmful selected edits
- the next compatibility attempt should use group-balanced or pairwise
  beneficial-vs-harmful ranking within each shot, not flat BCE over all
  nonzero candidates

## D5 Group-Balanced Candidate Compatibility

Implemented and tested on 2026-04-27 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--candidate-compat-objective group_balanced`
- `--candidate-compat-negative-ratio`
- `--candidate-compat-no-positive-negative-count`

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_groupbal/experiment_summary.json`

Design:

- train the same candidate compatibility head
- sample candidate rows by shot group instead of flat all-candidate BCE
- keep beneficial nonzero candidates and sample harmful nonzero candidates
  around them
- keep a small number of negative-only groups so the head still sees harmful
  candidates from unsolved-or-unhelpful groups

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |

Compatibility diagnostic from the checkpoint:

| validation family | true positive fraction | predicted positive fraction |
| --- | ---: | ---: |
| stage_a_si1000 | 0.016619 | 0.002292 |
| stage_b_local | 0.010626 | 0.001377 |

Forced held-out `stage_c_corr` check:

- margin `0`, threshold `0`:
  `0.888671875 -> 0.880859375`, improved `48`, harmed `56`
- margin `0`, threshold `0.5`:
  `0.888671875 -> 0.878906250`, improved `32`, harmed `42`
- margin `1`:
  no edit, no change

Interpretation:

- group-balanced BCE overcorrects the flat-BCE problem
- it becomes too conservative and still does not isolate enough beneficial
  nonzero edits to beat PyMatching
- absolute thresholding is the wrong interface for this compatibility signal
- next step should be pairwise within-shot compatibility ranking:
  beneficial nonzero candidates should outrank harmful nonzero candidates, and
  candidate selection should use that relative rank

## D5 Pairwise Candidate Compatibility Ranking

Implemented and tested on 2026-04-27 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--candidate-compat-objective pairwise_rank`
- `--selector-candidate-compat-top-k-grid`

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_pairwise/experiment_summary.json`

Design:

- train an auxiliary compatibility head with within-shot pairwise loss
- beneficial nonzero candidates should outrank harmful nonzero candidates
- use relative top-k compatibility rank to mask nonzero candidates before the
  main selector chooses an edit

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | delta | improved | harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 | 0 | 0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 | 0 | 0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |

Forced held-out `stage_c_corr` check:

- margin `0`, top-k `0`: `0.888671875 -> 0.879882812`,
  improved `52`, harmed `61`
- margin `0`, top-k `1/2/4/8`: same result as top-k `0`
- margin `2`: no edit, no change

Interpretation:

- the auxiliary pairwise compatibility rank does not alter the selected harmful
  nonzero candidates
- harmful candidates selected by the main selector remain high-ranked by the
  auxiliary head
- the next step should merge the beneficial-vs-harmful ranking term into the
  main selector itself rather than using a separate compatibility gate

## D5 Main Selector Pairwise Benefit/Harm Ranking

Implemented and tested on 2026-04-27 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-benefit-harm-pairwise-loss-weight`
- `--selector-benefit-harm-pairwise-margin`

Artifacts:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise_margin15/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_selector_pairwise_w16/experiment_summary.json`

Design:

- add a direct pairwise term to the main selector group-rank objective
- within each shot group, beneficial nonzero candidates should outrank harmful
  nonzero candidates
- this is no longer a detached auxiliary gate

Selected d5 result:

| run | selected mode | selected selector behavior | stage_c baseline | stage_c edited |
| --- | --- | --- | ---: | ---: |
| pairwise weight `1` | global_policy | no edit | 0.888671875 | 0.888671875 |
| pairwise weight `1`, margin-grid includes `1.5` | global_policy | no edit | 0.888671875 | 0.888671875 |
| pairwise weight `16` | global_policy | no edit | 0.888671875 | 0.888671875 |

Forced sweep observation:

- one weight-`1` checkpoint had a narrow positive full-eval band at selector
  margin `1.5`:
  - `stage_a_si1000`: `0.900390625 -> 0.901367188`
  - `stage_b_local`: `0.904296875 -> 0.907226562`
  - `stage_c_corr`: `0.888671875 -> 0.889648438`
- this did not become selected behavior when margin `1.5` was added to the
  validation grid
- weight `16` still harms when edits are forced and reverts to no-edit at the
  selected margins

Interpretation:

- the direct selector pairwise term is wired correctly and active in training
- however, selected d5 performance remains raw PyMatching
- selector-only calibration has not stably recovered d5 oracle headroom
- the next model change should alter the candidate set or candidate
  representation, not add another selector-side scalar/ranking loss

## D5 Motif Evidence Merge

Implemented and tested on 2026-04-28 in
`decoders/syndrome_edit_predecoder.py`.

Design:

- when a policy-generated candidate duplicates a motif/local-motif candidate
  with the same edit mask, merge motif evidence into the existing candidate
  feature row
- this lets the selector know that a high-confidence policy edit also matches
  an oracle-derived motif pattern

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifmerge_pairwise/experiment_summary.json`

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | delta |
| --- | --- | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | +0.000000000 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | +0.000000000 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | +0.000000000 |

Forced low-margin result:

| family | edited accuracy | delta | improved | harmed |
| --- | ---: | ---: | ---: | ---: |
| stage_a_si1000 | 0.901367188 | +0.000976562 | 44 | 43 |
| stage_b_local | 0.903320312 | -0.000976562 | 48 | 49 |
| stage_c_corr | 0.881835938 | -0.006835938 | 52 | 59 |

Interpretation:

- candidate representation improved slightly on one seen family but still harms
  held-out `stage_c_corr`
- the d5 failure is not just missing motif provenance on duplicate candidates
- next candidate-set test should restrict raw policy candidates and rely on
  motif-derived candidates, or build transition-conditioned motif candidates

## D5 Motif-Only Candidate Pool

Implemented and tested on 2026-04-28 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-policy-candidate-mode {all,none}`
- mode `none` disables raw threshold/top-k policy candidates while keeping
  identity and motif/local-motif candidates

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifonly_pairwise/experiment_summary.json`

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | candidate oracle | mean candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | 0.999023438 | 33.0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | 0.996093750 | 33.0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | 0.999023438 | 33.0 |

Interpretation:

- disabling raw policy candidates removes one harmful emission path, but it
  still does not make the selector choose motif/local-motif edits
- candidate availability is not the blocker because the motif-only oracle is
  still near-saturated
- the next likely bottleneck is candidate representation: the selector needs
  explicit detector placement / motif-identity information, not just
  probability summaries and motif count

## D5 Geometry/Placement Candidate Features

Implemented and tested on 2026-04-28 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-candidate-geometry-features`
- candidate feature rows append normalized mean/std/span summaries of selected
  detector `(time,row,col)` coordinates
- transition-feature offsets are inferred from candidate feature width, so
  benefit/harm transition features still work when geometry features are
  enabled

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_geom_motifonly_pairwise/experiment_summary.json`

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | candidate oracle | mean candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | 0.999023438 | 33.0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | 0.996093750 | 33.0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | 0.999023438 | 33.0 |

Interpretation:

- coordinate placement summaries are not enough to make the safe selector emit
  useful d5 edits
- the candidate pool still has enough oracle headroom, so the failure remains
  representation/selection rather than availability
- the next useful feature should encode local motif pattern identity and
  anchor-pattern structure directly

## D5 Local Motif Pattern/Anchor Candidate Features

Implemented and tested on 2026-04-28 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-candidate-pattern-features`
- candidate feature rows append local-pattern-present, normalized pattern id,
  log pattern count, and normalized anchor `(time,row,col)`

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patterngeom_motifonly_pairwise/experiment_summary.json`

Selected d5 result:

| family | candidate-selector behavior | baseline PyMatching | edited PyMatching | candidate oracle | mean candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| stage_a_si1000 | no edit | 0.900390625 | 0.900390625 | 0.999023438 | 33.0 |
| stage_b_local | no edit | 0.904296875 | 0.904296875 | 0.996093750 | 33.0 |
| stage_c_corr | no edit | 0.888671875 | 0.888671875 | 0.999023438 | 33.0 |

Interpretation:

- pattern/anchor metadata alone still does not unlock selected d5 gains
- the best selector epoch emits some nonzero candidates on validation, but
  their mean selected target score is negative, so the margin guardrail
  correctly suppresses final emission
- the next likely missing information is anchor-local syndrome/evidence:
  selected detector events and small-neighborhood event/probability summaries

## D5 Anchor-Local Evidence Candidate Features

Implemented and tested on 2026-04-29 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-candidate-local-evidence-features`
- candidate feature rows append selected-detector event/probability summaries
  and radius-1 anchor-neighborhood event/probability summaries

Artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_localevidence_patterngeom_motifonly_pairwise/experiment_summary.json`

Selected d5 result:

| family | selected behavior | baseline PyMatching | edited PyMatching |
| --- | --- | ---: | ---: |
| stage_a_si1000 | global no-edit | 0.900390625 | 0.900390625 |
| stage_b_local | global no-edit | 0.904296875 | 0.904296875 |
| stage_c_corr | global no-edit | 0.888671875 | 0.888671875 |

Candidate-selector branch:

| family | baseline PyMatching | edited PyMatching | improved | harmed | oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage_a_si1000 | 0.900390625 | 0.900390625 | 1 | 1 | 0.999023438 |
| stage_b_local | 0.904296875 | 0.905273438 | 1 | 0 | 0.996093750 |
| stage_c_corr | 0.888671875 | 0.887695312 | 0 | 1 | 0.999023438 |

Interpretation:

- handcrafted local evidence is the first feature-only ablation in this group
  that induces sparse nonzero emission
- that emission is still not selected because held-out `stage_c_corr` is
  harmed by one shot
- the next useful model change should move from scalar feature appends to a
  learned candidate-conditioned local patch scorer

## D5 Local-Patch Candidate Features

Implemented and tested on 2026-04-30 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-candidate-local-patch-features`
- candidate feature rows append a radius-1 `3x3x3` anchor patch
- each patch cell stores detector event and pre-decoder edit probability
- missing detector cells are zero-filled

Artifacts:

- `artifacts/eval/nn/sedp_d5_smoke_localpatch/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_modest_localpatch/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_localpatch_patterngeom_motifonly_pairwise/experiment_summary.json`

Full d5 router1k selected result:

| family | selected behavior | baseline PyMatching | edited PyMatching |
| --- | --- | ---: | ---: |
| ideal | global no-edit | 1.000000000 | 1.000000000 |
| stage_a_si1000 | global no-edit | 0.900390625 | 0.900390625 |
| stage_b_local | global no-edit | 0.904296875 | 0.904296875 |
| stage_c_corr | global no-edit | 0.888671875 | 0.888671875 |

Full candidate-selector branch:

| family | baseline PyMatching | edited PyMatching | improved | harmed | oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ideal | 1.000000000 | 1.000000000 | 0 | 0 | 1.000000000 |
| stage_a_si1000 | 0.900390625 | 0.900390625 | 0 | 0 | 0.999023438 |
| stage_b_local | 0.904296875 | 0.904296875 | 0 | 0 | 0.996093750 |
| stage_c_corr | 0.888671875 | 0.888671875 | 0 | 0 | 0.999023438 |

Interpretation:

- preserving local patch layout as appended candidate features is not enough
  to unlock selected d5 gains
- the validation guardrail chooses selector margin `2.0`, suppressing all
  nonzero edits in the full run
- this remains compatible with the broader interpretation: the candidate pool
  has high oracle headroom, but the current selector representation/objective
  still cannot reliably choose beneficial nonzero edits at d5

## D5 Patch-Head Selector

Implemented and tested on 2026-05-01 in
`decoders/syndrome_edit_predecoder.py`.

New support:

- `--selector-patch-head`
- `--selector-patch-hidden-dim`
- the selector now has an optional learned patch branch that encodes the
  local-patch feature slice separately before scoring candidates

Smoke artifact:

- `artifacts/eval/nn/sedp_d5_smoke_patchhead_v3/experiment_summary.json`

Smoke result:

- selected mode remains `global_policy`
- no selector edits were emitted on the tiny split
- the summary records `selector_patch_head=True`

Modest d5 artifact:

- `artifacts/eval/nn/sedp_d5_modest_patchhead/experiment_summary.json`

Modest selected result:

| family | baseline PyMatching | edited PyMatching | improved | harmed | oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ideal | 1.000000000 | 1.000000000 | 0 | 0 | 1.000000000 |
| stage_a_si1000 | 0.902343750 | 0.914062500 | 8 | 2 | 0.998046875 |
| stage_b_local | 0.912109375 | 0.933593750 | 11 | 0 | 0.996093750 |
| stage_c_corr | 0.892578125 | 0.898437500 | 11 | 8 | 0.998046875 |

Full d5 router1k artifact:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_patterngeom_motifonly_pairwise/experiment_summary.json`

Full selected result:

| family | selected behavior | baseline PyMatching | edited PyMatching | oracle |
| --- | --- | ---: | ---: | ---: |
| ideal | global no-edit | 1.000000000 | 1.000000000 | 1.000000000 |
| stage_a_si1000 | global no-edit | 0.900390625 | 0.900390625 | 0.999023438 |
| stage_b_local | global no-edit | 0.904296875 | 0.904296875 | 0.996093750 |
| stage_c_corr | global no-edit | 0.888671875 | 0.888671875 | 0.999023438 |

Interpretation:

- this is the first d5 patch-head selected win at modest scale
- it does not survive the full router1k validation guardrail, which selects
  margin `4.0` and suppresses edits
- the next move should target selector objective/calibration for rare
  beneficial nonzero edits

## D5 Patch-Head Seed Sweep

Completed on 2026-05-01.

Artifacts:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_patterngeom_motifonly_pairwise/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed1/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed2/experiment_summary.json`
- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed3/experiment_summary.json`

Selected-policy held-out `stage_c_corr` comparison:

| seed | selected mode | PyMatching | selected decoder | delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | global_policy | 0.888671875 | 0.888671875 | +0.000000000 |
| 1 | local_motif_selector | 0.888671875 | 0.895507813 | +0.006835938 |
| 2 | global_policy | 0.888671875 | 0.888671875 | +0.000000000 |
| 3 | local_motif_selector | 0.888671875 | 0.896484375 | +0.007812500 |

Candidate-branch held-out `stage_c_corr` comparison:

| seed | PyMatching | candidate branch | delta | improved | harmed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.888671875 | 0.888671875 | +0.000000000 | 0 | 0 |
| 1 | 0.888671875 | 0.895507813 | +0.006835938 | 26 | 19 |
| 2 | 0.888671875 | 0.893554688 | +0.004882813 | 5 | 0 |
| 3 | 0.888671875 | 0.896484375 | +0.007812500 | 15 | 7 |

Interpretation:

- this is the first d5 full-router1k path with selected PyMatching-beating
  seeds
- seed `1` and seed `3` select the neural pre-decoder branch and beat raw
  PyMatching on held-out `stage_c_corr`
- seed `2` does not pass selected-mode guardrail, but its candidate branch
  still beats PyMatching with no harmed held-out shots
- the remaining bottleneck is selected-mode stability, not candidate generation

## Active Patch-Head Non-Inferiority Distance Ladder

Status as of 2026-05-02: completed.

The active recipe is patch-head local selector with
`--selector-adoption-min-delta 0.0`.

Held-out `stage_c_corr`, mean over seeds `0..3`:

| distance | adopted seeds | PyMatching | selected decoder | delta | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | 4/4 | 0.928710938 | 0.938232422 | +0.009521484 | 1.000000000 |
| d5 | 4/4 | 0.888671875 | 0.898925781 | +0.010253906 | 0.999023438 |
| d7 no guard | 2/4 | 0.873046875 | 0.868652344 | -0.004394531 | 0.961181641 |
| d7 no-edit guard 0.005 | 1/4 selector, 3/4 raw | 0.873046875 | 0.875244141 | +0.002197266 | 0.988037109 |

Interpretation:

- d3 and d5 now have seed-stable selected gains under the same active recipe
- d7 selected no-edit guardrail prevents harmful global fallback and makes the
  mean held-out result slightly positive
- d7 remains mostly no-edit, so the unsolved problem is recovering nonzero
  local-selector gains under the guardrail

## D7 Guarded Selection Calibration

Completed on 2026-05-02.

New comparison support:

- `tools/compare_predecoder_seed_sweep.py`
- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_noeditguard_margin005_selection_compare.json`
- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_noeditguard_margin005_selection_compare_seed0_7.json`

d7 no-edit guardrail with margin `0.005`, held-out `stage_c_corr`:

| seed | selected mode | val selected delta over no-edit | held-out selected delta | candidate delta | improved | harmed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | raw_no_edit | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |
| 1 | raw_no_edit | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |
| 2 | raw_no_edit | +0.000000000 | +0.000000000 | -0.008789062 | 0 | 0 |
| 3 | local_motif_selector | +0.006505271 | +0.008789062 | +0.008789062 | 13 | 4 |

Selection criterion comparison:

| criterion | chosen seed | mode | held-out selected delta |
| --- | ---: | --- | ---: |
| absolute selected validation metric | 0 | raw_no_edit | +0.000000000 |
| selected validation delta over no-edit | 3 | local_motif_selector | +0.008789062 |
| candidate validation delta over no-edit | 3 | local_motif_selector | +0.008789062 |

Interpretation:

- the margin `0.005` guardrail is still useful: the lower-margin v2 sweep
  selected seed `0`, whose held-out `stage_c_corr` delta was `-0.004882812`
- score-penalty-zero pilots on seeds `0` and `3` are negative; both select
  `raw_no_edit` and give zero selected/candidate gain
- d7 model selection should use validation improvement over no-edit instead
  of absolute validation accuracy when comparing guarded runs

Extended seed check, seeds `4..7`, was run with
`--selector-local-motif-max-classes 16` to match the active local-motif
candidate recipe. Sanity local-pattern counts are `10, 8, 10, 9`.

8-seed result:

| seed | selected mode | local patterns | val candidate delta over no-edit | held-out selected delta | candidate delta | improved | harmed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | raw_no_edit | 9 | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |
| 1 | raw_no_edit | 9 | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |
| 2 | raw_no_edit | 9 | +0.000001511 | +0.000000000 | -0.008789062 | 0 | 0 |
| 3 | local_motif_selector | 10 | +0.006505271 | +0.008789062 | +0.008789062 | 13 | 4 |
| 4 | raw_no_edit | 10 | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |
| 5 | raw_no_edit | 8 | +0.003249608 | +0.000000000 | -0.009765625 | 0 | 0 |
| 6 | raw_no_edit | 10 | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |
| 7 | raw_no_edit | 9 | +0.000000000 | +0.000000000 | +0.000000000 | 0 | 0 |

8-seed mean selected held-out `stage_c_corr` delta is `+0.001098633`.
The current recipe is safe but not yet a stable d7 solution: it recovers one
nonzero local-selector seed and correctly blocks at least one sub-margin
harmful candidate seed (`5`).

Seed `3` / seed `5` diagnostic:

- tool: `tools/diagnose_predecoder_selection.py`
- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed3_vs_seed5_stagec_selection_diagnostic.json`

Held-out `stage_c_corr` per-shot selection:

| seed | selector margin | selected nonzero | positive target score | negative target score | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.25 | 17 | 13 | 4 | +0.008789062 |
| 5 | 0.00 | 68 | 29 | 39 | -0.009765625 |

Logit-gap interpretation:

- seed `3` selected nonzero gaps are all above `1.25`
  (`min 1.266623`, median `1.450292`)
- seed `5` best-nonzero gap max is only `1.209157`; a seed-3-like margin
  would have suppressed all seed-5 nonzero edits
- seed `5` over-edits already-correct non-I shots, especially
  `Z->I|target=Z` (`26` harms) and `Y->X|target=Y` (`9` harms)
- the next calibration test should therefore focus on selector emit-margin
  floor / margin-scale stability, not on adding another candidate feature

Post-hoc margin floor check:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_7_stagec_margin125_diagnostic.json`
- applying selector emit margin `1.25` to seeds `0..7` on held-out
  `stage_c_corr` preserves seed `3` and suppresses seed `5`
- only seed `3` emits nonzero edits at this floor; all other seeds are no-edit
- the 8-seed mean remains `+0.001098633`, so this floor is useful for safety
  but does not by itself solve d7 nonzero recovery

Margin-sweep calibration:

- artifacts:
  - `artifacts/eval/nn/sedp_d7_seed3_stagea_val_margin_sweep.json`
  - `artifacts/eval/nn/sedp_d7_seed3_stageb_val_margin_sweep.json`
  - `artifacts/eval/nn/sedp_d7_seed5_stagea_val_margin_sweep.json`
  - `artifacts/eval/nn/sedp_d7_seed5_stageb_val_margin_sweep.json`
  - `artifacts/eval/nn/sedp_d7_seed3_seed5_stagec_margin_sweep.json`
  - `artifacts/eval/nn/sedp_d7_seed3_seed5_margin_sweep_summary.json`

Validation split detail:

- manifest experiments split each train family with `seed + offset`
- for the current two train families:
  `stage_a_si1000` uses `seed`, `stage_b_local` uses `seed + 1`

Seed `3` validation / held-out by margin:

| margin | validation mean delta | held-out `stage_c_corr` delta |
| ---: | ---: | ---: |
| 0.00 | -0.064935065 | -0.037109375 |
| 0.50 | -0.025974026 | -0.000976562 |
| 1.00 | -0.006493506 | +0.007812500 |
| 1.25 | +0.006493506 | +0.008789062 |
| 1.50 | +0.000000000 | +0.003906250 |
| 1.75 | -0.000000000 | +0.000976562 |
| 2.00 | -0.000000000 | +0.000000000 |
| 4.00 | +0.000000000 | +0.000000000 |

Seed `5` validation / held-out by margin:

| margin | validation mean delta | held-out `stage_c_corr` delta |
| ---: | ---: | ---: |
| 0.00 | +0.003246753 | -0.009765625 |
| 0.50 | +0.000000000 | -0.003906250 |
| 1.00 | +0.000000000 | +0.000000000 |
| 1.25 | +0.000000000 | +0.000000000 |
| 1.50 | +0.000000000 | +0.000000000 |
| 1.75 | +0.000000000 | +0.000000000 |
| 2.00 | +0.000000000 | +0.000000000 |
| 4.00 | +0.000000000 | +0.000000000 |

Conclusion:

- `--selected-no-edit-min-delta 0.005` is justified by the margin sweep:
  seed `3` clears it at margin `1.25`, while seed `5` does not clear it at
  any margin
- lowering the guard below about `0.00325` would adopt seed `5` even though its
  held-out `stage_c_corr` candidate branch is harmful
- this validates the current guard margin as a selection-calibration rule, but
  it still leaves d7 nonzero recovery at `1/8` seeds

8-seed fixed-margin profile:

- artifact:
  `artifacts/eval/nn/sedp_d7_seed0_7_margin125_validation_heldout_profile.json`
- fixed selector emit margin: `1.25`
- validation split rule:
  `stage_a_si1000` uses `seed`, `stage_b_local` uses `seed + 1`

| seed | validation mean delta | validation nonzero | validation improved/harmed | held-out delta | held-out nonzero | held-out improved/harmed | held-out max gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | +0.167316 |
| 1 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | -0.316389 |
| 2 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | +0.807335 |
| 3 | +0.006493506 | 6 | 4/2 | +0.008789062 | 17 | 13/4 | +1.923424 |
| 4 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | +0.649664 |
| 5 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | +1.209157 |
| 6 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | -1.136392 |
| 7 | +0.000000000 | 0 | 0/0 | +0.000000000 | 0 | 0/0 | +0.244566 |

Interpretation:

- seed `3` is the only seed with a validation-qualified high-gap cluster above
  margin `1.25`
- seed `5` is close but sub-margin; its held-out max gap is `1.209157`, so the
  `1.25` floor suppresses its harmful broad low-margin edits
- absolute selector validation metric is misleading here: seed `3` has lower
  selector validation metric than several no-edit seeds, but it is the only
  seed with margin-qualified positive edits

## Seed-Control Diagnostic

Follow-up on 2026-05-03 found that predecoder training used `seed` for data
splits but did not fix model initialization, torch sampler order, or selector
group shuffling. `syndrome_edit_predecoder.py` now seeds numpy/torch at the
training entry points and adds optional selector epoch diagnostics:

- `--selector-epoch-diagnostic-margin-grid`
- epoch records include validation delta over no-edit, selected nonzero count,
  improved/harmed counts, target-score sign counts, and best-nonzero gap
  quantiles by margin

Seed-fixed d7 pilot artifacts:

- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed3_seedfixed_epochdiag/experiment_summary.json`
- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed5_seedfixed_epochdiag/experiment_summary.json`
- `artifacts/eval/nn/sedp_d7_seed3_seed5_seedfixed_epochdiag_selection_compare.json`
- `artifacts/eval/nn/sedp_d7_seed3_seed5_seedfixed_epochdiag_margin125_epoch_summary.json`
- `artifacts/eval/nn/sedp_d7_seed3_seed5_seedfixed_stagec_margin_diagnostic.json`

Seed-fixed held-out `stage_c_corr`:

| seed | selected mode | selected delta | candidate selected-margin delta | post-hoc margin 1.25 delta | note |
| ---: | --- | ---: | ---: | ---: | --- |
| 3 | raw_no_edit | +0.000000000 | -0.001953125 | +0.000976562 | one positive held-out edit at margin 1.25, not validation-qualified |
| 5 | raw_no_edit | +0.000000000 | +0.000000000 | +0.000000000 | no held-out edits at margin 1.0 or 1.25 |

Interpretation:

- the old seed `3` result remains evidence that a high-gap local cluster can
  help d7
- it is not yet deterministic seed-stability evidence because the earlier
  sweep was not fully RNG-controlled
- future d7 stability claims should use the seed-fixed code path

## Seed-Fixed D7 Sweep

The full deterministic seed-fixed `0..7` d7 sweep is now complete.

Artifacts:

- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_seedfixed_epochdiag/experiment_summary.json`
  through seed `7`
- `artifacts/eval/nn/sedp_d7_seedfixed_epochdiag_selection_compare_seed0_7.json`
- `artifacts/eval/nn/sedp_d7_seedfixed_epochdiag_margin125_epoch_summary_seed0_7.json`
- `artifacts/eval/nn/sedp_d7_seedfixed_margin_floor_policy_summary_seed0_7.json`
- `artifacts/eval/nn/sedp_d7_seed2_seedfixed_stagec_margin_diagnostic.json`

Held-out `stage_c_corr` selected-mode summary:

| seed | selected mode | validation selected delta | held-out selected delta | candidate improved/harmed |
| ---: | --- | ---: | ---: | ---: |
| 0 | raw_no_edit | +0.000000000 | +0.000000000 | 2/2 |
| 1 | raw_no_edit | +0.000000000 | +0.000000000 | 0/0 |
| 2 | local_motif_selector | +0.012998091 | -0.004882812 | 17/22 |
| 3 | raw_no_edit | +0.000000000 | +0.000000000 | 3/5 |
| 4 | raw_no_edit | +0.000000000 | +0.000000000 | 0/0 |
| 5 | raw_no_edit | +0.000000000 | +0.000000000 | 0/0 |
| 6 | raw_no_edit | +0.000000000 | +0.000000000 | 0/0 |
| 7 | raw_no_edit | +0.000000000 | +0.000000000 | 0/0 |

Aggregate:

- selected mode count: `7/8` raw no-edit, `1/8` local selector
- mean selected held-out delta: `-0.000610352`
- mean candidate held-out delta: `-0.000854492`

Seed `2` is the new false-positive adoption case. It clears the current
validation guard by a large margin, but its held-out gain is negative. The
margin diagnostic shows this is low-margin over-editing:

| margin | held-out delta | nonzero edits | improved/harmed |
| ---: | ---: | ---: | ---: |
| 0.0 | -0.004882812 | 39 | 17/22 |
| 1.0 | +0.000000000 | 0 | 0/0 |
| 1.25 | +0.000000000 | 0 | 0/0 |

At selector-best epochs, forcing margin floor `1.0` or `1.25` prevents every
seed from clearing the `0.005` no-edit guard. That policy would select raw
no-edit for all seed-fixed d7 runs, avoid seed `2` harm, and produce zero
learned d7 gain. This is safer than the current automatic low-margin adoption
but still does not recover d7 oracle headroom.

The seed `2` margin-floor recipe was also checked as an actual rerun rather
than only a post-hoc diagnostic:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_seedfixed_marginfloor10/experiment_summary.json`
- comparison artifact:
  `artifacts/eval/nn/sedp_d7_seed2_seedfixed_marginfloor10_compare.json`
- recipe change: selector emit-margin grid restricted to
  `1.0 1.25 1.5 1.75 2.0 4.0`
- selected mode: `raw_no_edit`
- held-out `stage_c_corr` selected delta: `+0.000000000`

This validates the d7 margin floor as a safety policy: it blocks the seed `2`
false-positive adoption. It should not be presented as learned d7 improvement.

## Seed-Fixed D5 Revalidation

After the RNG-control fix, d5 seeds `0..3` were rerun with the active
patch-head local selector recipe.

Artifacts:

- `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_seedfixed_epochdiag/experiment_summary.json`
  through seed `3`
- `artifacts/eval/nn/sedp_d5_seedfixed_epochdiag_selection_compare_seed0_3.json`
- `artifacts/eval/nn/sedp_d5_seedfixed_epochdiag_selection_rows_seed0_3.json`
- `artifacts/eval/nn/sedp_d5_seedfixed_posthoc_noeditguard_margin005_summary_seed0_3.json`

Held-out `stage_c_corr`:

| seed | selected mode | selected margin | selected delta | candidate delta | candidate improved/harmed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | local_motif_selector | 4.0 | +0.000000000 | +0.000000000 | 0/0 |
| 1 | global_policy | 0.5 | -0.018554688 | +0.000000000 | 0/0 |
| 2 | global_policy | 1.5 | +0.000000000 | +0.021484375 | 28/6 |
| 3 | local_motif_selector | 0.5 | +0.023437500 | +0.023437500 | 28/4 |

Aggregate:

- original selected-mode mean delta: `+0.001220703`
- candidate-branch mean delta: `+0.011230469`
- post-hoc selected no-edit guard with margin `0.005` mean delta:
  `+0.005859375`

Interpretation:

- the previous d5 mean `+0.010253906` claim does not survive unchanged under
  fully seed-controlled training
- d5 is still more promising than d7 because the candidate branch remains
  strongly positive on seeds `2` and `3`
- the selected-mode policy is now the bottleneck: seed `1` is a harmful
  global-policy false positive, while seed `2` leaves a positive candidate
  branch unadopted
- no-edit guarding is useful for d5 as well, not only d7

## Seed-Fixed D3 Revalidation And Current Ladder

d3 seeds `0..3` were rerun under the seed-fixed code path.

Artifacts:

- `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed0_seedfixed_epochdiag/experiment_summary.json`
  through seed `3`
- `artifacts/eval/nn/sedp_d3_seedfixed_epochdiag_selection_compare_seed0_3.json`
- `artifacts/eval/nn/sedp_d3_seedfixed_epochdiag_selection_rows_seed0_3.json`
- `artifacts/eval/nn/sedp_d3_seedfixed_posthoc_noeditguard_margin005_summary_seed0_3.json`
- `artifacts/eval/nn/sedp_seedfixed_distance_ladder_summary.json`

d3 held-out `stage_c_corr`:

| seed | selected mode | selected margin | selected delta | improved/harmed |
| ---: | --- | ---: | ---: | ---: |
| 0 | local_motif_selector | 0.0 | +0.010742188 | 31/20 |
| 1 | local_motif_selector | 1.5 | +0.006835938 | 20/13 |
| 2 | local_motif_selector | 0.0 | +0.008789062 | 29/20 |
| 3 | local_motif_selector | 0.0 | +0.003906250 | 24/20 |

d3 mean selected/candidate delta is `+0.007568359`. The no-edit guard margin
`0.005` would not change d3 adoption because all seeds clear validation by more
than the guard.

Current seed-fixed ladder:

| distance | policy | held-out `stage_c_corr` mean delta | interpretation |
| --- | --- | ---: | --- |
| d3 | original selected | +0.007568359 | stable positive |
| d3 | integrated candidate-first policy | +0.007568359 | canonical regression passed |
| d5 | original selected | +0.001220703 | positive but weak; selected-mode issue |
| d5 | post-hoc no-edit guard `0.005` | +0.005859375 | safer, still below candidate branch |
| d5 | candidate branch | +0.011230469 | learned signal exists |
| d5 | integrated candidate-first policy | +0.011230469 | real selected-mode recovery |
| d7 | original selected | -0.000610352 | unsafe false positive |
| d7 | margin-floor/no-edit safe policy | +0.000000000 | safe, no learned gain |
| d7 | integrated candidate-first policy | +0.000000000 | safe, no learned gain |

## Candidate-First Adoption Simulation

The next selected-mode calibration candidate is a candidate-first safety rule.
It was evaluated post-hoc with:

- tool: `tools/simulate_predecoder_adoption_policy.py`
- artifact:
  `artifacts/eval/nn/sedp_seedfixed_candidate_first_adoption_policy_sim.json`

Policy sketch:

- do not adopt the global edit policy by default
- adopt the candidate selector if validation delta over no-edit is strong:
  `>=0.02`
- or adopt the candidate selector if validation delta clears `0.005` and the
  selected margin is at least `0.5`
- or adopt a validation-tied candidate if selected margin is at least `1.0`
  and validation selected nonzero count is at least `6`
- otherwise select raw no-edit

Seed-fixed simulation result:

| distance | policy mean delta | selected policy behavior |
| --- | ---: | --- |
| d3 | +0.007568359 | all four seeds stay local selector |
| d5 | +0.011230469 | seed `2` candidate branch is recovered; seed `1` is blocked |
| d7 | +0.000000000 | all seeds raw no-edit; seed `2` false positive is blocked |

This is currently the best adoption-policy candidate because it preserves d3,
recovers the d5 candidate signal, and avoids d7 low-margin false positives.
It was later integrated and became the active selected-mode policy checked in
the following section.

## Candidate-First Integrated Policy Check

The candidate-first safety rule is now implemented in
`decoders/syndrome_edit_predecoder.py` as
`--selector-adoption-policy candidate_first_safety`. Default behavior is still
`global_noninferiority`.

Actual d5 seed `0..3` integrated-policy sweep:

- artifacts:
  `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise/experiment_summary.json`
  through seed `3`
- comparison:
  `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_3.json`

| seed | selected mode | reason | held-out `stage_c_corr` delta |
| ---: | --- | --- | ---: |
| 0 | raw_no_edit | default_no_edit | +0.000000000 |
| 1 | raw_no_edit | default_no_edit | +0.000000000 |
| 2 | local_motif_selector | candidate_tie_with_high_margin_evidence | +0.021484375 |
| 3 | local_motif_selector | candidate_positive_delta_with_margin | +0.023437500 |

Mean selected delta is `+0.011230469`, matching the simulated
candidate-first policy and the candidate-branch mean.

d7 seed `2` integrated safety smoke:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_smoke_pairwise/experiment_summary.json`
- selected mode: `raw_no_edit`
- selected delta: `+0.000000000`
- candidate branch delta: `-0.004882812`

The integrated policy therefore recovers the d5 selected-mode gain and blocks
the known d7 low-margin false positive. The full canonical d7 seed `0..7`
integrated policy sweep is now complete:

- artifacts:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise_seq/experiment_summary.json`
  through seed `7`
- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json`
- all seeds select `raw_no_edit`
- mean selected held-out `stage_c_corr` delta is `+0.000000000`
- mean candidate-branch held-out `stage_c_corr` delta is `-0.000854492`

This confirms the policy is d7-safe in the integrated path, but it does not
recover a learned d7 gain.

Canonical d3 integrated-policy regression:

- artifacts:
  `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise_seq/experiment_summary.json`
  through seed `3`
- comparison:
  `artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_3.json`
- all four seeds select `local_motif_selector`
- mean selected held-out `stage_c_corr` delta is `+0.007568359`
- distance ladder artifact:
  `artifacts/eval/nn/sedp_candidatefirst_distance_ladder_summary.json`

## Next Work

The next useful implementation step is:

1. keep patch-head + non-inferiority adoption as the current leading learned
   pre-decoder path for d3/d5
2. keep the transition-prior, hard top-k compatibility, flat BCE compatibility,
   group-balanced BCE compatibility, auxiliary pairwise compatibility, and main
   selector pairwise variants as ablations
3. keep motif-evidence merge and motif-only candidate-pool restriction as
   completed d5 ablations
4. keep geometry/placement-aware candidate features as a completed d5
   representation ablation
5. keep local motif pattern-id / anchor-pattern-aware candidate features as a
   completed d5 representation ablation
6. keep anchor-local syndrome/evidence candidate features as a completed d5
   representation ablation
7. keep appended local-patch candidate features as a completed d5 ablation
8. keep selected no-edit guardrail active for d7
9. use validation improvement over no-edit as a diagnostic, but not as a
   sufficient guarded d7 adoption criterion under seed-controlled training
10. keep `--selected-no-edit-min-delta 0.005` as the current d7 guard margin
    unless a future sweep shows a stronger validation/held-out alignment
11. treat low-margin d7 adoption as unsafe; test/keep a d7 selector margin
    floor of at least `1.0` before adoption, then use epoch diagnostics to look
    for genuinely high-gap positive clusters
12. treat guarded d7 nonzero local-selector recovery as the current bottleneck
13. move from adoption-policy validation to d7 recovery analysis; d3/d5/d7
   canonical integrated candidate-first sweeps are complete
14. require selected-mode improvement over raw PyMatching on held-out
   `stage_c_corr`, not just d3/d5 gains or candidate-oracle headroom

## D7 Recovery Epoch Diagnostics

New tool:

- `tools/summarize_selector_epoch_diagnostics.py`

Artifacts:

- `artifacts/eval/nn/sedp_d3_candidatefirst_seq_epoch_diagnostic_summary_seed0_3.json`
- `artifacts/eval/nn/sedp_d5_candidatefirst_epoch_diagnostic_summary_seed0_3.json`
- `artifacts/eval/nn/sedp_d7_candidatefirst_seq_epoch_diagnostic_summary_seed0_7.json`
- `artifacts/eval/nn/sedp_d7_recovery_epoch_diagnostic_comparison.json`

Epoch/margin diagnostic counts:

| distance | positive nonzero rows | margin>=1 positive rows | candidate-first strong rows | positive-margin rows | high-margin tie rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | 66 | 48 | 45 | 48 | 48 |
| d5 | 14 | 14 | 0 | 9 | 10 |
| d7 | 6 | 3 | 0 | 1 | 0 |

D7 diagnosis:

- best validation row is seed `2`, epoch `4`, margin `0.0`, with mean
  validation delta `+0.012987013`, nonzero `12`, improved/harmed `8/4`, and
  max gap only `0.550849`
- that is low-margin evidence; the canonical seed `2` held-out candidate
  branch is harmful at `-0.004882812`
- the only d7 margin `>=1` positive rows have nonzero support `1-2`, so they
  are too sparse for selected-mode adoption
- therefore the current d7 blocker is not that candidate-first thresholds are
  too strict; the current training rarely creates robust high-margin positive
  clusters at d7

## D7 Diagnostic Epoch Selection Recovery Check

Implemented on 2026-05-04:

- optional selector epoch-selection mode:
  `--selector-epoch-selection-mode diagnostic_system`
- default remains `proxy`
- comparison tooling now records selector epoch, adoption reason, adoption
  margin, and validation nonzero support

The recovery check combines:

- identity-margin loss weight `0.5`
- pairwise benefit/harm ranking weight `1.0`
- selector epoch diagnostic grid `0.0 1.0 1.25`
- unchanged `candidate_first_safety` adoption

Artifacts:

- runs:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_idmargin05_diagselect_pairwise_seq/experiment_summary.json`
  through seed `7`
- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_7.json`
- epoch diagnostics:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_epoch_diagnostic_summary_seed0_7.json`

D7 held-out `stage_c_corr` result:

| recipe | selected modes | mean selected delta | mean candidate delta |
| --- | --- | ---: | ---: |
| canonical candidate-first | 8/8 raw no-edit | +0.000000000 | -0.000854492 |
| identity-margin `0.5` + diagnostic epoch selection | 2/8 local selector, 6/8 raw no-edit | +0.000854492 | +0.000488281 |

Adopted seeds:

| seed | selector epoch | adoption reason | margin | validation nonzero | held-out selected delta |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 2 | candidate_positive_delta_with_margin | 1.25 | 2 | +0.001953125 |
| 2 | 6 | candidate_positive_delta_with_margin | 1.25 | 5 | +0.004882812 |

Interpretation:

- this is the first seed-controlled d7 selected-mode learned gain under the
  current safety policy
- it is not yet strong enough to call d7 solved; only `2/8` seeds adopt the
  selector and diagnostic positive evidence remains sparse
- the safety policy remains justified because seed `5` has slightly positive
  validation candidate signal but a harmful held-out candidate branch
  (`-0.002929688`), and selected mode remains raw no-edit

## D7 Identity-Margin Weight Sentinel Ablation

The next conservative check compared identity-margin loss weights while keeping
diagnostic epoch selection and `candidate_first_safety` fixed.

Sentinel seeds:

- seed `0`: positive under weight `0.5`
- seed `2`: strongest positive under weight `0.5`
- seed `5`: safety-sensitive held-out harmful candidate branch

Artifacts:

- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin025_diagselect_selection_compare_seed0_2_5.json`
- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_2_5.json`
- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin10_diagselect_selection_compare_seed0_2_5.json`

Held-out `stage_c_corr` over seeds `0,2,5`:

| identity-margin weight | local selected | mean selected delta | mean candidate delta | conclusion |
| ---: | ---: | ---: | ---: | --- |
| 0.25 | 2/3 | +0.000651042 | -0.000325521 | unsafe; seed 0 selected delta is -0.002929688 |
| 0.5 | 2/3 | +0.002278646 | +0.001302083 | best tested compromise |
| 1.0 | 0/3 | +0.000000000 | -0.000651042 | too conservative |

Conclusion:

- keep identity-margin loss weight `0.5` as the active d7 recovery setting
- do not full-sweep weights `0.25` or `1.0` unless a later selector
  epoch-selection change materially alters the seed-sentinel behavior
- the next d7 calibration step should target epoch-selection robustness or
  selector score stability, not another scalar identity-margin sweep

## D7 Small-Volume Epoch-Selection Probe

Two low-cost checks were run before committing to more training:

- a post-hoc support-aware tie-break over existing seed `0,2,5` diagnostics
- a single seed `2` rerun with selector epoch diagnostic grid
  `0.0 1.0 1.25 1.5`

Artifact:

- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_diagselect_grid015_pairwise_seq/experiment_summary.json`

Result:

- unchanged from the existing seed `2` recovery run
- selector epoch `6`
- adoption margin `1.25`
- validation nonzero `5`
- held-out `stage_c_corr` selected delta `+0.004882812`

Conclusion:

- adding margin `1.5` to the diagnostic epoch-selection grid is not useful on
  the strongest positive d7 seed
- do not expand this probe to more seeds unless another change makes the
  diagnostic grid relevant

## D7 Selector-Epoch Count Probe

Seed `2` was rerun with `--selector-epochs 8` because its active 6-epoch run
selected the final selector epoch.

Artifact:

- `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_diagselect_epochs8_pairwise_seq/experiment_summary.json`

Result:

- selected behavior is unchanged from the default 6-epoch seed `2` run
- selector epoch `6`
- adoption margin `1.25`
- validation nonzero `5`
- held-out `stage_c_corr` selected delta `+0.004882812`

Diagnostic detail:

- epoch `7` has a validation tie at margin `1.0` with more support
  (`7` nonzero, `5/2` improved/harmed)
- epoch `8` starts to show over-edit risk at margin `1.25`
  (`-0.003246753`, `13` nonzero, `6/7` improved/harmed)

Conclusion:

- increasing selector epochs to `8` is not a useful d7 recovery move on the
  strongest positive seed
- keep selector epochs at the default `6` for now

## Next D7 Work Under Usage Limits

Current d7 baseline numbers for future comparison:

| reference | mean selected delta |
| --- | ---: |
| canonical candidate-first, safe no-edit | +0.000000000 |
| idmargin `0.5` + diagnostic epoch selection | +0.000854492 |
| idmargin `0.5` sentinel seeds `0,2,5` | +0.002278646 |
| idmargin `0.5` seed `2` only | +0.004882812 |

Next-session rule:

- do post-hoc scoring first, using existing epoch diagnostics
- do not train unless the post-hoc scoring changes the expected selected epoch
  or reveals a new promising high-margin row
- if training is justified, run one seed only before any sentinel expansion
- seed `2` is the first positive-signal seed; seed `5` is only a safety check
  after a positive result

Recommended next check:

- score existing d7 epoch rows by high-margin validation delta rather than
  mean selector accuracy alone
- reject rows with harmed >= improved
- prefer margin `>=1.0`, positive delta, positive-minus-negative target
  evidence, and moderate nonzero support
- if this rule does not identify a better epoch than the current choices, do
  not implement another selector epoch-selection mode

## D7 Seed8 False Positive And Harm-Cap Guard

When the d7 recovery recipe was extended beyond seeds `0..7`, seed `8`
exposed a harmful adoption case:

| run | selected mode | reason | val delta | val imp/harm | margin | held-out delta | held-out imp/harm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| old seed 8 | local_motif_selector | candidate_positive_delta_with_margin | +0.006481003 | 6/4 | 2.0 | -0.019531250 | 7/27 |

This means the old d7 `idmargin0.5 + diagnostic_system` recipe is not robust
enough for selected-mode claims.

Implemented guard:

- CLI:
  `--selector-candidate-first-positive-max-harmed`
- default: `1`
- applies to the positive-delta branch only
- blocks candidates where `selector_delta >= positive_delta` but validation
  harmed count exceeds the cap
- prevents the same high-positive-delta candidate from falling through to the
  tie branch

Compatibility:

- d3 selected-mode behavior is unchanged because d3 adoption is via strong
  validation delta
- d5 seed `2` remains selected via high-margin tie
- d5 seed `3` remains selected because validation harmed count is `1`
- d7 seed `0` and seed `2` remain selected
- d7 seed `8` is blocked

Actual d7 sentinel rerun:

- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap1_selection_compare_seed0_2_8.json`

| seed | selected mode | reason | held-out selected delta |
| ---: | --- | --- | ---: |
| 0 | local_motif_selector | candidate_positive_delta_with_margin | +0.001953125 |
| 2 | local_motif_selector | candidate_positive_delta_with_margin | +0.004882812 |
| 8 | raw_no_edit | candidate_positive_delta_harm_guard | +0.000000000 |

Mean selected delta over seeds `0,2,8`: `+0.002278646`.

Next d7 selection result should be reported with this harm cap enabled. The
old no-harm-cap d7 extension should not be continued.

## D7 Guarded 16-Seed Extension

After the seed `8` harm guard, seeds `9..11` were run with
`--selector-candidate-first-positive-max-harmed 1`.

Result:

- seed `9`: raw no-edit
- seed `10`: raw no-edit
- seed `11`: raw no-edit by harm guard
- seed `11` candidate branch was held-out positive (`+0.003906250`) but had
  validation harmed count `2`, so it was blocked

Seed `13` then exposed a second false-positive type:

| seed | guard state | selected mode | reason | margin | held-out delta |
| ---: | --- | --- | --- | ---: | ---: |
| 13 | harm cap only | local_motif_selector | candidate_positive_delta_with_margin | 1.75 | -0.000976562 |
| 13 | harm cap + max margin | raw_no_edit | candidate_positive_delta_margin_guard | 1.75 | +0.000000000 |

New guard:

- CLI:
  `--selector-candidate-first-positive-max-margin`
- default:
  `1.5`
- applies only to the positive-delta branch
- blocks sparse high-margin positive validation clusters above the cap

Final mixed 0..15 cap1 guarded result:

Artifact:

- `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_guarded_mixed_selection_compare_seed0_15.json`

| metric | value |
| --- | ---: |
| local selector selected | 2/16 |
| selected local seeds | 0, 2 |
| mean selected delta | +0.000427246 |
| mean candidate-branch delta | -0.000854492 |
| harmful selected seed count | 0 |
| harmful candidate seeds | 4 |

Seed11/seed13 calibration follow-up:

- diagnostics:
  - `artifacts/eval/nn/sedp_d7_seed11_seed13_stagec_margin_diagnostic.json`
  - `artifacts/eval/nn/sedp_d7_seed11_stagea_val_margin_diagnostic.json`
  - `artifacts/eval/nn/sedp_d7_seed11_stageb_val_margin_diagnostic.json`
  - `artifacts/eval/nn/sedp_d7_seed13_stagea_val_margin_diagnostic.json`
  - `artifacts/eval/nn/sedp_d7_seed13_stageb_val_margin_diagnostic.json`
- seed `11` at margin `1.5` is validation positive (`+0.006493506`,
  `4/2` improved/harmed) and held-out positive (`+0.003906250`, `10/6`)
- seed `13` at margin `1.75` is validation positive (`+0.009740260`,
  `3/0`) but held-out harmful (`-0.000976562`, `5/6`)
- this supports relaxing the harmed-shot cap from `1` to `2` while keeping
  `positive_max_margin=1.5`
- `decoders/syndrome_edit_predecoder.py` now defaults
  `DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED` to `2`
- `tools/simulate_predecoder_adoption_policy.py` now mirrors the positive
  harmed-shot and max-margin guards
- seed `11` cap2 rerun:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed11_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
- current cap2 mixed summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_15.json`

| metric | value |
| --- | ---: |
| local selector selected | 3/16 |
| selected local seeds | 0, 2, 11 |
| mean selected delta | +0.000671387 |
| mean candidate-branch delta | -0.000854492 |
| harmful selected seed count | 0 |
| harmful candidate seed count | 4 |

First out-of-sample cap2 seed:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed16_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_selection_compare_seed16.json`
- seed `16`: raw no-edit, reason `default_no_edit`, validation candidate
  delta `0.000000000`, held-out selected delta `0.000000000`
- current 0..16 mixed summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_16.json`

| metric | value |
| --- | ---: |
| local selector selected | 3/17 |
| selected local seeds | 0, 2, 11 |
| mean selected delta | +0.000631893 |
| mean candidate-branch delta | -0.000804228 |
| harmful selected seed count | 0 |
| harmful candidate seed count | 4 |

Second out-of-sample cap2 seed:

- artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed17_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
- comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_selection_compare_seed17.json`
- seed `17`: local selector, reason `candidate_positive_delta_with_margin`,
  margin `1.25`, validation delta `+0.009746037`, validation improved/harmed
  `4/1`
- held-out selected delta: `-0.004882812`, held-out improved/harmed `8/13`
- current 0..17 mixed summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_17.json`

| metric | value |
| --- | ---: |
| local selector selected | 4/18 |
| selected local seeds | 0, 2, 11, 17 |
| mean selected delta | +0.000325521 |
| mean candidate-branch delta | -0.001030816 |
| harmful selected seed count | 1 |
| harmful candidate seed count | 5 |

Seed `17` margin-diagnostic note:

- seed `17` is not caused by the cap2 relaxation; cap1 would also select it
  because validation harmed count is only `1`
- seed `0`, `2`, and `11` show validation positivity isolated at their
  selected margin
- seed `17` remains validation-positive at higher margins (`1.5`, `1.75`),
  but held-out remains harmful at `1.25` and `1.5`
- margin-profile artifact:
  `artifacts/eval/nn/sedp_d7_margin_profile_seed0_2_8_11_13_17.json`
- plateau-guard post-hoc simulation:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_posthoc_seed0_17.json`
- this "positive plateau" is now implemented as an optional integrated
  adoption guard:
  `--selector-candidate-first-positive-plateau-guard`
- post-hoc over seeds `0..17`, the plateau guard blocks seed `17`, keeps
  local selector `3/18`, gives mean selected delta `+0.000596788`, and has
  harmful selected seed count `0`
- d5 seed `3` compatibility artifact:
  `artifacts/eval/nn/sedp_d5_margin_profile_seed3.json`
- d5 seed `3` is not blocked by the plateau hypothesis; higher margin `1.0`
  has aggregate validation delta `0`, and held-out stage_c at selected margin
  is `+0.023437500`
- d7 seed `18`: raw no-edit; validation candidate delta `+0.003250404`,
  held-out candidate delta `-0.001953125`
- d7 seed `19`: raw no-edit; validation and held-out candidate deltas `0`
- cap2 mixed 0..19:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_19.json`
- plateau-guard post-hoc 0..19:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_posthoc_seed0_19.json`

| policy view | local selected | mean selected delta | harmful selected |
| --- | ---: | ---: | ---: |
| cap2 current | 4/20 | +0.000292969 | 1 |
| plateau post-hoc | 3/20 | +0.000537109 | 0 |
| plateau integrated | 3/20 | +0.000537109 | 0 |
| plateau integrated 0..21 | 3/22 | +0.000488281 | 0 |
| plateau integrated 0..23 | 3/24 | +0.000447591 | 0 |
| plateau integrated 0..25 | 3/26 | +0.000413161 | 0 |
| plateau integrated 0..27 | 3/28 | +0.000383650 | 0 |
| plateau integrated 0..29 | 3/30 | +0.000358073 | 0 |
| plateau integrated 0..31 | 3/32 | +0.000335693 | 0 |
| plateau integrated 0..33 | 3/34 | +0.000315947 | 0 |
| plateau integrated 0..35 | 3/36 | +0.000298394 | 0 |
| plateau integrated 0..37 | 3/38 | +0.000282689 | 0 |
| plateau integrated 0..39 | 3/40 | +0.000268555 | 0 |
| plateau integrated 0..41 | 3/42 | +0.000255766 | 0 |
| plateau integrated 0..43 | 3/44 | +0.000244141 | 0 |
| plateau integrated 0..45 | 3/46 | +0.000233526 | 0 |
| plateau integrated 0..47 | 3/48 | +0.000223796 | 0 |
| plateau integrated 0..49 | 3/50 | +0.000214844 | 0 |
| plateau integrated 0..51 | 3/52 | +0.000206581 | 0 |
| plateau integrated 0..53 | 3/54 | +0.000198929 | 0 |
| plateau integrated 0..54 failed | 4/55 | +0.000071023 | 1 |
| plateau + positive min-nonzero 5 post-hoc 0..54 | 2/55 | +0.000159801 | 0 |
| plateau + positive min-nonzero 5 sentinels 0..54 | 2/55 | +0.000159801 | 0 |
| plateau + positive min-nonzero 5 extension 0..55 | 2/56 | +0.000156948 | 0 |
| plateau + positive min-nonzero 5 extension 0..56 | 2/57 | +0.000154194 | 0 |
| plateau + positive min-nonzero 5 extension 0..57 | 2/58 | +0.000151536 | 0 |

Integrated plateau-guard verification:

- seed `17` artifact:
  `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed17_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_pairwise_seq/experiment_summary.json`
- seed `17` comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed17.json`
- seed `17` result:
  `raw_no_edit`, reason `candidate_positive_delta_plateau_guard`, held-out
  selected delta `+0.000000000`
- integrated 0..19 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_19.json`
- d7 seed `11` integrated compatibility:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed11.json`
- d5 seed `3` integrated compatibility:
  `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_plateauguard_selection_compare_seed3.json`
- d7 seed `20..21` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed20_21.json`
- integrated 0..21 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_21.json`
- d7 seed `22..23` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed22_23.json`
- integrated 0..23 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_23.json`
- d7 seed `24..25` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed24_25.json`
- integrated 0..25 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_25.json`
- d7 seed `26..27` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed26_27.json`
- integrated 0..27 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_27.json`
- d7 seed `28..29` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed28_29.json`
- integrated 0..29 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_29.json`
- d7 seed `30..31` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed30_31.json`
- integrated 0..31 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_31.json`
- d7 seed `32..33` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed32_33.json`
- integrated 0..33 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_33.json`
- d7 seed `34..35` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed34_35.json`
- integrated 0..35 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_35.json`
- d7 seed `36..37` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed36_37.json`
- integrated 0..37 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_37.json`
- d7 seed `38..39` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed38_39.json`
- integrated 0..39 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_39.json`
- d7 seed `40..41` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed40_41.json`
- integrated 0..41 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_41.json`
- d7 seed `42..43` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed42_43.json`
- integrated 0..43 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_43.json`
- d7 seed `44..45` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed44_45.json`
- integrated 0..45 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_45.json`
- d7 seed `46..47` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed46_47.json`
- integrated 0..47 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_47.json`
- d7 seed `48..49` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed48_49.json`
- integrated 0..49 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_49.json`
- d7 seed `50..51` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed50_51.json`
- integrated 0..51 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_51.json`
- d7 seed `52..53` integrated extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed52_53.json`
- integrated 0..53 summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_53.json`
- d7 seed `54` integrated failure:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_selection_compare_seed54.json`
- integrated 0..54 failed summary:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_54_failed.json`
- seed54 false-positive guard candidates:
  `artifacts/eval/nn/sedp_d7_seed54_false_positive_guard_candidates.json`
- support-guard post-hoc 0..54:
  `artifacts/eval/nn/sedp_d7_seed54_support_guard_posthoc_seed0_54.json`
- actual seed54 support-guard sentinel:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
- actual support-guard sentinel comparison:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
- support-guard sentinel mixed 0..54:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_54_sentinel.json`
- support-guard seed55 extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed55.json`
- support-guard mixed 0..55:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_55.json`
- support-guard seed56 extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed56.json`
- support-guard mixed 0..56:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_56.json`
- support-guard seed57 extension:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed57.json`
- support-guard mixed 0..57:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json`
- d7 seed `11` remains a true positive at held-out delta `+0.003906250`
- d5 seed `3` remains a true positive at held-out delta `+0.023437500`
- d7 seeds `20` and `21` both select raw no-edit with held-out delta `0`
- d7 seeds `22` and `23` both select raw no-edit with held-out delta `0`
- d7 seeds `24` and `25` both select raw no-edit with held-out delta `0`
- d7 seed `26` selects raw no-edit, while its candidate branch is harmful
  (`-0.011718750`); seed `27` selects raw no-edit with candidate delta `0`
- d7 seed `28` selects raw no-edit, while its candidate branch is positive
  by one shot (`+0.000976562`); seed `29` selects raw no-edit with candidate
  delta `0`
- d7 seeds `30` and `31` both select raw no-edit with selected/candidate
  deltas `0`
- d7 seeds `32` and `33` both select raw no-edit by harm guard; their
  candidate branches are harmful (`-0.010742188`, `-0.016601562`)
- d7 seed `34` selects raw no-edit while its candidate branch is harmful
  (`-0.002929688`); seed `35` selects raw no-edit with candidate delta `0`
- d7 seed `36` selects raw no-edit while its candidate branch is harmful
  (`-0.001953125`); seed `37` selects raw no-edit with candidate delta `0`
- d7 seed `38` selects raw no-edit while its candidate branch is harmful
  (`-0.001953125`); seed `39` selects raw no-edit with candidate delta `0`
- d7 seed `40` selects raw no-edit with candidate delta `0`; seed `41`
  selects raw no-edit while its candidate branch is harmful (`-0.000976562`)
- d7 seed `42` selects raw no-edit with candidate delta `0`; seed `43`
  selects raw no-edit while its candidate branch is positive (`+0.000976562`)
- d7 seed `44` selects raw no-edit with candidate delta `0`; seed `45`
  selects raw no-edit while its candidate branch is positive (`+0.000976562`)
- d7 seed `46` and seed `47` both select raw no-edit with candidate delta `0`
- d7 seed `48` and seed `49` both select raw no-edit with candidate delta `0`
- d7 seed `50` and seed `51` both select raw no-edit with candidate delta `0`
- d7 seed `52` selects raw no-edit with candidate delta `0`; seed `53`
  selects raw no-edit by harm guard while its candidate branch is harmful
  (`-0.010742188`)
- d7 seed `54` selects local selector at margin `1.25`, validation delta
  `+0.006508300`, but held-out selected delta is `-0.006835938`; validation
  support is stage_a-only at the selected margin
- adding `--selector-candidate-first-positive-min-nonzero 5` blocks seed `54`
  in an actual rerun with reason `candidate_positive_delta_support_guard`
- post-hoc, min-nonzero `5` preserves seed `2` and seed `11`, blocks seed
  `54`, and sacrifices seed `0`
- actual min-nonzero `5` sentinel reruns confirm seed `2` and seed `11` remain
  local selector with held-out deltas `+0.004882812` and `+0.003906250`
- seed `55` under min-nonzero `5` selects raw no-edit and blocks a harmful
  candidate branch with held-out delta `-0.004882812`
- seed `56` under min-nonzero `5` selects raw no-edit and blocks a harmful
  candidate branch with held-out delta `-0.000976562`
- support-guard mixed 0..56 selects only seeds `2,11`, mean selected delta is
  `+0.000154194`, candidate-branch mean is `-0.001541941`, and harmful
  selected count remains `0`
- seed `57` under min-nonzero `5` selects raw no-edit; held-out candidate and
  selected deltas are both `0`
- support-guard mixed 0..57 selects only seeds `2,11`, mean selected delta is
  `+0.000151536`, candidate-branch mean is `-0.001515356`, and harmful
  selected count remains `0`
- candidate-oracle analysis over support-guard 0..57:
  `artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json`
  - all seeds have positive oracle headroom
  - mean candidate-oracle delta is `+0.096679688`
  - actual candidate mean delta is `-0.001515356`
  - candidate outcomes are `6` positive, `35` neutral, and `17` harmful
- true/false diagnostic for seeds `2,11,54`:
  `artifacts/eval/nn/sedp_d7_support_guard_true_false_selection_diagnostic_seed2_11_54_stagec.json`
  - seed `2` at margin `1.25`: `+0.004882812`, improved/harmed `6/1`
  - seed `11` at margin `1.5`: `+0.003906250`, improved/harmed `10/6`
  - seed `54` at margin `1.25`: `-0.006835938`, improved/harmed `6/13`
- oracle/harm ranking diagnostic for seeds `2,11,54,55`:
  `artifacts/eval/nn/sedp_d7_support_guard_oracle_harm_ranking_diagnostic_seed2_11_54_55_stagec.json`
  - seed `2` margin `1.25`: oracle above margin `6`, negative above margin
    `1`
  - seed `11` margin `1.5`: oracle above margin `10`, negative above margin
    `6`
  - seed `54` margin `1.25`: oracle above margin `6`, negative above margin
    `13`
  - seed `55` margin `1.75`: oracle above margin `8`, negative above margin
    `13`
- optional hard-negative identity-margin loss was implemented with default
  weight `0` for compatibility
- hard-negative sentinel `weight=1.0`, `margin=1.5`:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_negidmargin10_m15_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
  - seed `54` candidate delta improves to `-0.001953125`, but remains harmful
  - seed `2` candidate delta becomes `-0.003906250`, so the known true-positive
    is destroyed
  - verdict: reject this setting as too strong
- hard-negative seed54 ranking diagnostic:
  `artifacts/eval/nn/sedp_d7_negidmargin10_m15_oracle_harm_ranking_diagnostic_seed54_stagec.json`
  - negative-over-identity count falls from `110` to `8`
  - oracle-positive over-margin evidence is also suppressed, so this is not a
    viable final recipe
- weak hard-negative sentinel `weight=0.25`, `margin=1.0`:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_negidmargin025_m10_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
  - seed `54` selects `local_motif_selector`
  - held-out selected/candidate delta `-0.001953125`
  - validation margin `1.0`, nonzero `5`, improved/harmed `4/1`; this passes
    the support guard and is therefore unsafe
  - verdict: reject this setting; seed `2` was not run
- weak hard-negative seed54 ranking diagnostic:
  `artifacts/eval/nn/sedp_d7_negidmargin025_m10_oracle_harm_ranking_diagnostic_seed54_stagec.json`
  - at margin `1.0`, oracle-positive above margin `6`, negative-target above
    margin `9`
- correct-split validation ranking-guard diagnostic:
  `artifacts/eval/nn/sedp_d7_support_guard_validation_ranking_guard_summary_seed2_11_54_55.json`
  - validation diagnostics for this multi-family recipe must use the training
    split convention: `stage_a` split seed `seed`, `stage_b` split seed
    `seed + 1`; the default split seed `0` is not reliable for most adoption
    decisions
  - the tested guard was: block adoption if validation negative-target
    above-margin count exceeds oracle-positive above-margin count at the
    candidate adoption margin
  - seed `2` and seed `11` are preserved, but seed `54` also passes this
    statistic with validation oracle/negative above-margin counts `2/0` at
    margin `1.25`, while held-out candidate delta is `-0.006835938`
  - weak hard-negative seed `54` also passes with combined correct-validation
    counts `3/1` at margin `1.0`, while held-out delta is `-0.001953125`
  - verdict: simple validation negative-over-margin excess is not a sufficient
    d7 adoption guard
- hard positive-vs-negative ranking sentinel:
  - code path: `--selector-positive-negative-hard-loss-weight` and
    `--selector-positive-negative-hard-margin`; defaults preserve old behavior
  - comparison:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
  - weight `1.0`, margin `0.5` changes the sentinel rows:
    seed `2` selected `+0.004882812 -> +0.003906250`;
    seed `11` selected `+0.003906250 -> 0` despite candidate
    `+0.004882812`; seed `54` candidate `-0.006835938 -> -0.003906250`
  - ranking diagnostic for seed `54`:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed54_stagec.json`
    - at margin `1.5`, oracle-positive above margin `6`, negative-target
      above margin `10`
  - verdict: partial candidate-branch improvement, but not an adoptable
    selected-mode recipe
- plateau-aware adoption calibration for the hard positive-vs-negative
  checkpoint family:
  - `tools/simulate_predecoder_adoption_policy.py` supports
    `--positive-plateau-guard`
  - artifact:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54.json`
  - with `positive_delta=0.003`, `positive_min_nonzero=1`, and plateau guard,
    the `1.0/0.5` checkpoint family selects seeds `2,11`, blocks seed `54`,
    and reaches mean selected delta `+0.002929688`
  - this ties the old support-guard selected mean on seeds `2,11,54`, but
    improves mean candidate delta from `+0.000651042` to `+0.001627604`
- weaker hard positive-vs-negative `0.5/0.5` was tested:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard05_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
  - original support-guard adoption selects all no-edit, mean selected delta
    `0`
  - calibrated adoption artifact:
    `artifacts/eval/nn/sedp_d7_posneghard05_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54.json`
  - it recovers only seed `11`, mean policy delta `+0.002278646`, while seed
    `2` candidate becomes harmful (`-0.003906250`)
  - verdict: reject `0.5/0.5`
- `1.0/0.5` hard positive-vs-negative extension to seeds `55,56,57`:
  - comparison over seeds `2,11,54,55,56,57`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54_57.json`
  - calibrated adoption over the same seeds:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54_57.json`
  - old support-guard on the same seeds: mean selected
    `+0.001464844`, mean candidate `-0.000651042`
  - `1.0/0.5` with calibrated adoption: mean selected `+0.001464844`,
    mean candidate `-0.002278646`
  - seed `55`: candidate `-0.006835938`, blocked by harm guard
  - seed `57`: candidate `-0.011718750`, blocked by harm guard
  - diagnostics:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed55_stagec.json`
    and
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed57_stagec.json`
    - seed `55`, margin `1.5`: improved/harmed `19/26`,
      oracle/negative above-margin `18/26`
    - seed `57`, margin `1.5`: improved/harmed `7/10`,
      oracle/negative above-margin `3/10`
  - verdict: reject broad extension of `1.0/0.5`
- simple family-level stage-consistency adoption check:
  - simulator support:
    `tools/simulate_predecoder_adoption_policy.py` now emits validation
    family-level candidate deltas/nonzero/improved/harmed counts and supports
    `--positive-family-min-delta`, `--positive-min-family-count`, and
    `--positive-max-family-harmed`
  - all-family nonnegative validation guard on the `1.0/0.5` calibrated
    adoption artifact:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_familymin0_count2_seed2_11_54_57.json`
  - result: blocks true-positive seed `2` because validation is mixed
    (`stage_a=-0.006493506`, `stage_b=+0.025974026`); mean policy delta drops
    to `+0.000813802`
  - family harmed-cap `2` on the same sentinel:
    `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_familymaxharm2_seed2_11_54_57.json`
  - result: same selected behavior as calibrated adoption, mean policy delta
    `+0.001464844`; no added discrimination
  - all-family nonnegative guard on the original support-guard recipe:
    `artifacts/eval/nn/sedp_d7_support_adoption_sim_posdelta003_posminnz1_plateau_familymin0_count2_seed2_11_54_57.json`
  - result: also blocks seed `2`, reducing mean policy delta to
    `+0.000651042`
  - verdict: reject simple family-level post-hoc adoption guards
- cross-family hard positive-vs-negative selector objective:
  - implemented default-off in `decoders/syndrome_edit_predecoder.py`:
    `--selector-cross-family-positive-negative-loss-weight` and
    `--selector-cross-family-positive-negative-margin`
  - design: a positive nonzero candidate from one training family is trained
    to outrank a hard negative nonzero candidate sampled from another training
    family
  - weak seed `54` sentinel `0.25/0.5`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_crossfam025_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
  - result: candidate branch remains `-0.006835938`, with improved/harmed
    `6/13`; selected mode remains safe raw no-edit
  - strong seed `54` sentinel `1.0/0.5`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_crossfam10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
  - result: candidate branch worsens to `-0.009765625`, with improved/harmed
    `8/18`; selected mode remains safe raw no-edit
  - verdict: reject this simple cross-family hard-negative objective

Interpretation:

- d7 is not safe over the checked 20 seeds with cap2 + max-margin guard alone
- integrated plateau guard is selected-safe only through seeds `0..53`; seed
  `54` invalidates the current adoption guard as a final safe policy
- the selected gain is still sparse, but seed `11` is now recovered without
  reopening seed `8` or seed `13`
- the current d7 bottleneck is no longer candidate-set coverage: oracle
  headroom is large, but the learned selector often ranks neutral or harmful
  edits above identity; the next work should focus on a target/ranking change
  that suppresses negative-target over-margin crossings without erasing
  true-positive nonzero evidence. Hard-negative identity-margin alone is now
  rejected: strong `1.0/1.5` erases seed `2`, weak `0.25/1.0` lets seed `54`
  through. The simple validation negative-over-margin excess guard is also
  rejected because it does not catch seed `54`; prefer a positive-vs-negative
  ranking redesign or a stronger cross-split/stage-generalization diagnostic.
  The hard positive-vs-negative `1.0/0.5` checkpoint family does not extend
  beyond the original sentinel; `0.5/0.5` is also rejected. A simple
  family-level validation threshold is also rejected because it blocks seed
  `2`, a true-positive held-out case with mixed validation-family behavior.
  A first in-training cross-family hard-negative objective is also rejected
  because it fails the false-positive seed `54` gate: weak weight does not
  improve the harmful candidate branch, and strong weight makes it worse.
  Further d7 work needs a genuinely stage-consistency-aware selector objective
  or should be deprioritized in favor of consolidating d3/d5 plus the d7
  limitation.
- consolidated evidence summary:
  `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`
  plus `PREDECODER_CONSOLIDATED_EVIDENCE.md`,
  `PREDECODER_FINAL_RESULT_TABLES.md`, and
  `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
  now fix the current model-selection boundary:
  d3 `+0.007568359`, d5 `+0.011230469`, and d7 support-guard
  `+0.000151536` selected mean held-out `stage_c_corr` deltas.
  The d7 candidate-oracle mean is still `+0.096679688`, so the remaining d7
  limitation is selector ranking/generalization, not candidate coverage.
- d7 targeted bottleneck analysis now exists:
  `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
  plus `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`; it fixes the next
  sentinel sets as preserve `2,11`, recover `0,28,43,45`, and block
  validation false positives `13,17,33,54,53,32,8,18`.
- d7 simple adoption-grid diagnostic:
  `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json`
  checked `183040` monotone threshold policies and found `0` passing policies;
  the best recovery policy recovers `0,28,43,45` but opens harmful seeds
  `13,17,18,54`, so threshold-only selected-mode tuning is no longer a
  rational next step.
- d7 candidate-compatibility pairwise top-k sentinel:
  `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
  blocks seed54 but destroys the seed2 true-positive candidate branch
  (`candidate delta -0.136718750`, `56/196` improved/harmed), so this recipe
  should not be expanded.
- remaining work is now tracked in `PREDECODER_REMAINING_WORK.md`; the default
  finish path is to freeze d3/d5 as the positive selected-mode result and write
  d7 as a selector-ranking/generalization scaling limitation unless a new
  objective passes the preserve/recover/block sentinel gate.

Do not spend the next iteration copying the exact Jung/Ali/Ha CNN structure.
Keep `RectCNN` only as a baseline/checkpoint for the paper-format reference.
