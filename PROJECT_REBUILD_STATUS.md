# Project Rebuild Status

This file records which parts of the existing codebase stay on the rebuilt mainline,
which parts stay only as reference, and which parts must keep changing.

Current detailed target schedule:

- `RESEARCH_PLAN_PREDECODER_MAIN.md`
- `MAIN_TARGET_WORK_SCHEDULE.md`
- `PREDECODER_REMAINING_WORK.md`

## Research Snapshot As Of 2026-05-02

The current research topic is fixed as:

> Transition-aware neural pre-decoding for surface-code logical-frame
> inference.

The final evaluated system is not a standalone neural `logical_class4` decoder.
The current best system is:

```text
syndrome volume
  -> neural patch-head local edit selector
  -> edited or unchanged syndrome
  -> PyMatching
  -> logical_class4 prediction
```

The current evidence base is:

- raw PyMatching baselines are complete for d3/d5/d7 class4 2k datasets
- direct single-model neural decoders remain secondary/negative baselines
  because they collapse with distance
- baseline comparison is now consolidated in
  `PREDECODER_BASELINE_COMPARISON.md` and
  `artifacts/eval/nn/sedp_baseline_comparison_summary.json`; the fair main
  claim compares selected predecoder against raw PyMatching on the same
  predecoder target artifacts, while FLFD/M3D/RectCNN are context baselines
- ablation/failure-path synthesis is now consolidated in
  `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md` and
  `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json`; it
  explains why direct neural classification, multiscale direct classification,
  scalar d7 adoption tuning, cross-family hard-negative training, and
  candidate-compatibility top-k are not the final direction
- final result consistency is checked in
  `artifacts/eval/nn/sedp_final_result_consistency_check.json`, with `37`
  Markdown-vs-consolidated-JSON checks passing and `0` failures
- the canonical method description is now `PREDECODER_METHOD_DESCRIPTION.md`
- d7 harmful-edit taxonomy is now consolidated in
  `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md` and
  `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`; the key
  result is validation-positive false-positive behavior (`13/22` harmful on
  held-out) with selected mode blocking all `17/17` harmful candidate seeds
- thesis core integration is now drafted in `GRADUATION_THESIS_DRAFT.md`,
  covering method, setup, results, ablation, d7 limitation, and discussion
- clean Korean thesis core prose is now drafted in
  `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`
- reproducibility packaging is now consolidated in
  `PREDECODER_REPRODUCIBILITY_PACKAGE.md`
- thesis figure packaging is now consolidated in
  `PREDECODER_FIGURE_PACKAGE.md`, with four SVG figures under
  `artifacts/figures/predecoder/`
- local-edit target builders show high oracle headroom, so candidate generation
  is not the main blocker
- d3 and d5 patch-head with non-inferiority selected-mode adoption select the
  local selector on seeds `0..3` and improve mean held-out `stage_c_corr`
- d7 requires selected no-edit guarding; with margin `0.005`, d7 no longer
  underperforms raw PyMatching on mean held-out `stage_c_corr`
- the remaining-work plan is now consolidated in
  `PREDECODER_REMAINING_WORK.md`: integrate the final result table, method
  description, ablation/failure-path synthesis, d3/d5 success writeup, d7
  limitation taxonomy/writeup, and reproducibility package before any optional
  new experiment; the next default step is final school-format polishing,
  figure insertion, appendix integration, and final validation

Current 4-seed held-out `stage_c_corr` selected comparison:

| distance | adopted seeds | PyMatching | selected decoder | delta |
| --- | ---: | ---: | ---: | ---: |
| d3 | 4/4 | 0.928710938 | 0.938232422 | +0.009521484 |
| d5 | 4/4 | 0.888671875 | 0.898925781 | +0.010253906 |
| d7 | 1/4 selector, 3/4 raw_no_edit | 0.873046875 | 0.875244141 | +0.002197266 |

The d7 candidate oracle remains high (`~0.988037109` mean under the guarded
sweep). The selected d7 path is now safe but mostly raw no-edit, so the next
problem is recovering d7 oracle gap under the guardrail.

The next default work is no longer broad d7 tuning. Optional d7 work must first
pass the sentinel gate; otherwise the project should present d7 as a controlled
selector-ranking/generalization limitation.

Current session follow-up:

- `decoders/syndrome_edit_predecoder.py` now supports
  `--selector-adoption-min-delta`
- the default `0.0` makes selector adoption a non-inferiority test against
  validation global policy, so exact validation ties no longer force fallback
  to `global_policy`
- diagnostic replay on existing patch-head summaries shows the old seed `2`
  checkpoint would switch from fallback to `local_motif_selector` because its
  validation selector/global metrics were tied while its held-out candidate
  branch improved `stage_c_corr 0.888671875 -> 0.893554688`
- fresh full d5 seed sweep artifacts are now recorded at:
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed1_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed2_adopt_noninferior/experiment_summary.json`
  - `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed3_adopt_noninferior/experiment_summary.json`
- the new sweep selects `local_motif_selector` for seeds `0..3`
- mean held-out `stage_c_corr` selected delta improves from old
  strict-adoption `+0.003662109` to new non-inferiority `+0.010253906`
- new selected mean deltas are also positive on seen families:
  `stage_a_si1000 +0.010986328`, `stage_b_local +0.009521484`
- active patch-head distance ladder is now complete:
  - d3 mean `stage_c_corr` delta: `+0.009521484`
  - d5 mean `stage_c_corr` delta: `+0.010253906`
  - d7 mean `stage_c_corr` delta: `-0.004394531`
- selected no-edit guardrail is now implemented:
  - selected mode `raw_no_edit`
  - `--selected-no-edit-guardrail`
  - `--selected-no-edit-min-delta`
- d7 guardrail with min-delta `0.005`:
  - artifacts:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_adopt_noninferior_noeditguard_margin005/experiment_summary.json`
    through seed `3`
  - mean held-out `stage_c_corr` delta: `+0.002197266`
- d7 guarded seed-selection calibration:
  - new tool: `tools/compare_predecoder_seed_sweep.py`
  - comparison artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_noeditguard_margin005_selection_compare.json`
  - absolute selected validation metric chooses seed `0` (`raw_no_edit`,
    held-out delta `+0.000000000`)
  - validation delta over no-edit chooses seed `3`
    (`local_motif_selector`, held-out `stage_c_corr`
    `0.873046875 -> 0.881835938`, delta `+0.008789062`)
- d7 extended seed check:
  - seeds `4..7` are complete with the active local-motif candidate recipe
    (`--selector-local-motif-max-classes 16`)
  - comparison artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_noeditguard_margin005_selection_compare_seed0_7.json`
  - over seeds `0..7`, selected modes are `7/8` `raw_no_edit` and `1/8`
    `local_motif_selector`
  - 8-seed mean held-out `stage_c_corr` selected delta is `+0.001098633`
  - seed `5` confirms the guardrail is useful: candidate validation delta
    `+0.003249608` does not clear margin `0.005`, and held-out candidate
    delta is actually `-0.009765625`
- seed `3` / seed `5` diagnostic:
  - new tool: `tools/diagnose_predecoder_selection.py`
  - artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed3_vs_seed5_stagec_selection_diagnostic.json`
  - seed `3` selects `17` held-out `stage_c_corr` nonzero edits at margin
    `1.25`, with `13` positive and `4` negative target-score edits
  - seed `5` candidate branch selects `68` nonzero edits at margin `0.0`,
    with `29` positive and `39` negative target-score edits
  - seed `5` best-nonzero logit-gap max is `1.209157`, below seed `3`'s
    margin `1.25`, so this is a broad low-gap over-edit case
- post-hoc margin floor check:
  - artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_7_stagec_margin125_diagnostic.json`
  - margin `1.25` preserves the seed `3` gain and suppresses seed `5` to
    no-edit
  - over seeds `0..7`, only seed `3` emits nonzero edits at this margin, so
    the floor is a safety calibration rather than a robust d7 recovery fix
- margin-sweep calibration:
  - summary artifact:
    `artifacts/eval/nn/sedp_d7_seed3_seed5_margin_sweep_summary.json`
  - seed `3` validation selects margin `1.25` with mean delta
    `+0.006493506`, and held-out `stage_c_corr` at that margin is
    `+0.008789062`
  - seed `5` validation best is only `+0.003246753`, below the active
    `0.005` no-edit guard margin, and its held-out `stage_c_corr` at that
    margin is `-0.009765625`
  - current conclusion: keep d7 `--selected-no-edit-min-delta 0.005`; the
    guard margin is doing real work, but d7 nonzero recovery remains unstable
- 8-seed fixed-margin profile:
  - artifact:
    `artifacts/eval/nn/sedp_d7_seed0_7_margin125_validation_heldout_profile.json`
  - at selector margin `1.25`, seed `3` is the only seed with validation
    nonzero edits and the only seed with held-out `stage_c_corr` nonzero edits
  - seed `3` validation mean delta is `+0.006493506`; held-out
    `stage_c_corr` delta is `+0.008789062`
  - seed `5` held-out max best-nonzero gap is `1.209157`, below the margin;
    seed `3` reaches `1.923424`
  - current diagnosis: the d7 issue is now score-scale/training stability
    across seeds, not local candidate availability
- seed-control follow-up:
  - `syndrome_edit_predecoder.py` now seeds numpy/torch at training entry, so
    future predecoder seed sweeps control model initialization, sampler order,
    and selector group shuffling as well as split seeds
  - new selector epoch margin diagnostics are available via
    `--selector-epoch-diagnostic-margin-grid`
  - seed-fixed d7 seed `3` and seed `5` reruns both selected `raw_no_edit`;
    seed `3` candidate held-out `stage_c_corr` at its selected margin was
    `-0.001953125`, while post-hoc margin `1.25` improved by only one shot
  - therefore the old d7 seed `3` gain remains useful evidence that a high-gap
    local cluster can help, but it is not yet deterministic seed-stability
    evidence
- deterministic seed-fixed d7 `0..7` sweep:
  - selected modes: `7/8` `raw_no_edit`, `1/8` `local_motif_selector`
  - the adopted local case is seed `2`; it clears validation by
    `+0.012998091` but hurts held-out `stage_c_corr` by `-0.004882812`
  - seed-fixed mean selected held-out `stage_c_corr` delta is
    `-0.000610352`
  - seed `2` harm is low-margin over-editing: margin `0.0` selects `39`
    nonzero edits with `17/22` improved/harmed, while margin `1.0+` suppresses
    all edits
  - validation delta over no-edit is now only a diagnostic, not a sufficient
    d7 adoption rule; d7 needs a margin floor or stronger calibration before
    any learned selected-mode claim
- d7 margin-floor recipe check:
  - seed `2` was rerun with selector emit-margin grid restricted to
    `1.0 1.25 1.5 1.75 2.0 4.0`
  - selected mode switched from harmful `local_motif_selector` at margin `0.0`
    to `raw_no_edit` at margin `1.0`
  - held-out `stage_c_corr` changed from `-0.004882812` in the original
    seed-fixed recipe to `+0.000000000`
  - this validates the margin floor as a d7 safety recipe, but it does not
    recover d7 learned gain
- d5 seed-fixed revalidation:
  - seed-fixed d5 seeds `0..3` were rerun after the RNG-control fix
  - original no-guard selected-mode mean held-out `stage_c_corr` delta is now
    only `+0.001220703`, not the previous `+0.010253906`
  - seed `1` selects a harmful `global_policy` branch:
    validation delta `+0.003246753`, held-out delta `-0.018554688`
  - post-hoc no-edit guard with margin `0.005` would block seed `1` and gives
    mean selected held-out delta `+0.005859375`
  - candidate branch mean held-out delta is `+0.011230469`, with seed `2`
    candidate branch positive but not selected
- d3 seed-fixed revalidation:
  - seed-fixed d3 seeds `0..3` remain positive on held-out `stage_c_corr`
  - per-seed selected deltas are `+0.010742188`, `+0.006835938`,
    `+0.008789062`, and `+0.003906250`
  - mean selected/candidate held-out delta is `+0.007568359`
  - no-edit guard margin `0.005` would not change d3 selected mode
- seed-fixed distance ladder:
  - summary artifact:
    `artifacts/eval/nn/sedp_seedfixed_distance_ladder_summary.json`
  - d3 remains a stable positive result
  - d5 has learned candidate signal but selected-mode calibration is weak
  - d7 has no stable learned gain; it is safe only with no-edit/margin-floor
    behavior
- candidate-first adoption policy simulation:
  - new tool: `tools/simulate_predecoder_adoption_policy.py`
  - artifact:
    `artifacts/eval/nn/sedp_seedfixed_candidate_first_adoption_policy_sim.json`
  - post-hoc policy preserves d3 mean `+0.007568359`, raises d5 to the
    candidate-branch mean `+0.011230469`, and keeps d7 at safe no-edit
    `+0.000000000`
  - this policy blocks harmful d5 seed `1` and d7 seed `2`, while adopting
    d5 seed `2`'s positive candidate branch
  - next step is implementation/evaluation inside the decoder rather than
    relying only on post-hoc simulation
- candidate-first adoption policy integration:
  - `decoders/syndrome_edit_predecoder.py` now supports
    `--selector-adoption-policy candidate_first_safety`
  - default behavior remains `global_noninferiority`
  - default candidate-first thresholds match the simulator:
    strong delta `0.02`, positive delta `0.005` with margin floor `0.5`,
    and tied high-margin adoption with margin floor `1.0` plus at least `6`
    validation nonzero edits
  - validation passed:
    `python -m py_compile decoders\syndrome_edit_predecoder.py tools\diagnose_predecoder_selection.py tools\compare_predecoder_seed_sweep.py tools\simulate_predecoder_adoption_policy.py`
  - actual d5 policy artifacts:
    `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise/experiment_summary.json`
    through seed `3`
  - d5 comparison artifact:
    `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_3.json`
  - d5 actual held-out `stage_c_corr` mean selected delta is
    `+0.011230469`, matching the post-hoc candidate-first simulation and the
    candidate-branch mean
  - d5 selected modes are raw no-edit for seeds `0,1` and
    `local_motif_selector` for seeds `2,3`; seed `2` is adopted by
    `candidate_tie_with_high_margin_evidence`, seed `3` by
    `candidate_positive_delta_with_margin`
  - d7 seed `2` safety smoke artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_smoke_pairwise/experiment_summary.json`
  - d7 seed `2` now selects `raw_no_edit` with held-out delta `0.0`, while
    its candidate branch remains harmful at `-0.004882812`
  - canonical full d7 policy artifacts:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise_seq/experiment_summary.json`
    through seed `7`
  - canonical d7 comparison artifact:
    `artifacts/eval/nn/sedp_d7_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json`
  - d7 selected modes are `raw_no_edit` for all seeds `0..7`
  - d7 mean selected held-out `stage_c_corr` delta is `+0.000000000`
  - d7 mean candidate-branch held-out `stage_c_corr` delta is
    `-0.000854492`
  - canonical d3 integrated-policy regression artifacts:
    `artifacts/eval/nn/sedp_d3_router1k_benefitharm_patchhead_seed0_candidatefirst_policy_pairwise_seq/experiment_summary.json`
    through seed `3`
  - d3 comparison artifact:
    `artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_3.json`
  - d3 selects `local_motif_selector` for all seeds `0..3` and preserves mean
    held-out `stage_c_corr` delta `+0.007568359`
  - distance ladder artifact:
    `artifacts/eval/nn/sedp_candidatefirst_distance_ladder_summary.json`
  - final integrated candidate-first ladder:
    d3 `+0.007568359`, d5 `+0.011230469`, d7 `+0.000000000`
- d7 recovery epoch diagnostics:
  - new tool:
    `tools/summarize_selector_epoch_diagnostics.py`
  - artifacts:
    `artifacts/eval/nn/sedp_d3_candidatefirst_seq_epoch_diagnostic_summary_seed0_3.json`,
    `artifacts/eval/nn/sedp_d5_candidatefirst_epoch_diagnostic_summary_seed0_3.json`,
    `artifacts/eval/nn/sedp_d7_candidatefirst_seq_epoch_diagnostic_summary_seed0_7.json`,
    and
    `artifacts/eval/nn/sedp_d7_recovery_epoch_diagnostic_comparison.json`
  - positive nonzero epoch/margin rows:
    d3 `66`, d5 `14`, d7 `6`
  - margin `>=1` positive nonzero rows:
    d3 `48`, d5 `14`, d7 `3`
  - candidate-first high-margin tied-evidence rows:
    d3 `48`, d5 `10`, d7 `0`
  - best d7 validation row is seed `2`, epoch `4`, margin `0.0`, delta
    `+0.012987013`, nonzero `12`, improved/harmed `8/4`; this is low-margin
    evidence and the canonical held-out candidate branch for seed `2` is
    harmful at `-0.004882812`
  - d7 margin `>=1` positive rows have only `1-2` validation nonzero edits,
    too sparse for selected-mode adoption
  - conclusion: the current d7 no-gain result is not an adoption-threshold
    problem; d7 needs training/score-scale changes that create robust
    high-margin positive clusters
- d7 identity-margin + diagnostic epoch-selection recovery check:
  - code now supports optional
    `--selector-epoch-selection-mode diagnostic_system`; default selector
    epoch selection remains `proxy`
  - comparison tool now reports selector epoch, adoption reason, selected
    margin, and validation nonzero support
  - artifacts:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_idmargin05_diagselect_pairwise_seq/experiment_summary.json`
    through seed `7`
  - comparison:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_7.json`
  - epoch diagnostics:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_epoch_diagnostic_summary_seed0_7.json`
  - selected modes are `2/8` `local_motif_selector` and `6/8`
    `raw_no_edit`
  - mean held-out `stage_c_corr` selected delta is `+0.000854492`;
    candidate-branch mean is `+0.000488281`
  - adopted seeds:
    seed `0` delta `+0.001953125`, seed `2` delta `+0.004882812`
  - seed `5` remains a safety check: its candidate branch is harmful on
    held-out `stage_c_corr` (`-0.002929688`) and the policy keeps selected
    mode at raw no-edit
  - conclusion: this is the first seed-fixed d7 learned selected-mode gain
    under the canonical safety policy, but it is still sparse and should be
    treated as a recovery signal rather than a solved d7 mechanism
- d7 identity-margin weight sentinel ablation:
  - compared identity-margin loss weights `0.25`, `0.5`, and `1.0` on seeds
    `0,2,5` with diagnostic epoch selection and unchanged candidate-first
    adoption
  - comparison artifacts:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin025_diagselect_selection_compare_seed0_2_5.json`,
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_2_5.json`,
    and
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin10_diagselect_selection_compare_seed0_2_5.json`
  - seed-sentinel mean held-out `stage_c_corr` selected deltas:
    weight `0.25` `+0.000651042`, weight `0.5` `+0.002278646`,
    weight `1.0` `+0.000000000`
  - weight `0.25` is unsafe because seed `0` is adopted but has held-out
    selected delta `-0.002929688`
  - weight `1.0` is too conservative because it suppresses both positive
    sentinel seeds to raw no-edit
  - conclusion: keep weight `0.5`; do not spend a full d7 sweep on `0.25` or
    `1.0` unless a later epoch-selection change makes them relevant
- small-volume d7 epoch-selection probe:
  - post-hoc support-aware tie-break on existing seed `0,2,5` records did not
    materially change epoch choices
  - seed `2` was rerun with selector epoch diagnostic grid
    `0.0 1.0 1.25 1.5`
  - artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_diagselect_grid015_pairwise_seq/experiment_summary.json`
  - result is unchanged from the existing seed `2` run:
    selector epoch `6`, margin `1.25`, validation nonzero `5`, held-out
    `stage_c_corr` selected delta `+0.004882812`
  - conclusion: do not expand the wider diagnostic grid probe; it does not
    improve the strongest positive seed
- small-volume d7 selector-epoch count probe:
  - seed `2` was rerun with `--selector-epochs 8` under the active
    `idmargin0.5 + diagnostic_system + candidate_first_safety` recipe
  - artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed2_candidatefirst_idmargin05_diagselect_epochs8_pairwise_seq/experiment_summary.json`
  - selected result is unchanged from the 6-epoch run:
    selector epoch `6`, margin `1.25`, validation nonzero `5`, held-out
    `stage_c_corr` selected delta `+0.004882812`
  - epoch `7` has a validation tie with more support, but epoch `8` shows
    over-edit risk (`-0.003246753` at margin `1.25`, `6/7`
    improved/harmed)
  - conclusion: keep default selector epochs `6`; do not expand this probe
- next-session constraint:
  - usage is limited, so the next d7 step must start with post-hoc analysis,
    not a full training sweep
  - compare alternative selector epoch-selection scores on existing
    `idmargin0.5 + diagnostic_system` epoch diagnostics
  - train at most one seed first if the post-hoc rule changes the expected
    selected epoch
  - keep candidate-first thresholds, identity-margin weight `0.5`,
    diagnostic grid `0.0 1.0 1.25`, and selector epochs `6` as the current
    active settings until a one-seed probe justifies changing them
- d7 seed8 false positive and positive harm-cap guard:
  - extending the old d7 `idmargin0.5 + diagnostic_system` recipe to seed `8`
    exposed a serious selected-mode false positive
  - old seed `8` selected `local_motif_selector` with validation delta
    `+0.006481003`, margin `2.0`, and validation improved/harmed `6/4`
  - held-out `stage_c_corr` selected delta was `-0.019531250`
    (`7/27` improved/harmed)
  - `decoders/syndrome_edit_predecoder.py` now supports
    `--selector-candidate-first-positive-max-harmed`; it was introduced at
    cap `1` and is currently calibrated to default `2`
  - the guard blocks positive-delta adoption when validation harmed count is
    above the cap and prevents fallback into the high-margin tie branch for
    that same positive-delta candidate
  - d3/d5 post-hoc compatibility is preserved: d3 uses strong-delta adoption,
    d5 seed `2` remains a high-margin tie, and d5 seed `3` has only one
    validation harmed shot
  - actual d7 sentinel rerun artifacts:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed0_candidatefirst_idmargin05_diagselect_posharmcap1_pairwise_seq/experiment_summary.json`,
    seed `2`, and seed `8`
  - comparison:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap1_selection_compare_seed0_2_8.json`
  - over seeds `0,2,8`, mean selected held-out delta is `+0.002278646`;
    seed `8` is raw no-edit with reason
    `candidate_positive_delta_harm_guard`
  - next step: continue robustness extension with harm cap enabled, starting
    with seeds `9..11`
- d7 guarded robustness extension to 16 seeds:
  - seeds `9..11` with harm cap were selected-mode safe
  - seed `11` candidate branch was held-out positive (`+0.003906250`) but was
    blocked because validation harmed count was `2`
  - seed `13` exposed a second failure mode: with harm cap only, it selected
    local selector at margin `1.75` and was held-out harmful by one shot
    (`-0.000976562`, `5/6` improved/harmed)
  - `decoders/syndrome_edit_predecoder.py` now supports
    `--selector-candidate-first-positive-max-margin`, default `1.5`
  - seed `13` rerun with max-margin guard selects raw no-edit with reason
    `candidate_positive_delta_margin_guard`
  - seeds `14..15` with both guards select raw no-edit and are safe
  - final mixed 0..15 guarded summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_guarded_mixed_selection_compare_seed0_15.json`
  - final mixed 0..15 cap1 guarded metrics:
    local selector `2/16`, mean selected delta `+0.000427246`, candidate
    branch mean `-0.000854492`, harmful selected seed count `0`, harmful
    candidate seed count `4`
  - seed `11` versus seed `13` analysis showed that seed `11` is a real
    held-out-positive margin-`1.5` case, while seed `13` is a sparse
    margin-`1.75` false positive correctly blocked by the max-margin guard
  - `DEFAULT_SELECTOR_CANDIDATE_FIRST_POSITIVE_MAX_HARMED` is now `2`
  - seed `11` cap2 rerun:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed11_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
  - current mixed 0..15 cap2 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_15.json`
  - current cap2 mixed metrics:
    local selector `3/16`, selected local seeds `0,2,11`, mean selected delta
    `+0.000671387`, candidate branch mean `-0.000854492`, harmful selected
    seed count `0`, harmful candidate seed count `4`
  - first out-of-sample cap2 seed:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed16_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
  - seed `16` is raw no-edit with validation candidate delta `0.000000000`
    and held-out selected delta `0.000000000`
  - current mixed 0..16 cap2 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_16.json`
  - current cap2 0..16 metrics:
    local selector `3/17`, mean selected delta `+0.000631893`, candidate
    branch mean `-0.000804228`, harmful selected seed count `0`, harmful
    candidate seed count `4`
  - seed `17` hit the out-of-sample stop condition:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed17_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_pairwise_seq/experiment_summary.json`
  - seed `17` selected local selector with validation delta `+0.009746037`,
    margin `1.25`, nonzero `5`, and validation improved/harmed `4/1`
  - seed `17` held-out selected delta was `-0.004882812`
    (`8/13` improved/harmed)
  - current mixed 0..17 cap2 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_17.json`
  - current cap2 0..17 metrics:
    local selector `4/18`, mean selected delta `+0.000325521`, candidate
    branch mean `-0.001030816`, harmful selected seed count `1`, harmful
    candidate seed count `5`
  - seed `0`, `2`, `8`, `11`, `13`, and `17` margin-profile artifact:
    `artifacts/eval/nn/sedp_d7_margin_profile_seed0_2_8_11_13_17.json`
  - seed `17` differs from selected-positive seeds `0`, `2`, and `11` because
    its validation-positive band continues to a higher emit margin; the
    selected-positive seeds have validation positivity isolated at their
    selected margin
  - plateau-guard post-hoc simulation:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_posthoc_seed0_17.json`
  - plateau-guard hypothesis over seeds `0..17` blocks seed `17`, gives local
    selector `3/18`, mean selected delta `+0.000596788`, and harmful selected
    seed count `0`
  - d5 seed `3` compatibility:
    `artifacts/eval/nn/sedp_d5_margin_profile_seed3.json`
  - d5 seed `3` is preserved by the plateau hypothesis; it has no higher
    positive aggregate validation margin and has held-out stage_c delta
    `+0.023437500`
  - d7 seed `18` and seed `19` were run as additional checks:
    - seed `18`: raw no-edit, validation candidate delta `+0.003250404`,
      held-out candidate delta `-0.001953125`
    - seed `19`: raw no-edit, validation and held-out candidate deltas `0`
  - current cap2 mixed 0..19 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_mixed_selection_compare_seed0_19.json`
  - cap2 0..19 metrics:
    local selector `4/20`, mean selected delta `+0.000292969`, candidate
    branch mean `-0.001025391`, harmful selected seed count `1`, harmful
    candidate seed count `6`
  - plateau-guard post-hoc 0..19:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_posthoc_seed0_19.json`
  - plateau-guard 0..19 metrics:
    local selector `3/20`, mean selected delta `+0.000537109`, harmful
    selected seed count `0`
  - optional integrated plateau guard is now implemented:
    `--selector-candidate-first-positive-plateau-guard`
  - integrated seed `17` plateau-guard artifact:
    `artifacts/eval/nn/sedp_d7_router1k_benefitharm_patchhead_seed17_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_pairwise_seq/experiment_summary.json`
  - integrated seed `17` selects `raw_no_edit` with reason
    `candidate_positive_delta_plateau_guard`; the attached selector margin
    profile records the higher positive margin `1.5`
  - integrated mixed 0..19 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_19.json`
  - integrated plateau-guard 0..19 metrics:
    local selector `3/20`, mean selected delta `+0.000537109`, candidate
    branch mean `-0.001025391`, harmful selected seed count `0`, harmful
    candidate seed count `6`
  - integrated compatibility sentinels:
    - d7 seed `11` stays local selector at margin `1.5` with held-out delta
      `+0.003906250`
    - d5 seed `3` stays local selector at margin `0.5` with held-out delta
      `+0.023437500`
  - d7 seed `20` and seed `21` were then extended with plateau guard enabled;
    both select `raw_no_edit` and have held-out selected delta `0`
  - integrated mixed 0..21 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_21.json`
  - integrated plateau-guard 0..21 metrics:
    local selector `3/22`, mean selected delta `+0.000488281`, candidate
    branch mean `-0.000932173`, harmful selected seed count `0`, harmful
    candidate seed count `6`
  - d7 seed `22` and seed `23` were then extended with plateau guard enabled;
    both select `raw_no_edit` and have held-out selected delta `0`
  - integrated mixed 0..23 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_23.json`
  - integrated plateau-guard 0..23 metrics:
    local selector `3/24`, selected local seeds `0,2,11`, mean selected delta
    `+0.000447591`, candidate branch mean `-0.000854492`, harmful selected
    seed count `0`, harmful candidate seed count `6`
  - d7 seed `24` and seed `25` were then extended with plateau guard enabled;
    both select `raw_no_edit` and have held-out selected delta `0`
  - integrated mixed 0..25 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_25.json`
  - integrated plateau-guard 0..25 metrics:
    local selector `3/26`, selected local seeds `0,2,11`, mean selected delta
    `+0.000413161`, candidate branch mean `-0.000788762`, harmful selected
    seed count `0`, harmful candidate seed count `6`
  - d7 seed `26` and seed `27` were then extended with plateau guard enabled:
    - seed `26` selects `raw_no_edit`; candidate branch is harmful with
      held-out delta `-0.011718750` and improved/harmed `11/23`
    - seed `27` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..27 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_27.json`
  - integrated plateau-guard 0..27 metrics:
    local selector `3/28`, selected local seeds `0,2,11`, mean selected delta
    `+0.000383650`, candidate branch mean `-0.001150949`, harmful selected
    seed count `0`, harmful candidate seed count `7`
  - d7 seed `28` and seed `29` were then extended with plateau guard enabled:
    - seed `28` selects `raw_no_edit`; candidate branch is held-out positive
      by one shot (`+0.000976562`)
    - seed `29` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..29 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_29.json`
  - integrated plateau-guard 0..29 metrics:
    local selector `3/30`, selected local seeds `0,2,11`, mean selected delta
    `+0.000358073`, candidate branch mean `-0.001041667`, harmful selected
    seed count `0`, harmful candidate seed count `7`
  - d7 seed `30` and seed `31` were then extended with plateau guard enabled;
    both select `raw_no_edit` and have held-out selected/candidate deltas `0`
  - integrated mixed 0..31 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_31.json`
  - integrated plateau-guard 0..31 metrics:
    local selector `3/32`, selected local seeds `0,2,11`, mean selected delta
    `+0.000335693`, candidate branch mean `-0.000976562`, harmful selected
    seed count `0`, harmful candidate seed count `7`
  - d7 seed `32` and seed `33` were then extended with plateau guard enabled:
    - seed `32` selects `raw_no_edit` by `candidate_positive_delta_harm_guard`;
      candidate branch is harmful with held-out delta `-0.010742188`
    - seed `33` selects `raw_no_edit` by `candidate_positive_delta_harm_guard`;
      candidate branch is harmful with held-out delta `-0.016601562`
  - integrated mixed 0..33 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_33.json`
  - integrated plateau-guard 0..33 metrics:
    local selector `3/34`, selected local seeds `0,2,11`, mean selected delta
    `+0.000315947`, candidate branch mean `-0.001723346`, harmful selected
    seed count `0`, harmful candidate seed count `9`
  - d7 seed `34` and seed `35` were then extended with plateau guard enabled:
    - seed `34` selects `raw_no_edit`; candidate branch is harmful with
      held-out delta `-0.002929688`
    - seed `35` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..35 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_35.json`
  - integrated plateau-guard 0..35 metrics:
    local selector `3/36`, selected local seeds `0,2,11`, mean selected delta
    `+0.000298394`, candidate branch mean `-0.001708984`, harmful selected
    seed count `0`, harmful candidate seed count `10`
  - d7 seed `36` and seed `37` were then extended with plateau guard enabled:
    - seed `36` selects `raw_no_edit`; candidate branch is harmful with
      held-out delta `-0.001953125`
    - seed `37` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..37 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_37.json`
  - integrated plateau-guard 0..37 metrics:
    local selector `3/38`, selected local seeds `0,2,11`, mean selected delta
    `+0.000282689`, candidate branch mean `-0.001670436`, harmful selected
    seed count `0`, harmful candidate seed count `11`
  - d7 seed `38` and seed `39` were then extended with plateau guard enabled:
    - seed `38` selects `raw_no_edit`; candidate branch is harmful with
      held-out delta `-0.001953125`
    - seed `39` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..39 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_39.json`
  - integrated plateau-guard 0..39 metrics:
    local selector `3/40`, selected local seeds `0,2,11`, mean selected delta
    `+0.000268555`, candidate branch mean `-0.001635742`, harmful selected
    seed count `0`, harmful candidate seed count `12`
  - d7 seed `40` and seed `41` were then extended with plateau guard enabled:
    - seed `40` selects `raw_no_edit`; candidate branch delta `0`
    - seed `41` selects `raw_no_edit`; candidate branch is harmful with
      held-out delta `-0.000976562`
  - integrated mixed 0..41 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_41.json`
  - integrated plateau-guard 0..41 metrics:
    local selector `3/42`, selected local seeds `0,2,11`, mean selected delta
    `+0.000255766`, candidate branch mean `-0.001581101`, harmful selected
    seed count `0`, harmful candidate seed count `13`
  - d7 seed `42` and seed `43` were then extended with plateau guard enabled:
    - seed `42` selects `raw_no_edit`; candidate branch delta `0`
    - seed `43` selects `raw_no_edit`; candidate branch delta `+0.000976562`
  - integrated mixed 0..43 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_43.json`
  - integrated plateau-guard 0..43 metrics:
    local selector `3/44`, selected local seeds `0,2,11`, mean selected delta
    `+0.000244141`, candidate branch mean `-0.001487038`, harmful selected
    seed count `0`, harmful candidate seed count `13`
  - d7 seed `44` and seed `45` were then extended with plateau guard enabled:
    - seed `44` selects `raw_no_edit`; candidate branch delta `0`
    - seed `45` selects `raw_no_edit`; candidate branch delta `+0.000976562`
  - integrated mixed 0..45 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_45.json`
  - integrated plateau-guard 0..45 metrics:
    local selector `3/46`, selected local seeds `0,2,11`, mean selected delta
    `+0.000233526`, candidate branch mean `-0.001401155`, harmful selected
    seed count `0`, harmful candidate seed count `13`
  - d7 seed `46` and seed `47` were then extended with plateau guard enabled:
    - seed `46` selects `raw_no_edit`; candidate branch delta `0`
    - seed `47` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..47 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_47.json`
  - integrated plateau-guard 0..47 metrics:
    local selector `3/48`, selected local seeds `0,2,11`, mean selected delta
    `+0.000223796`, candidate branch mean `-0.001342773`, harmful selected
    seed count `0`, harmful candidate seed count `13`
  - d7 seed `48` and seed `49` were then extended with plateau guard enabled:
    - seed `48` selects `raw_no_edit`; candidate branch delta `0`
    - seed `49` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..49 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_49.json`
  - integrated plateau-guard 0..49 metrics:
    local selector `3/50`, selected local seeds `0,2,11`, mean selected delta
    `+0.000214844`, candidate branch mean `-0.001289063`, harmful selected
    seed count `0`, harmful candidate seed count `13`
  - d7 seed `50` and seed `51` were then extended with plateau guard enabled:
    - seed `50` selects `raw_no_edit`; candidate branch delta `0`
    - seed `51` selects `raw_no_edit`; candidate branch delta `0`
  - integrated mixed 0..51 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_51.json`
  - integrated plateau-guard 0..51 metrics:
    local selector `3/52`, selected local seeds `0,2,11`, mean selected delta
    `+0.000206581`, candidate branch mean `-0.001239483`, harmful selected
    seed count `0`, harmful candidate seed count `13`
  - d7 seed `52` and seed `53` were then extended with plateau guard enabled:
    - seed `52` selects `raw_no_edit`; candidate branch delta `0`
    - seed `53` selects `raw_no_edit` by harm guard; candidate branch is
      harmful with held-out delta `-0.010742188`
  - integrated mixed 0..53 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_53.json`
  - integrated plateau-guard 0..53 metrics:
    local selector `3/54`, selected local seeds `0,2,11`, mean selected delta
    `+0.000198929`, candidate branch mean `-0.001392506`, harmful selected
    seed count `0`, harmful candidate seed count `14`
  - d7 seed `54` was then run with plateau guard enabled and failed selected
    safety:
    - seed `54` selects `local_motif_selector`; validation delta
      `+0.006508300`, held-out selected delta `-0.006835938`
    - candidate branch also has held-out delta `-0.006835938`
    - at selected margin `1.25`, validation support is stage_a-only:
      `stage_a_si1000` delta `+0.012987013`, nonzero `2`; `stage_b_local`
      delta `0`, nonzero `0`
    - seed `55` was skipped because the stop-on-harm rule fired
  - integrated mixed 0..54 failed summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_plateauguard_mixed_selection_compare_seed0_54_failed.json`
  - integrated plateau-guard 0..54 failed metrics:
    local selector `4/55`, selected local seeds `0,2,11,54`, mean selected
    delta `+0.000071023`, candidate branch mean `-0.001491477`, harmful
    selected seed count `1`, harmful candidate seed count `15`
  - added positive support calibration option:
    `--selector-candidate-first-positive-min-nonzero`
    - default `0` preserves existing behavior
    - post-hoc min-nonzero `5` over seeds `0,2,11,54` selects only `2,11`,
      blocks seed `54`, and sacrifices weak seed `0`
    - post-hoc 0..54 with min-nonzero `5`: mean selected delta
      `+0.000159801`, harmful selected count `0`
  - actual d7 seed `54` support-guard sentinel:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
    - selected `raw_no_edit`
    - reason `candidate_positive_delta_support_guard`
    - held-out selected delta `0`, candidate delta `-0.006835938`
  - actual support-guard true-positive sentinels:
    - seed `11` with min-nonzero `5` remains `local_motif_selector`; held-out
      selected delta `+0.003906250`
    - seed `2` with min-nonzero `5` remains `local_motif_selector`; held-out
      selected delta `+0.004882812`
    - comparison artifact:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
  - support-guard mixed 0..54 sentinel summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_54_sentinel.json`
    - local selector `2/55`, selected seeds `2,11`
    - mean selected delta `+0.000159801`, harmful selected count `0`
  - d7 seed `55` was run with support guard enabled:
    - selected `raw_no_edit`, reason `default_no_edit`
    - held-out selected delta `0`, candidate delta `-0.004882812`
    - comparison artifact:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed55.json`
  - support-guard mixed 0..55 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_55.json`
    - local selector `2/56`, selected seeds `2,11`
    - mean selected delta `+0.000156948`, harmful selected count `0`
  - d7 seed `56` was run with support guard enabled:
    - selected `raw_no_edit`, reason `default_no_edit`
    - held-out selected delta `0`, candidate delta `-0.000976562`
    - comparison artifact:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed56.json`
  - support-guard mixed 0..56 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_56.json`
    - local selector `2/57`, selected seeds `2,11`
    - mean selected delta `+0.000154194`, candidate-branch mean
      `-0.001541941`, harmful selected count `0`, harmful candidate count `17`
  - d7 seed `57` was run with support guard enabled:
    - selected `raw_no_edit`, reason `default_no_edit`
    - held-out selected delta `0`, candidate delta `0`
    - comparison artifact:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed57.json`
  - support-guard mixed 0..57 summary:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json`
    - local selector `2/58`, selected seeds `2,11`
    - mean selected delta `+0.000151536`, candidate-branch mean
      `-0.001515356`, harmful selected count `0`, harmful candidate count `17`
  - d7 support-guard candidate-oracle analysis:
    `artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json`
    - all 58 checked seeds have positive candidate-oracle headroom
    - mean candidate-oracle delta `+0.096679688`
    - actual candidate mean delta `-0.001515356`
    - candidate classes: `6` positive, `35` neutral, `17` harmful
  - d7 true/false selected-shot diagnostic:
    `artifacts/eval/nn/sedp_d7_support_guard_true_false_selection_diagnostic_seed2_11_54_stagec.json`
    - seed `54` has high oracle headroom, but selected candidates harm more
      often than they help (`6/13` improved/harmed at margin `1.25`)
    - this points to selector ranking/generalization as the d7 bottleneck,
      not candidate-set coverage
  - d7 oracle/harm ranking diagnostic:
    `artifacts/eval/nn/sedp_d7_support_guard_oracle_harm_ranking_diagnostic_seed2_11_54_55_stagec.json`
    - added oracle-positive rank/gap and negative-target over-margin counts to
      `tools/diagnose_predecoder_selection.py`
    - seed `2` margin `1.25`: oracle above margin `6`, negative above margin
      `1`, held-out delta `+0.004882812`
    - seed `11` margin `1.5`: oracle above margin `10`, negative above margin
      `6`, held-out delta `+0.003906250`
    - seed `54` margin `1.25`: oracle above margin `6`, negative above margin
      `13`, held-out delta `-0.006835938`
    - seed `55` margin `1.75`: oracle above margin `8`, negative above margin
      `13`, held-out candidate delta `-0.004882812`
  - implemented optional hard-negative identity-margin selector loss:
    - `--selector-negative-identity-margin-loss-weight`
    - `--selector-negative-identity-margin`
    - defaults preserve old behavior
  - hard-negative sentinel with weight `1.0`, margin `1.5`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_negidmargin10_m15_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
    - seed `54`: candidate delta improves from `-0.006835938` to
      `-0.001953125`, but remains harmful
    - seed `2`: true-positive is lost; candidate delta becomes
      `-0.003906250`
    - verdict: reject `1.0/1.5` as too strong
  - hard-negative seed54 ranking diagnostic:
    `artifacts/eval/nn/sedp_d7_negidmargin10_m15_oracle_harm_ranking_diagnostic_seed54_stagec.json`
    - negative-over-identity count falls from `110` to `8`, but useful
      oracle-positive margins are also suppressed
  - weak hard-negative sentinel with weight `0.25`, margin `1.0`:
    `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_negidmargin025_m10_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
    - seed `54` selects `local_motif_selector` and is harmful on held-out
      `stage_c_corr`
    - held-out selected/candidate delta `-0.001953125`
    - validation margin `1.0`, nonzero `5`, improved/harmed `4/1`, so the
      support guard allows it
    - verdict: reject `0.25/1.0` as unsafe
  - weak hard-negative seed54 ranking diagnostic:
    `artifacts/eval/nn/sedp_d7_negidmargin025_m10_oracle_harm_ranking_diagnostic_seed54_stagec.json`
    - at margin `1.0`, oracle-positive above margin `6` but negative-target
      above margin `9`
  - conclusion: d7 selected-mode cap2 plus plateau guard is not safe enough for
    a final claim; it removes the seed `17` selected harm and preserves known
    d7/d5 true positives, but seed `54` is a new selected false positive, so
    the support guard should replace the old extension recipe; after the oracle
    analysis and ranking diagnostic, hard-negative identity-margin alone should
    be considered rejected: `1.0/1.5` destroys seed `2`, while `0.25/1.0` lets
    seed `54` through. The next step should be a positive-vs-negative ranking
    redesign, not another identity-margin-only run.
  - validation ranking-guard check:
    `artifacts/eval/nn/sedp_d7_support_guard_validation_ranking_guard_summary_seed2_11_54_55.json`
    - corrected split handling matters: validation diagnostics for this
      multi-family recipe must use `stage_a` split seed `seed` and `stage_b`
      split seed `seed + 1`; default split seed `0` is not the training
      validation split for most seeds
    - the simple adoption statistic "block when validation negative-target
      above-margin count exceeds oracle-positive above-margin count" is
      rejected
    - it preserves seed `2` and seed `11`, but seed `54` still passes the
      statistic at margin `1.25` with validation oracle/negative above-margin
      counts `2/0` despite held-out candidate delta `-0.006835938`
    - weak hard-negative seed `54` also passes the same statistic at margin
      `1.0` with combined validation oracle/negative above-margin counts `3/1`
      despite held-out delta `-0.001953125`
    - next d7 work should use a genuine positive-vs-negative ranking redesign
      or a cross-split/stage-generalization diagnostic stronger than this
      validation negative-excess guard
  - hard positive-vs-negative ranking sentinel:
    - implemented optional selector loss flags
      `--selector-positive-negative-hard-loss-weight` and
      `--selector-positive-negative-hard-margin`, both defaulting to `0.0`
    - sentinel artifact:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
    - weight `1.0`, margin `0.5` preserves seed `2` but reduces its held-out
      delta from `+0.004882812` to `+0.003906250`
    - seed `11` candidate branch improves to `+0.004882812`, but selected
      adoption falls to `raw_no_edit`, so selected delta becomes `0`
    - seed `54` candidate harm improves from `-0.006835938` to
      `-0.003906250`, but remains harmful
    - verdict: this is a partial candidate-branch improvement, not a selected
      recipe under the original adoption thresholds
    - plateau-aware adoption simulator now supports `--positive-plateau-guard`
    - with `positive_delta=0.003`, `positive_min_nonzero=1`, and plateau
      guard, the `1.0/0.5` checkpoint family selects seed `2` and seed `11`,
      blocks seed `54`, and reaches mean selected delta `+0.002929688`,
      matching the old support-guard sentinel while improving mean candidate
      delta over the same seeds to `+0.001627604`
    - weaker hard-ranking `0.5/0.5` was tested and rejected:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard05_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54.json`
      leaves all selected paths at no-edit under the original support guard,
      and calibrated adoption recovers only seed `11` while seed `2` candidate
      becomes harmful
    - this made `1.0/0.5` plus calibrated adoption worth a small extension
      check, but not yet a broad sweep
    - extension of `1.0/0.5` to seeds `55,56,57` failed to improve the
      sentinel set:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_posneghard10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_11_54_57.json`
    - calibrated adoption over seeds `2,11,54,55,56,57`:
      `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_seed2_11_54_57.json`
      keeps selected safety and matches old selected mean `+0.001464844`,
      but candidate mean worsens from old support-guard `-0.000651042` to
      `-0.002278646`
    - seed `55` and seed `57` are the new blockers: candidate deltas
      `-0.006835938` and `-0.011718750`, both blocked by harm guard
    - diagnostics:
      `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed55_stagec.json`
      and
      `artifacts/eval/nn/sedp_d7_posneghard10_m05_oracle_harm_ranking_diagnostic_seed57_stagec.json`
    - verdict: broad extension of `1.0/0.5` is rejected; d7 needs a different
      selector objective, not another scalar hard-ranking/adoption tweak
  - simple family-level stage-consistency adoption check:
    - `tools/simulate_predecoder_adoption_policy.py` now records validation
      family-level candidate deltas/nonzero/improved/harmed counts and
      supports `--positive-family-min-delta`, `--positive-min-family-count`,
      and `--positive-max-family-harmed`
    - all-family nonnegative validation guard on the `1.0/0.5` calibrated
      adoption artifact blocks true-positive seed `2` because its validation
      split is mixed (`stage_a=-0.006493506`,
      `stage_b=+0.025974026`), reducing mean policy delta to `+0.000813802`
    - family harmed-cap `2` on the same six-seed sentinel adds no
      discrimination: selected behavior remains the same as calibrated
      adoption, with mean policy delta `+0.001464844`
    - the same all-family nonnegative guard on the original support-guard
      recipe also blocks seed `2` and reduces mean policy delta to
      `+0.000651042`
    - artifacts:
      `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_familymin0_count2_seed2_11_54_57.json`,
      `artifacts/eval/nn/sedp_d7_posneghard10_m05_adoption_sim_posdelta003_posminnz1_plateau_familymaxharm2_seed2_11_54_57.json`,
      and
      `artifacts/eval/nn/sedp_d7_support_adoption_sim_posdelta003_posminnz1_plateau_familymin0_count2_seed2_11_54_57.json`
    - verdict: simple family-level adoption guards are rejected; if d7 is
      continued, stage consistency needs to be learned or diagnosed inside the
      selector objective rather than bolted on as an all-family validation
      threshold
  - cross-family hard positive-vs-negative selector objective:
    - implemented default-off flags in `decoders/syndrome_edit_predecoder.py`:
      `--selector-cross-family-positive-negative-loss-weight` and
      `--selector-cross-family-positive-negative-margin`
    - design: compare a group's best positive nonzero candidate logit against
      a hard negative nonzero candidate sampled from a different train family
    - seed `54` weak sentinel `0.25/0.5`:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_crossfam025_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
      leaves the candidate branch unchanged at `-0.006835938`
      (`6/13` improved/harmed)
    - seed `54` strong sentinel `1.0/0.5`:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_crossfam10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json`
      worsens the candidate branch to `-0.009765625` (`8/18`
      improved/harmed)
    - verdict: this simple cross-family hard-negative form is rejected for now;
      it did not pass the false-positive seed54 gate, so there is no rational
      basis to expand it to true-positive seeds
  - consolidation:
    - new tool:
      `tools/build_predecoder_consolidation_summary.py`
    - generated artifact:
      `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`
    - readable summary:
      `PREDECODER_CONSOLIDATED_EVIDENCE.md`
    - final table document:
      `PREDECODER_FINAL_RESULT_TABLES.md`
    - d3/d5 structure reference:
      `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
    - d3/d5 robustness follow-up:
      `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
      and `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json`
      confirm held-out positive/neutral/harmful seed counts of d3 `4/0/0`
      and d5 `2/2/0`
    - noise-family analysis:
      `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
      and `artifacts/eval/nn/sedp_noise_family_analysis_summary.json`
      summarize `stage_a_si1000`, `stage_b_local`, and held-out
      `stage_c_corr`; d7 is included as validation-to-heldout mismatch
      contrast
    - d7 targeted bottleneck analysis:
      `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`
      and
      `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
    - d7 adoption-grid diagnostic:
      `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json`
      checks `183040` simple policies and finds `0` passing
      preserve/recover/block policies
    - d7 candidate-compatibility pairwise top-k sentinel:
      `artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json`
      blocks seed54 but destroys seed2 with candidate delta
      `-0.136718750`, so this recipe is rejected
    - remaining-work checklist:
      `PREDECODER_REMAINING_WORK.md`
      freezes the remaining path as final table/method cleanup, d7 limitation
      writeup, optional sentinel-gated d7 objective only, and reproducibility
      packaging
    - current selected held-out `stage_c_corr` deltas:
      d3 `+0.007568359`, d5 `+0.011230469`, d7 support-guard
      `+0.000151536`
    - current claim boundary is d3/d5 positive selected-mode recovery plus d7
      as a controlled selector-ranking/generalization limitation

## Keep

- `config.py`
- `logical_targets.py`
- `logical_frame.py`
- `logical_bell.py`
- `noise_si1000.py`
- `geometry/rotated_rect.py`
- `decoders/baseline_rectcnn.py`
- `tools/build_dual_axis_manifest.py`
- `tools/audit_logical_frame_support.py`
- `tools/run_dual_axis_experiment.py`
- `tools/run_dual_axis_pymatching.py`
- `tools/evaluate_hybrid_fallback.py`
- `tools/evaluate_learned_hybrid_router.py`
- `tools/build_pymatching_edit_targets.py`
- `artifacts/`

These parts already support the new direction:
geometry-aware neural decoders, controlled experiment setup, scaffold-level logical-frame auditing,
prototype per-shot logical-class supervision, multi-noise evaluation, and confidence-aware hybrid fallback evaluation.

## Modify

- `circuits.py`
- `noise_willowcore.py`
- `sample_dataset.py`
- `decoders/baseline_pymatching.py`
- `decoders/research_noise_aware_3d.py`
- `decoders/factorized_logical_frame_decoder.py`
- `decoders/multiscale_factorized_decoder.py`
- `decoders/syndrome_edit_predecoder.py`

These parts remain on the mainline, but they still need structural work:

- move from single-basis binary supervision to logical-axis supervision and later logical-class targets
- support stronger evaluation against logical failure, not just binary flip labels
- add more realistic and more diverse noise families
- extend the first factorized logical-frame model from its current tempered-imbalance and temperature-calibrated form toward stronger fallback and recovery-level evaluation
- turn the first pre-decoder implementation into a useful system-level decoder instead of the current over-edit / no-edit pilot regimes

## Legacy

- `legacy_archive/decoders/baseline_nn.py`
- `legacy_archive/decoders/baseline_tracknn.py`
- `legacy_archive/decoders/baseline_trackformer.py`
- `legacy_archive/decoders/track_common.py`
- `legacy_archive/baseline_trackformer_eventcentric_v3_fast.py`

These remain in the repository only for historical comparison and ablation.
They are not part of the rebuilt decoder mainline.

## Immediate Mainline

The active mainline now looks like this:

1. `tools/audit_logical_frame_support.py` verifies whether the raw single-basis scaffold can support same-shot logical X/Z supervision.
2. `logical_bell.py` adds a Bell-pair readout path that turns the current z-basis scaffold into a true per-shot logical_class4 label circuit.
3. `sample_dataset.py` builds geometry-aware datasets from either the legacy single-basis target path or the new Bell-pair class4 path.
4. `tools/build_dual_axis_manifest.py` pairs basis-x and basis-z datasets when the experiment still uses axis-wise supervision.
5. `tools/run_dual_axis_experiment.py` runs aligned axis-wise experiments.
6. `tools/run_dual_axis_pymatching.py` runs the aligned classical baseline.
7. `decoders/research_noise_aware_3d.py` remains the noise-aware baseline backend.
8. `decoders/factorized_logical_frame_decoder.py` is the first new mainline logical-class decoder with factorized X/Z heads, optional auxiliary axis supervision, optional focal-style class4 loss, optional hierarchical non-identity auxiliary supervision, tempered imbalance handling, and guardrailed post-hoc temperature calibration.
9. `decoders/multiscale_factorized_decoder.py` is the first multi-scale Dense3D successor candidate (`M3D-FLFD`) and reuses the existing class4 train/eval stack while replacing the shallow FLFD trunk with a multi-resolution encoder.
10. `tools/evaluate_hybrid_fallback.py` evaluates calibrated FLFD confidence thresholds against PyMatching fallback on the same class4 manifests.
11. `tools/evaluate_learned_hybrid_router.py` trains a frozen-feature logistic router over FLFD and PyMatching outputs, with both correctness-target and prefer-neural-target modes and with explicit metadata/noise-context features, then evaluates learned hybrid routing without changing the FLFD backbone.
12. `tools/build_pymatching_edit_targets.py` builds the first derived target layer for the pre-decoder branch by searching for small local detector-bit edits that convert wrong PyMatching shots into correct shots.
13. `decoders/syndrome_edit_predecoder.py` is the first SEDP-v1 implementation and now includes both the original 3-D edit-mask plus `needs_edit` heads and a first candidate-edit selector follow-up on top of them before final PyMatching decoding.
14. `decoders/baseline_pymatching.py` remains the classical comparison point.

## Baseline Completion Boundary

As of 2026-04-24, the current constructed class4 noise environment has
refreshed PyMatching baselines for both current 2k-shot manifests:

- `artifacts/datasets/dev_class4_2k/manifest.json`
- `artifacts/datasets/dev_class4_d5_2k/manifest.json`

The corresponding PyMatching artifacts are:

- `artifacts/eval/pymatching/d3_2k_class4_refresh.json`
- `artifacts/eval/pymatching/d5_2k_class4_refresh.json`

PyMatching class4 accuracy on the current noise families:

| dataset | ideal | stage_a_si1000 | stage_b_local | stage_c_corr |
| --- | ---: | ---: | ---: | ---: |
| d3/r3 2k | 1.000000000 | 0.937011719 | 0.917968750 | 0.925292969 |
| d5/r5 2k | 1.000000000 | 0.907226562 | 0.904296875 | 0.899902344 |

The d7 class4 scope was added on 2026-04-24:

- `artifacts/datasets/dev_class4_d7_2k/manifest.json`
- `artifacts/eval/pymatching/d7_2k_class4_refresh.json`

Updated PyMatching class4 accuracy:

| dataset | ideal | stage_a_si1000 | stage_b_local | stage_c_corr |
| --- | ---: | ---: | ---: | ---: |
| d3/r3 2k | 1.000000000 | 0.937011719 | 0.917968750 | 0.925292969 |
| d5/r5 2k | 1.000000000 | 0.907226562 | 0.904296875 | 0.899902344 |
| d7/r7 2k | 1.000000000 | 0.891113281 | 0.868652344 | 0.874511719 |

The main target paper is now treated as an output/evaluation-format anchor, not
an architectural constraint. `baseline_rectcnn.py` remains an optional
paper-style neural baseline, but exact structural matching to the target paper
is no longer required.

The current model-selection summary is:

- `MODEL_SELECTION_D3_D5_D7.md`

## Current Limitation

The project now has a first working per-shot `logical_class4` supervision path,
but it is not yet the end state:

- the raw single-basis scaffold still does not expose a joint same-shot logical X/Z frame on its own
- the current `logical_class4` path depends on an added ideal Bell-pair reference-qubit readout
- the active baseline decoders can now consume the new class4 targets, the first factorized class4 decoder implementation now exists, and class4-level hybrid fallback evaluation now exists, but recovery-oriented evaluation is still missing
- the current FLFD line now supports a tempered imbalance path, an optional focal class4 objective, an optional hierarchical non-identity auxiliary objective, and an experimental joint confidence-loss path; the tempered path is stable enough to keep the original smoke-level accuracy without the collapse reversal caused by naive strong balancing, but smoke runs show that focal loss, the first hierarchical non-identity attempt, and the first joint confidence head do not yet beat the baseline PyMatching policy on holdout families
- the new learned hybrid router can now avoid the trivial fallback-all solution on seen families while preserving seen-family accuracy, but smoke runs still leave holdout `stage_c_corr` at PyMatching-level accuracy instead of recovering the pure neural advantage there; switching the router target from plain correctness to direct prefer-neural labeling and adding explicit metadata/noise-context features have not yet changed that smoke-level outcome
- the first larger class4 rerun now shows that the earlier FLFD stagnation was primarily a data-regime problem, not yet clear evidence that the factorized architecture must be discarded: with `2048` shots per family and a smaller FLFD variant, the model stops collapsing to all-`I`, reaches mixed-test `macro_f1 ~= 0.414`, holdout `stage_c_corr macro_f1 ~= 0.449`, and achieves calibrated threshold-hybrid behavior that meaningfully interpolates between pure neural and fallback-all
- this means the current priority is not a full model reset; it is larger balanced-enough class4 data, smaller stable FLFD variants, and macro-F1/balanced-accuracy-driven evaluation before any architecture replacement claim is made
- however, the first `d5/r5` class4 rerun shows that this conclusion does not automatically scale with distance: even with `2048` shots per family and clear non-`I` support in train families, both a small and a somewhat larger FLFD still collapse badly compared with PyMatching, while threshold hybrid immediately selects fallback-all
- the current interpretation is therefore two-part: `d3` was mostly blocked by data regime, but `d5` exposes a real scaling limitation in the current FLFD line; the next architecture step should now be judged by whether it closes that distance-scaling gap, not by more smoke-level tweaks alone
- the first direct successor candidate, `decoders/multiscale_factorized_decoder.py`, is now implemented and has been run on both large `d3` and large `d5` class4 datasets; in its first comparisons it did not improve on the original FLFD, and a stronger `d5` configuration collapsed even harder
- the main target-paper format remains Jung/Ali/Ha, `Convolutional Neural Decoder for Surface Codes`, but the paper is now an output/evaluation-format anchor rather than a strict architecture target; `baseline_rectcnn.py` and the rectangular geometry layer remain a paper-style baseline, not the required final model structure
- d7 class4 2k has now been added to the current distance ladder, and PyMatching reaches `stage_a_si1000 0.891113281`, `stage_b_local 0.868652344`, and `stage_c_corr 0.874511719`
- the direct FLFD-small d7 rerun confirms the scaling problem: the same small FLFD setting that partially learned non-identity classes at d3 collapses at d5 and collapses to all-`X` behavior at d7, with `stage_c_corr` accuracy only `0.1953125`
- the d7 local-edit target builder preserves strong oracle headroom: on 1024-shot targets, local `k<=2` edits move `stage_a_si1000 0.889648438 -> 0.977539062`, `stage_b_local 0.868164062 -> 0.978515625`, and `stage_c_corr 0.873046875 -> 0.984375000`
- therefore the most promising next model family is a PyMatching-assist neural pre-decoder with explicit benefit/harm calibration over local edit candidates; exact RectCNN-style architecture matching is not the next priority
- the first benefit/harm selector implementation is now complete: `syndrome_edit_predecoder.py` supports `--selector-target-mode benefit_harm` and augments selector candidates with logical-transition features; this produces the first selected held-out d3 pre-decoder gain, `stage_c_corr 0.928710938 -> 0.939453125`, but d5/d7 still select `global_policy`, so the next bottleneck is distance-scaled selector calibration rather than candidate generation
- the first d5 distance-scaled selector-calibration follow-up is now also complete: `syndrome_edit_predecoder.py` supports `--selector-nonzero-bias-grid`, `--selector-harm-margin-loss-weight`, and `--selector-harm-margin`, and router labels for `baseline_failure` / `oracle_solvable` now use actual candidate correctness so they remain valid under benefit/harm target scores; real d5 router1k runs with nonzero bias, harm-margin, stronger hard-shot weighting, and corrected `oracle_solvable` routing still do not beat raw PyMatching on held-out `stage_c_corr`, and a post-hoc sweep confirms that forcing nonzero edits improves some shots but harms many more
- this narrows the next pre-decoder step: do not continue with scalar margin/bias/router-threshold tuning alone; the next useful change should be a target-class-aware logical-transition selector over the existing high-oracle local edit candidate pool
- the research topic is now fixed in `RESEARCH_PLAN_PREDECODER_MAIN.md` as "Transition-aware neural pre-decoding for surface-code logical-frame inference"; single-model decoding remains a secondary timeboxed branch, while NVIDIA Ising-style pre-decoding is related work rather than a copied design target
- the d3 reproducibility gate for the benefit/harm transition-feature selector is now complete: new seeds `1,2,3` all select `local_motif_selector` and improve held-out `stage_c_corr`, with mean delta `+0.010091146` over raw PyMatching
- the first target-transition-prior selector is now implemented in `decoders/syndrome_edit_predecoder.py` with `--selector-transition-prior-weight-grid` and transition-prior checkpoint save/load; the d5 router1k run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_transprior/experiment_summary.json` still selects `global_policy`, leaves `stage_c_corr 0.888671875 -> 0.888671875`, and forced margin-0 emission harms more shots than it helps (`50` improved, `59` harmed)
- the first hard transition-compatibility gate is also implemented with `--selector-transition-compat-top-k-grid`; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_compat_topk/experiment_summary.json` still leaves the candidate-selector branch at no-edit, and forced top-k checks show that narrow top-k suppresses all edits while top-k `8` still harms `stage_c_corr` (`0.888671875 -> 0.878906250`, `34` improved, `44` harmed)
- the first candidate-level BCE compatibility head is also implemented with `--selector-candidate-compat-threshold-grid`; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat/experiment_summary.json` still leaves the candidate-selector branch at no-edit, and forced threshold checks show that thresholds `0.1..0.9` do not remove the harmful selected edits because the true beneficial nonzero rate is only about `1-1.5%` while the BCE head predicts about `23%` positives
- the first group-balanced candidate compatibility head is also implemented with `--candidate-compat-objective group_balanced`; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_groupbal/experiment_summary.json` still leaves the candidate-selector branch at no-edit, and checkpoint diagnostics show the opposite calibration failure from flat BCE: true beneficial nonzero rate is about `1-1.7%`, but predicted positive rate is only about `0.1-0.2%`
- the first pairwise candidate compatibility ranking head is also implemented with `--candidate-compat-objective pairwise_rank` and `--selector-candidate-compat-top-k-grid`; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_candidatecompat_pairwise/experiment_summary.json` still leaves the candidate-selector branch at no-edit, and forced top-k checks show no change because the harmful candidates selected by the main selector are already high-ranked by the auxiliary compatibility head
- the first main-selector pairwise benefit/harm ranking term is also implemented with `--selector-benefit-harm-pairwise-loss-weight` and `--selector-benefit-harm-pairwise-margin`; d5 runs with weight `1` and `16` still select `global_policy` / no-edit, although one forced full-eval sweep from the weight-`1` checkpoint showed a narrow non-selected positive band at margin `1.5` (`stage_c_corr 0.888671875 -> 0.889648438`)
- the first candidate-representation ablation is now implemented: duplicate policy/motif candidates merge motif evidence into the candidate feature row; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifmerge_pairwise/experiment_summary.json` still selects no-edit, and forced low-margin emission still harms held-out `stage_c_corr` (`0.888671875 -> 0.881835938`)
- the first candidate-set restriction ablation is now implemented: `--selector-policy-candidate-mode none` disables raw threshold/top-k policy candidates while keeping identity and motif/local-motif candidates; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_motifonly_pairwise/experiment_summary.json` still leaves selected `stage_c_corr` unchanged (`0.888671875 -> 0.888671875`), but the motif-only candidate oracle remains very high (`0.999023438` on held-out `stage_c_corr`) with `33.0` candidates per shot
- the first geometry/placement candidate-feature ablation is now implemented: `--selector-candidate-geometry-features` appends normalized detector-coordinate mean/std/span summaries for candidate edit indices; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_geom_motifonly_pairwise/experiment_summary.json` still leaves selected `stage_c_corr` unchanged (`0.888671875 -> 0.888671875`) while oracle remains high (`0.999023438`)
- the first local motif pattern/anchor candidate-feature ablation is now implemented: `--selector-candidate-pattern-features` appends local-pattern-present, normalized pattern id, log pattern count, and normalized anchor coordinates; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patterngeom_motifonly_pairwise/experiment_summary.json` still leaves selected `stage_c_corr` unchanged (`0.888671875 -> 0.888671875`) while oracle remains high (`0.999023438`)
- the first anchor-local evidence candidate-feature ablation is now implemented: `--selector-candidate-local-evidence-features` appends selected-detector and radius-1 anchor-neighborhood event/probability summaries; the d5 run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_localevidence_patterngeom_motifonly_pairwise/experiment_summary.json` makes the candidate-selector branch emit sparse edits, improving `stage_b_local` by one shot (`0.904296875 -> 0.905273438`) but harming held-out `stage_c_corr` by one shot (`0.888671875 -> 0.887695312`), so selected mode remains global no-edit
- the first local-patch candidate-feature implementation is now wired as an opt-in path: `--selector-candidate-local-patch-features` appends a radius-1 `3x3x3` anchor patch with detector event and edit-probability channels to each candidate feature row; this has passed `py_compile`, the short d5 smoke run `artifacts/eval/nn/sedp_d5_smoke_localpatch/experiment_summary.json` completes, the modest run `artifacts/eval/nn/sedp_d5_modest_localpatch/experiment_summary.json` gives only a one-shot seen-family candidate-selector gain, and the full d5 router1k run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_localpatch_patterngeom_motifonly_pairwise/experiment_summary.json` again selects `global_policy` with no emitted selector edits
- the first true patch-head selector implementation is now wired: `--selector-patch-head` makes `CandidateEditSelector` extract the local-patch candidate feature slice, encode it through a small MLP, and concatenate the learned patch embedding with the non-patch candidate features and shot embedding; the smoke run `artifacts/eval/nn/sedp_d5_smoke_patchhead_v3/experiment_summary.json` completes and records `selector_patch_head=True`, the modest run `artifacts/eval/nn/sedp_d5_modest_patchhead/experiment_summary.json` is the first d5 patch-head selected win (`stage_c_corr 0.892578125 -> 0.898437500`), but the full d5 router1k run `artifacts/eval/nn/sedp_d5_router1k_benefitharm_patchhead_patterngeom_motifonly_pairwise/experiment_summary.json` still selects `global_policy`
- the first full d5 patch-head seed sweep is now complete: seed `1` and seed `3` select `local_motif_selector` and beat raw PyMatching on held-out `stage_c_corr` (`0.888671875 -> 0.895507813` and `0.888671875 -> 0.896484375`), seed `2` stays `global_policy` but its candidate branch still improves held-out `stage_c_corr` to `0.893554688`, and seed `0` stays no-edit; mean selected delta over seeds `0..3` is `+0.003662109`, while mean candidate-branch delta is `+0.004882813`
- this means the separate transition-prior head, hard top-k gate, flat BCE candidate-compatibility head, group-balanced BCE compatibility head, auxiliary pairwise compatibility head, direct main-selector pairwise term, motif-evidence merge, raw-policy candidate restriction, simple geometry summaries, pattern/anchor metadata, handcrafted local evidence, appended local-patch features, and the first patch-head selector are useful ablations; patch-head is now the first full d5 PyMatching-beating selected mechanism in some seeds, but the next pre-decoder change should stabilize selector calibration / selected-mode adoption across seeds rather than adding another feature branch
- this means the repository now has evidence against the simplest “make the direct dense 3-D decoder more multi-scale” move; the next best architecture direction is no longer another dense-trunk tweak, but a pre-decoder branch informed by later Ising-style materials or another qualitatively different decoder system
- the first concrete pre-decoder prerequisite now exists as `tools/build_pymatching_edit_targets.py`; bounded `k<=2` pilot runs on the larger `d3` and `d5` class4 manifests already show strong local-edit oracle headroom relative to raw PyMatching, which is enough evidence to justify training the first syndrome-edit pre-decoder model
- the first concrete pre-decoder model now also exists as `decoders/syndrome_edit_predecoder.py`; however, the first pilot training recipe is not yet good enough: under accuracy-first safe policy selection the `d3` pilot collapses to the identity no-edit policy, while the same recipe on `d5` still over-edits and harms baseline PyMatching
- the first hard-shot weighted-sampling follow-up is now also implemented inside `decoders/syndrome_edit_predecoder.py`; it stabilizes the branch away from catastrophic over-editing, but the safe selected policy still remains the identity no-edit policy on current pilots, so sampling alone is not enough to unlock the oracle headroom found by the edit-target builder
- the first hard-shot-only edit-supervision follow-up is now also implemented inside the same pre-decoder branch; it still fails to unlock the oracle headroom, which now points more strongly to an objective mismatch between detector-level BCE targets and the real system-level goal of improving final PyMatching
- the first decision-aware follow-up is now also implemented inside `decoders/syndrome_edit_predecoder.py` as a candidate-edit ranking / selection layer over the existing threshold/top-k policy grid; however, the current evidence is still only smoke-level: a tiny `d3 stage_a_si1000` run verifies that the selector plumbing, checkpointing, and evaluation paths work, but the validation guardrail still selects the baseline `global_policy` mode rather than proving a real improvement over the safe no-edit regime
- the first real `d3` / `d5` selector pilot reruns are now also complete, and they sharpen the same conclusion: even with the selector path enabled, validation still selects `global_policy`, final eval accuracy remains at raw PyMatching on every family, and the selector chooses zero edits in practice
- at the same time, the selector-candidate oracle remains above baseline on the nontrivial families, which means there is still some useful headroom inside the generated candidate pool; the failure is therefore no longer "missing selector plumbing" but the deeper fact that the current training path still does not make beneficial nonzero edits attractive to a safe system-level selector
- the first in-training decision-aware follow-up is now also complete: `decoders/syndrome_edit_predecoder.py` supports a first identity-vs-target pairwise ranking loss on solved hard shots, and real `d3` / `d5` pilot reruns with that loss still fall back to `global_policy`, still choose zero edits in practice, and still remain at raw PyMatching accuracy
- the first stronger selector-training follow-up is now also complete: the branch now supports a per-shot group-ranking selector objective over the generated candidate set with hard-shot upweighting, and real `d3` / `d5` pilot reruns with that objective still fall back to `global_policy`, still choose zero edits in practice, and still remain at raw PyMatching accuracy
- the first explicit edit-validity-structured follow-up is now also complete: the branch now supports a motif-vocabulary selector over observed hard-shot edit masks, and real `d3` / `d5` pilot reruns with that path still fall back to `global_policy`, still choose zero edits in practice, and still remain at raw PyMatching accuracy
- the first structured candidate-pool follow-up is now also complete: the branch now supports augmenting the selector candidate set with observed motif actions, and real `d3` / `d5` pilot reruns with that stronger pool still fall back to `global_policy`, still choose zero edits in practice, and still do not unlock holdout gains over baseline PyMatching
- the first explicit selector-side identity-vs-nonzero follow-up is now also complete: the branch now supports a per-shot selector margin loss that tries to rank the best validated nonzero candidate above identity when such a candidate exists, and real `d3` / `d5` pilot reruns with that loss still fall back to `global_policy`, still choose zero edits in practice, and still do not unlock holdout gains over baseline PyMatching
- the first action-path structured-action follow-up is now also complete: the branch now supports a motif-action competition loss directly on `edit_logits + needs_edit_logits`, and real `d3` / `d5` pilot reruns with that loss still do not unlock holdout gains, though they do appear to stabilize the previous `d5` over-editing failure back to baseline-level behavior
- the first action-motif inference follow-up is now also complete: the branch can now emit structured motif actions directly at inference with a validation-selected emit margin; this finally produces nonzero actions on `d3` and improves seen-family eval, but it slightly harms holdout `stage_c_corr`, while `d5` suppresses all action emission under the validation guardrail
- the first local/generalizable action follow-up is now also complete: the branch supports `selection_mode = local_motif`, builds relative `(dt, dr, dc)` edit patterns from hard-shot oracle masks, expands them over valid detector-coordinate anchors, and guardrails emission with both `local_motif_emit_margin` and `local_motif_min_bit_logit`
- real gated `d3` / `d5` local-motif reruns still select `global_policy`; `d3` local inference only improves seen-family `stage_b_local` by one shot and leaves holdout `stage_c_corr` unchanged, while `d5` remains unchanged under the selected local gate
- the first local-motif selector follow-up is now also complete: the branch supports `selection_mode = local_motif_selector`, adds top-k local placement candidates to the decision-aware candidate selector, labels those candidates by actual PyMatching correctness, and guardrails selector emission with validation-selected `selector_emit_margin`
- real local-motif selector pilots show that local candidate generation is not the current bottleneck: the local candidate oracle is effectively saturated on `d3` and remains very high on `d5`, but the learned selector is not calibrated; default settings stay identity-only, strong hard-shot settings over-emit and harm some families, and selector emit-margin guardrails either suppress edits or still fail to beat `global_policy`
- the first factorized hard-shot router follow-up is now also complete: the branch supports `selection_mode = local_motif_router`, trains a shot-level `HardShotRouter` from system-level local candidate labels, and only lets the local selector emit nonzero actions on routed shots
- real routed local-motif pilots still select `global_policy`; the router architecture and checkpoint/eval path are wired, but in the current 256-shot pilot regime route-positive labels are only about `5-10%` on d3 train-family validation splits, and the learned router collapses to all-route or no-route behavior rather than a calibrated hard-shot subset
- the first larger-target router rerun is now also complete: `predecoder_targets_d3_2k_router1k` and `predecoder_targets_d5_2k_router1k` process 1024 shots per family and preserve strong local-edit oracle headroom, but the learned `local_motif_router` still selects `global_policy`, routes no eval shots, and leaves `d3 stage_c_corr` at `0.9287109375` and `d5 stage_c_corr` at `0.888671875`
- the first router-supervision follow-up is now also complete: `syndrome_edit_predecoder.py` supports baseline-failure / oracle-solvability router pretraining and balanced route minibatches, and the `sedp_d3_router1k_router_pretrain_balanced` / `sedp_d5_router1k_router_pretrain_balanced` runs show that this changes router diagnostics but still selects `global_policy`, routes no eval shots, and leaves selected stage_c decoding unchanged
- one new signal did appear: in the d3 balanced/pretrained run, the action-motif eval path improves `stage_a_si1000 0.9287109375 -> 0.9443359375`, `stage_b_local 0.90625 -> 0.94140625`, and `stage_c_corr 0.9287109375 -> 0.931640625`; this is not a selected-router win, but it makes action-motif selected-mode evaluation the next most useful targeted check
- focused action-motif selected-mode reruns are now also complete: `sedp_d3_router1k_actionmotif_selected` selects a non-identity `global_policy` and gives a small `d3 stage_c_corr 0.9287109375 -> 0.9306640625` gain, while `sedp_d5_router1k_actionmotif_selected` remains identity/no-edit
- this means the next pre-decoder step is now not just "more candidate pool", not just "a stronger selector loss", not just "structured-action supervision alone", not just "static whole-mask action emission", not just "local motif placement generation by itself", not just "local placement candidate selector by itself", not just "factorization plumbing by itself", not just a moderate 256->1024 same-recipe target-size increase, and not just baseline-failure pretraining with balanced router minibatches, but checking whether the small d3 non-identity edit-policy gain is reproducible and then adding benefit/harm calibration if it is not; it should not be another rerun of the same post-hoc selector recipe, not another rerun of the same simple identity-vs-target margin loss, not more selector-only fitting on the same candidate pool, and not a return to sampling-only tuning
- the project still lacks a native Willow-style logical-frame circuit and a final class4 recovery decoder

The audit path still matters because it makes the base limitation explicit:

- the single-basis scaffold directly measures only one logical observable per shot
- the conjugate logical observable remains structurally random in that same experiment
- therefore the raw scaffold alone cannot honestly emit same-shot `logical_class4`
