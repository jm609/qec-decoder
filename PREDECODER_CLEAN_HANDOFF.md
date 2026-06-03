# Predecoder Clean Handoff

This is the compact handoff for continuing the project with limited context.
Read this before the longer `NEXT_SESSION_HANDOFF.md`.

## Latest State as of 2026-05-28

The research result is now fixed. The remaining work is manuscript/poster
finalization, Overleaf compile checking, and layout polish. Do not change the
submitted English title unless the user explicitly asks.

Submitted English title to keep:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Active manuscript files:

- `main.tex`: Korean polished draft with the submitted English title.
- `main_en.tex`: English polished draft after the latest English-style
  feedback pass.
- `main.bib`: shared bibliography for both drafts.

Current Overleaf packages:

- Korean package: `artifacts/overleaf_predecoder_package.zip`
- English package: `artifacts/overleaf_predecoder_package_en.zip`
- Compiler target: XeLaTeX.

Progress estimate:

| area | progress | note |
| --- | ---: | --- |
| research experiments/evidence | 100% | d3/d5 success and d7 limitation are fixed |
| Korean manuscript content | 90-95% | needs final Overleaf visual/log check |
| English manuscript content | 90-95% | latest English feedback applied |
| poster | 25-35% | narrative/figures/results are ready, final poster layout remains |
| overall thesis+poster deliverable | 85-90% | remaining work is production and formatting |

Latest validation commands already passed:

```powershell
python tools\check_main_tex_static.py
python tools\check_main_tex_static.py --main-tex main_en.tex --out artifacts\eval\nn\main_en_tex_static_check_summary.json
python -m py_compile tools\prepare_overleaf_package.py tools\check_main_tex_static.py
python tools\prepare_overleaf_package.py
python tools\prepare_overleaf_package.py --main-file main_en.tex --out-dir artifacts\overleaf_predecoder_package_en --zip artifacts\overleaf_predecoder_package_en.zip
```

Latest manuscript edits completed:

- Kept the submitted `Transition-Aware` title.
- Added a clear definition of `transition-aware`: it means the selector uses
  the downstream PyMatching decision before and after a local edit, not a
  separate temporal transition model.
- Removed `showkeys` from both LaTeX documents.
- Unified Korean manuscript terminology to use `surface code`.
- Added `logical_class4` terminology clarification: it is the four-class
  logical decision label, also called logical frame or logical decision in
  prose.
- Clarified that held-out `stage_c_corr` local-edit search is not used for
  training, motif selection, threshold selection, or seed-level adoption.
  It is only post hoc oracle analysis.
- Clarified dataset split wording: training local-edit targets use fixed
  1024-shot subsets from train/validation families; held-out 1024 shots are
  for final evaluation and oracle analysis only.
- Clarified the fixed validation split of 154 shots.
- English draft polish completed:
  - `this work` is used consistently instead of mixing `this thesis` and
    `this manuscript`.
  - Repeated defensive `not A, but B` phrasing was reduced.
  - selector feature/loss details were restored.
  - CI and p-values were rounded in paper style.
  - abstract/result wording was made more direct and less template-like.

Fixed research claims:

| distance | status | held-out selected gain vs raw PyMatching |
| --- | --- | ---: |
| d3 | robust positive selected-mode result | `+0.66` pp |
| d5 | conservative selected-mode improvement | `+0.56` pp |
| d7 | controlled limitation, not solved recovery | `+0.02` pp |

Interpretation to preserve:

- d3 is positive over checked seeds.
- d5 should not be oversold; the selected gain comes from seeds 2 and 3, while
  other seeds fall back to raw no-edit.
- d7 has recoverable local edits in oracle analysis, but the selector does not
  choose them reliably under held-out noise-family shift.
- The project should not add new experimental claims unless the user explicitly
  asks for a new experiment branch. For thesis/poster submission, keep the
  evidence fixed and focus on presentation quality.

Recommended next session start prompt:

```text
/C:/Users/82108/fp/PREDECODER_CLEAN_HANDOFF.md와 /C:/Users/82108/fp/NEXT_SESSION_HANDOFF.md의 2026-05-28 최신 섹션부터 읽어줘.
제목은 이미 제출했으므로 "Design and Evaluation of a Transition-Aware Neural Pre-Decoder for Surface-Code Quantum Error Correction"으로 유지해야 한다.
현재 main.tex/main_en.tex는 제출용 논문 초안이고, 다음 작업은 새 실험이 아니라 Overleaf 컴파일 로그/레이아웃 점검 또는 포스터 작성 준비다.
먼저 최신 zip과 정적 검사 상태를 확인하고, 필요한 후속 작업을 진행해줘.
```

## Research Topic

Submitted English title:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Technical topic:

> Transition-aware neural pre-decoding for surface-code logical-frame
> inference.

The current model is not a replacement for PyMatching. It is a neural
pre-decoder that proposes a local syndrome edit, then hands the edited or raw
syndrome to PyMatching for final logical-frame decoding.

## Current Decoder Structure

```text
36-channel syndrome/noise volume
  -> SyndromeEditPreDecoder 3D residual trunk
  -> local motif candidate set
  -> patch-head CandidateEditSelector
  -> selected local detector edit or raw no-edit fallback
  -> PyMatching
  -> logical_class4 prediction
```

The strongest successful recipe uses:

- `local_motif_selector`
- `benefit_harm` selector target mode
- patch-head selector
- candidate-first safety/adoption policy
- geometry, pattern, local-evidence, and local-patch candidate features
- pairwise benefit/harm loss
- selected-mode guardrails that fall back to raw no-edit when adoption is not
  justified

## Completed Work

The project has completed:

1. Dataset and target construction for d3, d5, and d7 router1k predecoder
   experiments.
2. Raw PyMatching baseline evaluation.
3. Neural predecoder implementation with local motif candidates and patch-head
   candidate selection.
4. Candidate-first selected-mode safety and calibration tooling.
5. d3 and d5 selected-mode success validation on held-out `stage_c_corr`.
6. d7 58-seed support-guard sweep and oracle-gap analysis.
7. d7 targeted selector-bottleneck analysis with preserve/recover/block
   sentinel sets.
8. Rejection of threshold-only d7 adoption tuning.
9. Rejection of the tested cross-family hard-negative d7 objective.
10. Rejection of d7 candidate-compatibility pairwise top-k because it blocks
    seed54 but destroys seed2.
11. Consolidated evidence documents and generated summary JSON.
12. d3/d5 robustness summary over seed `0..7`, confirming no harmful selected
    held-out seeds at either successful distance.
13. Noise-family analysis showing d3 positive behavior across validation and
    held-out families, d5 mixed validation behavior but non-harmful held-out
    behavior, and d7 validation-to-heldout mismatch.
14. Baseline comparison summary that separates the fair same-artifact
    PyMatching comparison from older direct neural context baselines
    (FLFD/M3D/RectCNN).
15. Ablation/failure-path synthesis explaining why standalone direct neural
    decoders, scalar d7 adoption tuning, cross-family hard-negative training,
    and candidate-compatibility top-k are not the final direction.
16. Final result table consistency check: `37` Markdown-vs-consolidated-JSON
    checks pass with `0` failures.
17. Canonical method description covering the 36-channel input volume, 3D
    residual trunk, local motif candidates, patch-head selector, selected-mode
    adoption, and PyMatching handoff.
18. D7 harmful-edit taxonomy showing that validation-positive candidate
    branches become held-out harmful in `13/22` cases, while selected mode
    blocks all `17/17` harmful candidate seeds.
19. Thesis core integration draft in `GRADUATION_THESIS_DRAFT.md`, covering
    method, setup, results, ablation, d7 limitation, and discussion.
20. Reproducibility package in `PREDECODER_REPRODUCIBILITY_PACKAGE.md`, with
    source files, canonical artifacts, regeneration commands, syntax checks,
    and the d7 sentinel gate.
21. Clean Korean thesis core draft in
    `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`, covering the abstract, chapter
    prose, result tables, d7 limitation analysis, conclusion, and figure/table
    captions.
22. Figure package in `PREDECODER_FIGURE_PACKAGE.md` with six generated SVG
    sources and six Overleaf-ready PNG files under
    `artifacts/figures/predecoder/`.
23. Hyperparameter sensitivity table in
    `PREDECODER_HYPERPARAMETER_SENSITIVITY.md`, showing d7 identity-margin
    weight `0.5` is the best checked sentinel compromise over seeds `0,2,5`.
24. D3/d5 paired statistics in `PREDECODER_D3_D5_PAIRED_STATISTICS.md`,
    showing d3 one-sided exact sign/sign-flip p-value `0.003906250` and d5
    one-sided value `0.250000000`, which keeps the d5 wording conservative.
25. Evaluation-report action plan in `RESEARCH_EVALUATION_ACTION_PLAN.md`,
    mapping the latest `8.7/10` report's required items to completed changes.
26. `main.tex` manuscript update with a d5 seed-level fallback table and a
    tightened Stage A/B/C scope. Stage D/E implementation details are not part
    of the main manuscript claim unless an appendix is explicitly requested.
27. Cross-paper context table added to `main.tex`, comparing this work against
    AlphaQubit, NVIDIA Ising-Decoding, a near-term neural decoder, and Google
    Willow below-threshold decoding context with a non-head-to-head caveat.
28. Static manuscript check added in `tools/check_main_tex_static.py`; current
    artifact `artifacts/eval/nn/main_tex_static_check_summary.json` passes with
    `0` failed errors and `0` failed warnings.
29. Figure package builder now records all six manuscript figures in
    `artifacts/figures/predecoder/predecoder_figure_package_summary.json`.
30. Overleaf package builder added in `tools/prepare_overleaf_package.py`.
    Current upload package is `artifacts/overleaf_predecoder_package.zip` and
    targets XeLaTeX with PNG-based figure inclusion.
31. Figure quality pass completed on 2026-05-20: fig1--fig6 were regenerated
    with wider canvases, shorter multi-line labels, and separated legend/summary
    panels to reduce overflow.
32. Overleaf compile fix completed on 2026-05-20: `main.tex` no longer uses
    `svg` or `\includesvg`; generated PNG figures are packaged directly.
33. Manuscript polish pass completed on 2026-05-20 after successful Overleaf
    compilation: `main.tex` was rewritten in a more natural Korean thesis
    style, and all six figures were regenerated with larger text and simpler
    layouts.

## Current Results

All results below are held-out `stage_c_corr` comparisons against raw no-edit
PyMatching.

| distance | raw PyMatching | selected predecoder | candidate branch | oracle | selected delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.935302734` | `0.935302734` | `0.992187500` | `+0.006591797` |
| d5 | `0.888671875` | `0.894287109` | `0.891845703` | `0.978515625` | `+0.005615234` |
| d7 | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

Interpretation:

- d3 and d5 are the successful positive selected-mode results.
- d3 is uniformly positive over seed `0..7`; d5 is positive-mean and
  selected-safe, but not a strong seed-level significance claim.
- d5 selected-mode gain comes from seeds `2` and `3`; candidate-only harmful
  seeds `4` and `6` are blocked by fallback.
- d7 is not a robust learned-recovery result.
- d7 still has large candidate-oracle headroom, so the limitation is selector
  ranking/generalization rather than candidate coverage.

## D7 State

Current d7 support-guard evidence over seeds `0..57`:

- mean selected delta: `+0.000151536`
- mean actual candidate delta: `-0.001515356`
- mean candidate-oracle delta: `+0.096679688`
- selected-positive seeds: `2, 11`
- missed candidate-positive seeds: `0, 28, 43, 45`
- harmful candidate seeds blocked by selected mode: `17`
- all `58` seeds have positive oracle headroom
- validation-positive candidate seeds split into `13` held-out harmful, `4`
  held-out neutral, and `5` held-out positive outcomes
- d7 false-positive taxonomy artifact:
  `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`
- d7 hyperparameter sensitivity artifact:
  `artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json`

The required d7 sentinel gate for any new objective is:

| role | seeds | required behavior |
| --- | --- | --- |
| preserve true positives | `2, 11` | do not destroy existing learned gains |
| recover missed positives | at least one of `0, 28, 43, 45` | improve candidate ranking |
| block false positives | `8, 13, 17, 33, 54` | keep harmful candidate behavior blocked |

## Rejected D7 Directions

Do not continue these without a genuine redesign:

| direction | result |
| --- | --- |
| scalar adoption-threshold grid | `183040` policies checked, `0` pass the sentinel gate |
| cross-family positive-vs-negative objective | fails the seed54 false-positive gate |
| candidate-compatibility pairwise top-k | seed54 blocked, but seed2 candidate delta becomes `-0.136718750` |
| risk-aware/risk-guard selector head | fair seed2 rerun restores candidate settings, but selected mode still falls back to raw no-edit; harm hard guard does not reliably filter selected harmful candidates |
| risk-guard plus same-shot hard ranking | seed2 candidate delta improves to `+0.002929688`, but selected mode still falls back to raw no-edit because validation harmed count exceeds the safety cap |

## Remaining Work

The rational forward plan is now fixed in `PREDECODER_REMAINING_WORK.md`.
The post-presentation d7 selector experiment is separately fixed in
`RISK_AWARE_SELECTOR_EXPLORATORY_PLAN.md`. Treat it as optional exploratory
work: it must not modify the frozen main evidence unless it passes the
sentinel gate, the full d7 run, and d3/d5 regression checks.
The implementation/result note is in
`RISK_AWARE_SELECTOR_EXPLORATORY_RESULTS.md`: checked `risk_aware` and
`risk_guard` seed-2 variants failed to preserve the frozen seed-2 selected
gain. A fair `risk_guard` rerun restored the scalar local-motif candidate
settings, but selected mode still fell back to raw no-edit because validation
support was too sparse. A post-hoc harm-logit hard guard also failed to filter
the selected high-utility harmful candidates reliably. A final
`risk_guard + positive_negative_hard 0.5/0.5` check improved seed-2 candidate
delta relative to plain `risk_guard`, but selected mode still failed because
validation harmful edits exceeded the safety cap. Do not expand this exact path
before seed-2 preservation is recovered by a genuinely different objective.

Completed work that should not be repeated as a priority:

- d3/d5 seed `0..7` robustness check and bootstrap CI
- d3/d5 paired exact seed-level tests
- noise-family summary
- baseline comparison boundary
- ablation/failure-path synthesis
- final result consistency check
- method description
- d7 harmful-edit taxonomy
- hyperparameter sensitivity table
- thesis core integration draft
- Korean thesis core draft
- reproducibility package
- figure package
- scalar d7 adoption-grid search
- tested d7 cross-family hard-negative and candidate-compatibility paths

Required finish work:

1. Polish `main.tex` into the required final university format.
2. Upload `artifacts/overleaf_predecoder_package.zip` to Overleaf and compile
   with XeLaTeX. Confirm the Overleaf file tree contains
   `artifacts/figures/predecoder/*.png`.
3. Add the reproducibility appendix from
   `PREDECODER_REPRODUCIBILITY_PACKAGE.md`.
4. Run the final regeneration, `main.tex` static check, and syntax-check
   commands.
5. Keep any optional d7 redesign behind the preserve/recover/block sentinel
   gate.

Optional research work:

- Continue only with a real selector-ranking/generalization redesign; the
  current allowed path is the risk-aware selector plan in
  `RISK_AWARE_SELECTOR_EXPLORATORY_PLAN.md`.
- First test only on sentinel seeds.
- Report candidate and selected deltas separately.
- Stop immediately if the sentinel gate fails.
- d3/d5 seed `4..7`, confidence intervals, and one compact d7
  hyperparameter sensitivity table are already complete; add no more unless
  the final thesis explicitly needs it.

## Recommended Next Session

The next session should do one of two things:

1. Finish paper/report packaging using d3/d5 success plus d7 limitation.
2. If continuing d7, propose one new ranking redesign and test only the
   sentinel gate before any broad seed extension.

Default recommendation: finish the paper/report path first. The d7 search has
hit a rational stopping point for the current model family, and the core
technical draft, clean Korean draft, and reproducibility package now exist.
The main figure package also exists; remaining work is mostly formatting and
appendix integration.

The current thesis structure and draft text are tracked in
`GRADUATION_THESIS_DRAFT.md`.

## Key Files

- `GRADUATION_THESIS_DRAFT.md`
- `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`
- `PREDECODER_FIGURE_PACKAGE.md`
- `PREDECODER_METHOD_DESCRIPTION.md`
- `PREDECODER_FINAL_RESULT_TABLES.md`
- `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
- `PREDECODER_D3_D5_PAIRED_STATISTICS.md`
- `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
- `PREDECODER_BASELINE_COMPARISON.md`
- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
- `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`
- `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md`
- `PREDECODER_HYPERPARAMETER_SENSITIVITY.md`
- `PREDECODER_REPRODUCIBILITY_PACKAGE.md`
- `PREDECODER_CONSOLIDATED_EVIDENCE.md`
- `PREDECODER_REMAINING_WORK.md`
- `RESEARCH_DOCUMENTATION_AUDIT.md`
- `RISK_AWARE_SELECTOR_EXPLORATORY_RESULTS.md`
- `RISK_AWARE_SELECTOR_EXPLORATORY_PLAN.md`
- `RESEARCH_EVALUATION_ACTION_PLAN.md`
- `NEXT_SESSION_HANDOFF.md`
- `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`
- `artifacts/eval/nn/sedp_final_result_consistency_check.json`
- `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json`
- `artifacts/eval/nn/sedp_baseline_comparison_summary.json`
- `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`
- `artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json`
