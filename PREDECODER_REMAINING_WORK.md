# Predecoder Remaining Work

This document fixes the remaining work after consolidating the d3/d5 selected
predecoder success and the current d7 selector-ranking limitation.

## Latest Remaining Work as of 2026-05-28

The research result should now be treated as fixed for thesis/poster purposes.
The next work is not a new decoder experiment unless the user explicitly asks
for it. The remaining tasks are submission production tasks.

Fixed title:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Do not change this title; it has already been submitted.

Active drafts:

- `main.tex`: Korean polished draft.
- `main_en.tex`: English polished draft.
- `legacy_archive/main_ko_reference.tex`: preserved older Korean reference state.

Current package outputs:

- `artifacts/overleaf_predecoder_package.zip`: Korean package, main file
  `main.tex`.
- `artifacts/overleaf_predecoder_package_en.zip`: English package, main file
  `main_en.tex`.

Current progress estimate:

| area | progress | next gate |
| --- | ---: | --- |
| research experiments/evidence | 100% | keep claims fixed |
| Korean manuscript content | 90-95% | Overleaf compile and visual review |
| English manuscript content | 90-95% | Overleaf compile and visual review |
| poster | 25-35% | build final poster layout from fixed figures/results |
| overall thesis+poster deliverable | 85-90% | PDF/poster production finish |

Latest completed manuscript fixes:

- `showkeys` removed.
- `surface code` terminology unified in the Korean draft.
- `logical_class4` / logical frame / logical decision terminology clarified.
- held-out oracle search guarded against leakage: it is post hoc analysis only.
- dataset split and caption now separate train/validation target generation
  from held-out evaluation/oracle analysis.
- validation split `154` shots explained as a fixed split.
- English draft style pass completed to reduce AI-template wording and restore
  selector training detail.

Immediate next tasks:

1. Upload the relevant Overleaf zip and compile with XeLaTeX.
2. Fix only compile/log/layout problems: table overlap, figure placement,
   bibliography style, line breaks, and school-required formatting.
3. Prepare the poster from the fixed narrative:
   d3/d5 selected-mode success, d5 conservative adoption, d7 selector
   generalization limitation.
4. Do not alter the result claims unless a new explicitly requested experiment
   produces a clean, documented, reproducible result.
5. Before final submission, rerun:

```powershell
python tools\check_main_tex_static.py
python tools\check_main_tex_static.py --main-tex main_en.tex --out artifacts\eval\nn\main_en_tex_static_check_summary.json
python tools\prepare_overleaf_package.py
python tools\prepare_overleaf_package.py --main-file main_en.tex --out-dir artifacts\overleaf_predecoder_package_en --zip artifacts\overleaf_predecoder_package_en.zip
```

Submitted English title:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Current thesis draft:

- `main.tex`
- `GRADUATION_THESIS_DRAFT.md`
- `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`
- `MAIN_TEX_THESIS_STRUCTURE.md`
- integrated core draft added: method, setup, result, ablation, d7 limitation,
  and discussion sections
- clean Korean core draft added for submission-format writing
- `main.tex` now contains the LaTeX manuscript body using the agreed logical
  flow; figures are included through a PDF/PNG macro and the generated PNG
  files are now present for Overleaf compilation
- `tools/rasterize_predecoder_figures.py` rasterizes the generated SVG sources
  to PNG without requiring Inkscape or Overleaf SVG conversion

Method description:

- `PREDECODER_METHOD_DESCRIPTION.md`

Reproducibility package:

- `PREDECODER_REPRODUCIBILITY_PACKAGE.md`
- `ENVIRONMENT.md`
- `requirements.txt`

Figure package:

- `PREDECODER_FIGURE_PACKAGE.md`
- `artifacts/figures/predecoder/predecoder_figure_package_summary.json`
- `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg`
- `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg`

Evaluation-report action plan:

- `RESEARCH_EVALUATION_REPORT.md`
- `RESEARCH_EVALUATION_ACTION_PLAN.md`

Latest evaluation-report alignment:

- The active evaluation score is `8.7/10`, not the older `8.4/10` planning
  state.
- `main.tex` now includes a d5 seed-level fallback table, making clear that
  seeds `2` and `3` provide the selected-mode gain while candidate-only harmful
  seeds `4` and `6` are blocked.
- The current manuscript keeps the quantitative scope focused on Stage A/B/C.
  Do not reintroduce Stage D/E implementation details into the main text unless
  the user explicitly asks for appendix-style scope notes.
- The cross-paper comparison table has now been added to `main.tex`. The
  remaining 8.7-to-9.0 roadmap is optional compact sensitivity sweep, final
  Overleaf compile, and consistency checks at closeout.

Final result consistency check:

- `artifacts/eval/nn/sedp_final_result_consistency_check.json`
- `tools/build_final_result_consistency_summary.py`

Main manuscript static check:

- `tools/check_main_tex_static.py`
- `artifacts/eval/nn/main_tex_static_check_summary.json`
- Current status: `pass=true`, `0` failed errors, `0` failed warnings. The six
  Overleaf-ready PNG figure files are present.

Overleaf package:

- `tools/prepare_overleaf_package.py`
- `OVERLEAF_COMPILE_GUIDE.md`
- `artifacts/overleaf_predecoder_package/`
- `artifacts/overleaf_predecoder_package.zip`
- Current status: package `pass=true`, compiler target `XeLaTeX`, required
  files present.

D3/d5 robustness follow-up:

- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
- `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json`
- `PREDECODER_D3_D5_PAIRED_STATISTICS.md`
- `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json`

Noise-family follow-up:

- `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
- `artifacts/eval/nn/sedp_noise_family_analysis_summary.json`

Baseline-comparison follow-up:

- `PREDECODER_BASELINE_COMPARISON.md`
- `artifacts/eval/nn/sedp_baseline_comparison_summary.json`

Ablation/failure-path synthesis:

- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
- `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json`

D7 harmful-edit taxonomy:

- `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`
- `artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json`

Oracle recovery distribution:

- `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md`
- `artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json`

Hyperparameter sensitivity:

- `PREDECODER_HYPERPARAMETER_SENSITIVITY.md`
- `artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json`

## Current Fixed State

The project should now treat these as fixed evidence:

| distance | claim status | held-out selected delta |
| --- | --- | ---: |
| d3 | positive selected-mode result | `+0.006591797` |
| d5 | conservative selected-mode result | `+0.005615234` |
| d7 | controlled scaling limitation | `+0.000151536` |

The d7 candidate-oracle gap remains large:

- mean actual candidate delta: `-0.001515356`
- mean candidate-oracle delta: `+0.096679688`
- all `58` checked d7 seeds have positive oracle headroom

Therefore the remaining d7 issue is selector ranking/generalization, not
candidate coverage.

## Rational Next-Work Review

This section fixes the forward plan after the d3/d5 robustness, noise-family,
and baseline-comparison follow-ups.

Completed and no longer a priority:

| item | status | reason |
| --- | --- | --- |
| d3/d5 basic robustness check | complete | seed `0..7` gives d3 `8/0/0` and d5 `2/6/0` positive/neutral/harmful selected outcomes, with no harmful selected held-out seed |
| d3/d5 paired statistics | complete | exact sign/sign-flip tests support d3 strongly but keep d5 conservative: d3 one-sided p `0.003906250`, d5 one-sided p `0.250000000` |
| baseline comparison boundary | complete | raw PyMatching is fixed as the fair same-artifact baseline; FLFD/M3D/RectCNN are context baselines |
| noise-family summary | complete | d3/d5/d7 behavior across validation and held-out families is already summarized |
| ablation/failure-path synthesis | complete | direct neural baselines, selected predecoder success, and rejected d7 paths are now summarized in one paper-ready table |
| final result consistency check | complete | `37` Markdown-vs-JSON checks pass with `0` failures |
| method description | complete | `PREDECODER_METHOD_DESCRIPTION.md` now fixes the implemented input, trunk, candidate, selector, adoption, and PyMatching handoff structure |
| d7 harmful-edit taxonomy | complete | validation-positive d7 candidate branches are held-out harmful in `13/22` cases; all `17` harmful candidate seeds are blocked by selected mode |
| seed-level oracle recovery distribution | complete | selected mean oracle-gap recovery is d3 `10.38%`, d5 `6.25%`, d7 `0.14%`; d7 candidate-oracle recovery remains high at `86.84%` |
| thesis core integration draft | complete | `GRADUATION_THESIS_DRAFT.md` now contains paper-ready English core sections for method, setup, results, ablation, d7 limitation, and discussion |
| Korean thesis core draft | complete | `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md` now contains clean Korean abstract, chapter-level prose, result tables, d7 limitation analysis, conclusion, and figure/table captions |
| main.tex thesis body | complete draft | old template content is replaced with the predecoder thesis structure; figure inclusion now uses a PDF/PNG macro; remaining LaTeX work is compiler check and final school-format metadata |
| reproducibility package | complete | `PREDECODER_REPRODUCIBILITY_PACKAGE.md` records source files, canonical JSON artifacts, regeneration commands, syntax checks, and d7 sentinel gate |
| environment/dependency record | complete | `requirements.txt` and `ENVIRONMENT.md` record Python `3.10.20`, CPU-only PyTorch `2.10.0`, Stim `1.15.0`, and PyMatching `2.3.1` |
| minimal regression tests | complete | `tests/test_predecoder_regression.py` now runs 18 tests covering candidate-first safety adoption, paper-facing summary artifacts, `main.tex` static checks, the six-figure package summary, and the Overleaf package manifest |
| figure package | complete | `PREDECODER_FIGURE_PACKAGE.md`, six regenerated SVG sources, and six Overleaf-ready PNG figures cover pipeline, architecture, main accuracy comparison, d7 oracle/false-positive limitation, d7 validation-vs-heldout mismatch, and oracle recovery distribution; layouts were widened on 2026-05-20 to reduce overflow/overlap, and `tools/build_predecoder_figure_package.py` records all six figures in the summary |
| hyperparameter sensitivity | complete | d7 sentinel seeds `0,2,5` show identity-margin weight `0.5` is the best checked compromise; `0.25` admits harmful seed0, and `1.0` suppresses true-positive seed2 |
| evaluation-report action plan | complete | updated to the latest `8.7/10` report: d5 conservative wording, d5 fallback table, d7 oracle recovery emphasis, Stage D/E exclusion, paired statistics, and tests |
| scalar d7 adoption calibration | stop | `183040` policies found `0` passing preserve/recover/block |
| tested d7 cross-family hard-negative and candidate-compatibility paths | stop | they fail sentinel preservation or false-positive blocking |

The remaining work should be chosen by this rule:

> Prefer work that makes the final research claim more defensible. Do not
> spend time on broad d7 tuning unless a new idea first passes the small
> sentinel gate.

Priority classes:

| class | work type | value | risk | decision |
| --- | --- | --- | --- | --- |
| A | final thesis/report packaging | required for graduation | low | do next |
| A | method and result-table consistency check | required for defensible claims | low | complete; rerun at final closeout |
| A | reproducibility package | required for auditability | low | complete; use as appendix source |
| B | ablation/failure-path synthesis integration | high explanatory value | low | integrated into core draft; polish in final prose |
| B | d7 selector-ranking limitation writeup | high research value | low | integrated into core draft; polish in final prose |
| C | additional d3/d5 seeds or confidence intervals | modest extra confidence | medium time cost | complete for seed `0..7`; do not extend further unless requested |
| C | one genuinely new d7 ranking objective | possible research upside | high time/risk | sentinel-only first |
| D | broad d7 seed expansion of rejected recipes | low value | high time cost | avoid |
| D | new feature branches without explaining current failure | low value | high complexity | avoid |

Practical progress estimate:

| axis | estimate | interpretation |
| --- | ---: | --- |
| model/research development | `100%` | core evidence is fixed for the current thesis claim |
| analysis consolidation | `95%` | robustness, paired statistics, noise-family, baseline, ablation synthesis, method description, final-table check, d7 limitation, and reproducibility docs exist |
| thesis writing/package | `90-95%` | Korean and English drafts are polished; remaining work is Overleaf visual/log cleanup and school formatting |
| poster | `25-35%` | narrative, figures, and result tables exist; final poster layout remains |
| overall thesis+poster deliverable | `85-90%` | research is complete; submission production remains |

## Required Remaining Work

| priority | task | completion criterion |
| ---: | --- | --- |
| 1 | Polish `main.tex` into final school format | Confirm compiler/class requirements, author metadata, title page needs, and Korean typography |
| 2 | Finalize cross-paper comparison context | Table is now present in `main.tex`; final pass should verify citations and wording after PDF compile |
| 3 | Prepare Overleaf/PDF compile | Upload `artifacts/overleaf_predecoder_package.zip` to Overleaf and compile with XeLaTeX; figures now compile from PNG without SVG conversion |
| 4 | Prepare final appendix/reproducibility section | Use `PREDECODER_REPRODUCIBILITY_PACKAGE.md` and cite artifact paths |
| 5 | Final consistency pass | Regenerate summary artifacts, run final table check, run `tools/check_main_tex_static.py`, and run `py_compile` |
| 6 | Optional compact sensitivity sweep | Add at most one small d3 sentinel sweep such as top-k or hard-shot weight if time remains |
| 7 | Optional d7 redesign only if justified | New d7 objective must pass the sentinel gate before any broad sweep |

## Optional Research Extensions

These are not required for the main thesis claim, but they are reasonable only
if time remains after the required work.

| option | when it is worth doing | stop condition |
| --- | --- | --- |
| add d3/d5 seed `4..7` | complete | seed-expanded results are already folded into the final tables |
| compute confidence intervals or bootstrap intervals | complete for d3/d5 seed `0..7` | do not turn it into another model-search loop |
| exact paired seed-level tests | complete for d3/d5 selected deltas | use mainly to keep d5 wording conservative |
| one compact hyperparameter sensitivity table | complete for d7 identity-margin weight on seeds `0,2,5` | do not present it as exhaustive optimization |
| cross-paper comparison table | complete in `main.tex` | keep the non-head-to-head caveat and verify bibliography formatting during final compile |
| d7 harmful-edit taxonomy | complete; use it in the limitation chapter | do not turn it into a new feature branch without a separate sentinel-gated objective |
| Stage D/E leakage/DQLR extension | defer to follow-up work | requires new shot-level `postprocess.py` implementation and full revalidation |
| new d7 selector-ranking objective | only if it is a qualitatively new ranking/generalization idea | stop immediately if it fails the sentinel gate |

## Optional D7 Work

D7 should continue only if the next objective is a genuine selector-ranking
redesign. It should not be another scalar adoption threshold, support guard,
or top-k compatibility sweep.

Post-presentation d7 selector work is now tracked separately in
`RISK_AWARE_SELECTOR_EXPLORATORY_PLAN.md`. It is an optional exploratory path,
not part of the frozen main thesis evidence. The main d3/d5 positive result
and d7 scaling-limitation narrative should remain unchanged unless that plan
passes its sentinel, d7 full-run, and d3/d5 regression gates.

Any new d7 objective must first pass this small sentinel gate:

| gate role | required seeds | required behavior |
| --- | --- | --- |
| preserve true positives | `2, 11` | keep selected/candidate gains from being destroyed |
| recover missed positives | at least one of `0, 28, 43, 45` | improve candidate ranking without opening false positives |
| block false positives | `8, 13, 17, 33, 54` | keep harmful candidate behavior blocked |

The candidate and selected deltas must be reported separately. If the sentinel
gate fails, do not expand to the full `0..57` seed set.

The post-presentation risk-aware selector-head change has now been implemented
and tested on seed `2`. The fair `risk_guard` rerun restored the scalar
local-motif candidate settings, but still failed to preserve the frozen seed-2
selected-mode gain. A post-hoc harm-logit hard guard also failed to make the
selected high-utility candidates safe enough. A final `risk_guard` plus
same-shot positive-vs-negative hard-ranking check improved seed-2 candidate
delta from `+0.001953125` to `+0.002929688`, but selected mode still fell back
to raw no-edit because validation harmed count exceeded the safety cap. Treat
this exact path as rejected unless a genuinely different objective first
recovers seed-2 preservation.

## Rejected D7 Paths

These should not be expanded without a real redesign:

| path | reason |
| --- | --- |
| scalar adoption-threshold grid | `183040` policies checked, `0` pass preserve/recover/block |
| cross-family hard positive-vs-negative objective | fails the seed54 false-positive gate |
| candidate-compatibility pairwise top-k | blocks seed54 but destroys seed2 candidate delta to `-0.136718750` |
| risk-aware/risk-guard selector head | fair seed2 rerun restores candidate settings but selected mode falls back to raw no-edit; harm hard guard does not reliably filter selected harmful candidates |
| risk-guard plus same-shot hard ranking | seed2 candidate delta improves but selected mode remains blocked by the harm guard |

## Work To Avoid

- Do not add another feature branch before the current selector-ranking failure
  is explained in the final writeup.
- Do not present d7 as a robust learned improvement.
- Do not claim candidate-set exhaustion at d7.
- Do not broaden rejected d7 recipes to more seeds just to collect more
  negative evidence.

## Recommended Finish Plan

1. Write the final report around the d3/d5 positive result and d7 scaling
   limitation. The clean Korean core draft is now in
   `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`.
2. Polish the Korean draft into the final required university format.
3. Insert the generated figures from `PREDECODER_FIGURE_PACKAGE.md` and adjust
   captions to the school template.
4. Add the reproducibility appendix from `PREDECODER_REPRODUCIBILITY_PACKAGE.md`.
5. Keep optional d7 optimization behind the sentinel gate, not as the main
   project path.
6. Run the final syntax check and regenerate the consolidated evidence summary
   before closing the project.
