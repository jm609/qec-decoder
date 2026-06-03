# Predecoder Reproducibility Package

This document is the audit checklist for reproducing the final predecoder
evidence without rerunning long training jobs. It records the source scripts,
generated summaries, result documents, and minimum validation commands.

## Claim Boundary

Final claim supported by this package:

> A transition-aware patch-head neural pre-decoder improves raw PyMatching on
> held-out d3 and d5 surface-code logical-frame decoding by selecting local
> detector edits before matching, while d7 reveals a selector-ranking and
> generalization limitation rather than a lack of local-edit candidate
> headroom.

Do not use this package to claim:

- solved d7 learned recovery
- replacement of PyMatching by a neural decoder
- d7 candidate-set exhaustion
- broad noise-model robustness beyond the reported constructed families

## Main Source Files

| role | file |
| --- | --- |
| environment record | `ENVIRONMENT.md` |
| pinned dependency list | `requirements.txt` |
| evaluation-report action plan | `RESEARCH_EVALUATION_ACTION_PLAN.md` |
| circuit generation | `circuits.py` |
| experiment config | `config.py` |
| dataset generation | `sample_dataset.py` |
| SI1000 noise injection | `noise_si1000.py` |
| Willow-core/local noise injection | `noise_willowcore.py` |
| raw PyMatching baseline | `decoders/baseline_pymatching.py` |
| final neural predecoder | `decoders/syndrome_edit_predecoder.py` |
| direct neural context baseline | `decoders/factorized_logical_frame_decoder.py` |
| multiscale direct context baseline | `decoders/multiscale_factorized_decoder.py` |
| RectCNN readiness baseline | `decoders/baseline_rectcnn.py` |

## Canonical Result Documents

| purpose | document |
| --- | --- |
| method description | `PREDECODER_METHOD_DESCRIPTION.md` |
| final result tables | `PREDECODER_FINAL_RESULT_TABLES.md` |
| consolidated evidence | `PREDECODER_CONSOLIDATED_EVIDENCE.md` |
| d3/d5 success structure | `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md` |
| d3/d5 robustness | `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md` |
| d3/d5 paired statistics | `PREDECODER_D3_D5_PAIRED_STATISTICS.md` |
| oracle recovery distribution | `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md` |
| hyperparameter sensitivity | `PREDECODER_HYPERPARAMETER_SENSITIVITY.md` |
| noise-family analysis | `PREDECODER_NOISE_FAMILY_ANALYSIS.md` |
| baseline comparison | `PREDECODER_BASELINE_COMPARISON.md` |
| ablation/failure synthesis | `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md` |
| d7 targeted bottleneck | `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md` |
| d7 harmful-edit taxonomy | `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md` |
| figure package | `PREDECODER_FIGURE_PACKAGE.md` |
| thesis draft | `GRADUATION_THESIS_DRAFT.md` |
| clean Korean thesis core draft | `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md` |

## Canonical JSON Artifacts

| purpose | artifact |
| --- | --- |
| consolidated final evidence | `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json` |
| final table consistency check | `artifacts/eval/nn/sedp_final_result_consistency_check.json` |
| d3/d5 robustness | `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json` |
| d3/d5 seed0..7 bootstrap CI | `artifacts/eval/nn/sedp_d3_d5_seed0_7_bootstrap_ci_summary.json` |
| d3/d5 paired statistics | `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json` |
| oracle recovery distribution | `artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json` |
| hyperparameter sensitivity | `artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json` |
| noise-family analysis | `artifacts/eval/nn/sedp_noise_family_analysis_summary.json` |
| baseline comparison | `artifacts/eval/nn/sedp_baseline_comparison_summary.json` |
| ablation/failure synthesis | `artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json` |
| d7 selector bottleneck | `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json` |
| d7 validation-vs-heldout scatter | `artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json` |
| d7 harmful taxonomy | `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json` |
| d7 adoption-grid rejection | `artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json` |
| d7 support-guard candidate/oracle analysis | `artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json` |
| main.tex static check | `artifacts/eval/nn/main_tex_static_check_summary.json` |
| thesis figure package summary | `artifacts/figures/predecoder/predecoder_figure_package_summary.json` |
| Overleaf package manifest | `artifacts/overleaf_predecoder_package/overleaf_package_manifest.json` |

## Figure Artifacts

SVG files are the editable sources. PNG files are the Overleaf-ready figure
inputs used by `main.tex`.

| figure | SVG source | PNG compile input |
| --- | --- | --- |
| pipeline | `artifacts/figures/predecoder/fig1_predecoder_pipeline.svg` | `artifacts/figures/predecoder/fig1_predecoder_pipeline.png` |
| model architecture | `artifacts/figures/predecoder/fig2_model_architecture.svg` | `artifacts/figures/predecoder/fig2_model_architecture.png` |
| main accuracy comparison | `artifacts/figures/predecoder/fig3_main_accuracy_comparison.svg` | `artifacts/figures/predecoder/fig3_main_accuracy_comparison.png` |
| d7 oracle gap and false positives | `artifacts/figures/predecoder/fig4_d7_oracle_gap_false_positive.svg` | `artifacts/figures/predecoder/fig4_d7_oracle_gap_false_positive.png` |
| d7 validation-vs-heldout scatter | `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg` | `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.png` |
| oracle recovery distribution | `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg` | `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.png` |

## Main Result Table

All rows are held-out `stage_c_corr` logical-frame class4 accuracy.

| distance | seeds | raw PyMatching | selected predecoder | candidate branch | target local-edit oracle | selected delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0..7` | `0.928710938` | `0.935302734` | `0.935302734` | `0.992187500` | `+0.006591797` |
| d5 | `0..7` | `0.888671875` | `0.894287109` | `0.891845703` | `0.978515625` | `+0.005615234` |
| d7 | `0..57` | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

## Seed-Expanded CI Summary

The d3/d5 claim now uses seed `0..7` rather than seed `0..3`. The confidence
intervals below are seed-level bootstrap intervals with `20,000` resamples.

| distance | selected delta mean | 95% bootstrap CI | selected seed classes | note |
| --- | ---: | ---: | ---: | --- |
| d3 | `+0.006591797` | `[+0.004516602, +0.008544922]` | `8/0/0` | uniformly positive |
| d5 | `+0.005615234` | `[+0.000000000, +0.013671875]` | `2/6/0` | conservative selected-mode gain |

d5 candidate-only behavior has harmful seeds `4` and `6`; selected-mode
adoption blocks both by falling back to raw no-edit.

## Regenerate Summary Artifacts

These commands rebuild the paper-facing summary JSON files from existing
artifacts. They do not retrain the neural models.

```bash
python tools/build_predecoder_consolidation_summary.py
python tools/build_baseline_comparison_summary.py
python tools/build_ablation_failure_synthesis_summary.py
python tools/build_d3_d5_robustness_summary.py
python tools/build_d3_d5_seed_expansion_ci_summary.py
python tools/build_d3_d5_paired_statistics_summary.py
python tools/build_oracle_recovery_distribution_summary.py
python tools/build_hyperparameter_sensitivity_summary.py
python tools/build_noise_family_analysis_summary.py
python tools/build_d7_harmful_edit_taxonomy_summary.py
python tools/build_predecoder_figure_package.py
python tools/build_d7_validation_heldout_scatter.py
python tools/rasterize_predecoder_figures.py --scale 2
python tools/build_final_result_consistency_summary.py
python tools/check_main_tex_static.py
python tools/prepare_overleaf_package.py
```

Expected final consistency status:

```text
pass: true
num_checks: 37
num_failed: 0
```

Expected `main.tex` static check status:

```text
pass: true
num_failed_errors: 0
num_failed_warnings: 0
```

Expected Overleaf package status:

```text
pass: true
compiler: XeLaTeX
missing_required: []
```

## Minimum Syntax Check

Run this before final submission or handoff:

```powershell
$files = @('circuits.py', 'config.py', 'sample_dataset.py') + (Get-ChildItem decoders -Filter *.py).FullName + (Get-ChildItem tools -Filter *.py).FullName
python -m py_compile @files
```

On shells that expand globs before invoking Python, the equivalent compact
form is:

```bash
python -m py_compile circuits.py config.py sample_dataset.py decoders/*.py tools/*.py
```

## Minimal Regression Tests

The lightweight regression tests use only Python's standard `unittest` module.
They check the candidate-first safety adoption guard, the paper-facing summary
artifacts, the `main.tex` static-check path, the six-figure package summary,
and the Overleaf package manifest.

```bash
python -m unittest discover -s tests -v
```

Expected result:

```text
Ran 18 tests
OK
```

## Environment Rebuild

The dependency record is intentionally minimal:

- `requirements.txt`
- `ENVIRONMENT.md`

Install command:

```bash
python -m pip install -r requirements.txt
```

The recorded environment used Python `3.10.20`, CPU-only PyTorch `2.10.0`,
Stim `1.15.0`, and PyMatching `2.3.1`.

If time is tight, the minimum check for the generated summary tooling is:

```bash
python -m py_compile tools/build_predecoder_consolidation_summary.py tools/build_baseline_comparison_summary.py tools/build_ablation_failure_synthesis_summary.py tools/build_d3_d5_robustness_summary.py tools/build_d3_d5_seed_expansion_ci_summary.py tools/build_d3_d5_paired_statistics_summary.py tools/build_oracle_recovery_distribution_summary.py tools/build_hyperparameter_sensitivity_summary.py tools/build_noise_family_analysis_summary.py tools/build_d7_harmful_edit_taxonomy_summary.py tools/build_predecoder_figure_package.py tools/build_d7_validation_heldout_scatter.py tools/rasterize_predecoder_figures.py tools/build_final_result_consistency_summary.py tools/check_main_tex_static.py tools/prepare_overleaf_package.py
```

## Optional Dataset Smoke Command

Use this only when checking that dataset creation still works:

```bash
python sample_dataset.py --out-root artifacts/datasets/dev --distance 5 --rounds 10 --basis z --shots 1000
python inspect_samples.py --manifest artifacts/datasets/dev/manifest.json
python decoders/baseline_pymatching.py --manifest artifacts/datasets/dev/manifest.json
```

Do not overwrite published experiment outputs without recording it in the
handoff.

## D7 Sentinel Gate For Any Future Objective

Any new d7 selector-ranking objective must first pass this small gate before a
full seed sweep:

| role | seeds | required behavior |
| --- | --- | --- |
| preserve true positives | `2, 11` | keep selected/candidate gains |
| recover missed positives | at least one of `0, 28, 43, 45` | improve candidate ranking without opening false positives |
| block false positives | `8, 13, 17, 18, 32, 33, 53, 54` | keep harmful candidate behavior blocked |

If this gate fails, do not expand to all `0..57` d7 seeds.

## Final Thesis Assembly Checklist

| item | source | status |
| --- | --- | --- |
| method section | `PREDECODER_METHOD_DESCRIPTION.md` and `GRADUATION_THESIS_DRAFT.md` | integrated draft exists |
| evaluation-report closeout | `RESEARCH_EVALUATION_ACTION_PLAN.md` | required items addressed |
| main result table | `PREDECODER_FINAL_RESULT_TABLES.md` | consistency checked |
| d3/d5 robustness | `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md` | ready |
| d3/d5 paired statistics | `PREDECODER_D3_D5_PAIRED_STATISTICS.md` | ready |
| oracle recovery distribution | `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md` | ready |
| hyperparameter sensitivity | `PREDECODER_HYPERPARAMETER_SENSITIVITY.md` | ready |
| noise-family discussion | `PREDECODER_NOISE_FAMILY_ANALYSIS.md` | ready |
| baseline boundary | `PREDECODER_BASELINE_COMPARISON.md` | ready |
| ablation/failure synthesis | `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md` | ready |
| d7 limitation taxonomy | `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md` | ready |
| figure package | `PREDECODER_FIGURE_PACKAGE.md` | SVG sources and PNG compile inputs generated |
| Korean thesis core prose | `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md` | ready for school-format polish |
| reproducibility appendix | this document | ready for final polish |

## Remaining Reproducibility Risk

- Some older direct neural baseline artifacts are context baselines and should
  not be overused as strict head-to-head comparisons.
- Training reruns are not required for the final claim, but any new rerun must
  record seed, distance, noise family, selected/candidate/oracle metrics, and
  artifact path.
- The thesis should cite exact artifact paths for all generated summaries used
  in tables.
