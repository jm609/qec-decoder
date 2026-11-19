# Predecoder Figure Package

This document records the thesis figure package for the transition-aware
predecoder project. The figures are generated as SVG source files from fixed
summary artifacts and rasterized to PNG for Overleaf/XeLaTeX compilation.

2026-05-20 visual-quality update:

- All six SVG figures were regenerated with wider canvases and less crowded
  layouts.
- Long labels were split into shorter multi-line blocks.
- Legends and diagnostic summaries were moved into fixed side or bottom panels
  to reduce overlap with plots and boxes.
- All six figures were rasterized to PNG locally so Overleaf does not need SVG
  conversion.
- The Overleaf package was rebuilt after regeneration.

2026-05-20 polish update:

- The manuscript figures were regenerated again after PDF inspection showed
  that the earlier versions were too text-heavy and visually loose.
- Figure text was enlarged, long explanatory sentences were moved out of the
  images and into captions/body text, and Figure 5/6 were simplified around
  the diagnostic quantities that matter for the thesis argument.
- `main.tex` now includes figures at full `\textwidth` inside `figure*`
  environments to improve readability in the compiled PDF.

Builder:

- `tools/build_predecoder_figure_package.py`
- `tools/build_predecoder_figures_matplotlib.py`
- `tools/build_d7_validation_heldout_scatter.py`
- `tools/build_oracle_recovery_distribution_summary.py`
- `tools/rasterize_predecoder_figures.py`

Source artifacts:

- `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`
- `artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json`
- `artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json`
- `artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json`
- `artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json`

Summary artifact:

- `artifacts/figures/predecoder/predecoder_figure_package_summary.json`
- `artifacts/figures/predecoder_v2/predecoder_v2_figure_summary.json`

## Matplotlib V2 Figures

2026-05-27 update:

- A new matplotlib-based builder was added for cleaner diagram and result
  figures.
- The v2 outputs are isolated under `artifacts/figures/predecoder_v2/` and do
  not overwrite the original manuscript figure package.
- Figure 1 through Figure 6 are now available as both PNG and PDF. Figure 3
  through Figure 6 are generated directly from the fixed evaluation JSON
  artifacts:

2026-05-27 Overleaf cleanup:

- Figure 1, Figure 2, and Figure 4 were regenerated with a cleaner card-based
  layout, larger DejaVu Sans text, shorter labels, and safer arrow spacing.
- Long in-figure conclusion sentences were moved into shorter two-line
  callouts to avoid text crossing card boundaries after LaTeX scaling.
- The manuscript tables in `main.tex` and `main_en.tex` were compacted with
  `tabular*`, smaller text, percentage units, and shorter interpretation
  labels to reduce column overflow in XeLaTeX/Overleaf.

2026-05-28 visual redesign:

- Figure 1 was redesigned as a grouped pipeline:
  detector volume -> neural pre-decoder internals -> PyMatching, with the
  selected-mode safety policy shown as a separate callout.
- Figure 2 was redesigned as an architecture diagram that separates the 3D
  residual trunk, trunk heads, candidate set, `CandidateEditSelector`, and
  selected output path.
- Figure 4 was redesigned as a d7 diagnostic dashboard. All segmented-bar
  counts are now explicitly visible, including small segments such as `6` and
  `4`, using outside labels where needed.
- The noise-family table in `main.tex` and `main_en.tex` was promoted to a
  full-width `table*` layout so full stage names can be shown without crowded
  two-column alignment.

```text
artifacts/figures/predecoder_v2/fig1_predecoder_pipeline_v2.png
artifacts/figures/predecoder_v2/fig1_predecoder_pipeline_v2.pdf
artifacts/figures/predecoder_v2/fig2_model_architecture_v2.png
artifacts/figures/predecoder_v2/fig2_model_architecture_v2.pdf
artifacts/figures/predecoder_v2/fig3_main_accuracy_comparison_v2.png
artifacts/figures/predecoder_v2/fig3_main_accuracy_comparison_v2.pdf
artifacts/figures/predecoder_v2/fig4_d7_oracle_gap_false_positive_v2.png
artifacts/figures/predecoder_v2/fig4_d7_oracle_gap_false_positive_v2.pdf
artifacts/figures/predecoder_v2/fig5_d7_validation_heldout_scatter_v2.png
artifacts/figures/predecoder_v2/fig5_d7_validation_heldout_scatter_v2.pdf
artifacts/figures/predecoder_v2/fig6_oracle_recovery_distribution_v2.png
artifacts/figures/predecoder_v2/fig6_oracle_recovery_distribution_v2.pdf
```

Regeneration command:

```bash
python tools/build_predecoder_figures_matplotlib.py
```

The builder uses `matplotlib` with the non-interactive `Agg` backend, so it does
not require a working Tk installation.

V2 figure roles:

| figure | v2 compile input | thesis role |
| --- | --- | --- |
| Figure 1 | `artifacts/figures/predecoder_v2/fig1_predecoder_pipeline_v2.png` | overall neural predecoder plus PyMatching pipeline |
| Figure 2 | `artifacts/figures/predecoder_v2/fig2_model_architecture_v2.png` | 3D trunk, local candidate features, patch-head selector, and safety policy |
| Figure 3 | `artifacts/figures/predecoder_v2/fig3_main_accuracy_comparison_v2.png` | d3/d5/d7 held-out `stage_c_corr` accuracy comparison |
| Figure 4 | `artifacts/figures/predecoder_v2/fig4_d7_oracle_gap_false_positive_v2.png` | d7 oracle headroom, harmful candidate split, and selected fallback |
| Figure 5 | `artifacts/figures/predecoder_v2/fig5_d7_validation_heldout_scatter_v2.png` | d7 validation gain versus held-out candidate gain |
| Figure 6 | `artifacts/figures/predecoder_v2/fig6_oracle_recovery_distribution_v2.png` | seed-level oracle-gap recovery by distance |

## Regeneration Command

```bash
python tools/build_d7_validation_heldout_scatter.py
python tools/build_oracle_recovery_distribution_summary.py
python tools/build_predecoder_figure_package.py
python tools/rasterize_predecoder_figures.py --scale 2
```

Expected output:

```text
num_figures: 6
out_dir: artifacts/figures/predecoder
fig5: artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg
fig6: artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg
png_count: 6
```

## Figure List

| figure | SVG source | PNG compile input | thesis role |
| --- | --- | --- | --- |
| Figure 1 | `artifacts/figures/predecoder/fig1_predecoder_pipeline.svg` | `artifacts/figures/predecoder/fig1_predecoder_pipeline.png` | overall neural predecoder plus PyMatching pipeline |
| Figure 2 | `artifacts/figures/predecoder/fig2_model_architecture.svg` | `artifacts/figures/predecoder/fig2_model_architecture.png` | 3D trunk, local candidate features, patch-head selector, and selected-mode safety |
| Figure 3 | `artifacts/figures/predecoder/fig3_main_accuracy_comparison.svg` | `artifacts/figures/predecoder/fig3_main_accuracy_comparison.png` | d3/d5/d7 held-out `stage_c_corr` accuracy comparison |
| Figure 4 | `artifacts/figures/predecoder/fig4_d7_oracle_gap_false_positive.svg` | `artifacts/figures/predecoder/fig4_d7_oracle_gap_false_positive.png` | d7 oracle headroom, candidate outcome split, validation false positives, and safety fallback |
| Figure 5 | `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg` | `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.png` | d7 validation candidate delta versus held-out candidate delta |
| Figure 6 | `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg` | `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.png` | seed-level selected/candidate recovery of the target local-edit oracle gap |

## Captions

Figure 1. 제안하는 neural pre-decoder와 PyMatching 결합 구조. 36채널
syndrome/noise volume이 3D residual trunk와 local motif selector를 거쳐
selected local edit 또는 raw no-edit syndrome으로 변환되고, 최종 logical
frame은 PyMatching이 예측한다.

Figure 2. SyndromeEditPreDecoder와 patch-head CandidateEditSelector 구조.
3D trunk가 shot-level feature와 detector-level edit logits를 생성하고,
candidate selector가 local patch feature 및 benefit/harm feature를 이용해
candidate edit을 ranking한다.

Figure 3. d3, d5, d7의 held-out `stage_c_corr` accuracy 비교. d3와 d5는
selected predecoder가 raw PyMatching보다 높은 정확도를 보이나, d7은 거의
raw no-edit에 머문다.

Figure 4. d7 candidate-oracle gap과 validation false-positive 구조. 모든 d7
seed에 oracle headroom이 존재하지만, learned candidate branch는 positive,
neutral, harmful outcome으로 분산되며 validation-positive false positive가
주요 실패 유형으로 나타난다. selected-mode fallback은 harmful candidate
seed `17/17`개를 모두 raw no-edit으로 차단한다.

Figure 5. d7 validation delta와 held-out candidate delta의 seed-level scatter.
validation-positive branch `22`개 중 held-out harmful이 `13`개이고 true
positive는 `5`개뿐이므로, 현재 d7 병목이 단순 threshold calibration이 아니라
selector ranking/generalization 문제임을 보여준다.

Figure 6. seed-level oracle-gap recovery 분포. d3는 모든 seed에서 target
local-edit oracle gap의 일부를 회복하고, d5는 두 seed에서만 회복하지만 나머지
seed를 raw no-edit으로 유지한다. d7은 candidate-oracle headroom은 높지만
selected recovery가 거의 0에 머문다.

## Figure Data Check

The figures encode the following fixed values:

| item | value |
| --- | ---: |
| d3 raw / selected / oracle | `0.928710938 / 0.935302734 / 0.992187500` |
| d5 raw / selected / oracle | `0.888671875 / 0.894287109 / 0.978515625` |
| d7 raw / selected / oracle | `0.873046875 / 0.873198411 / 0.984375000` |
| d7 candidate outcomes | positive `6`, neutral `35`, harmful `17` |
| d7 oracle-positive seeds | `58/58` |
| d7 validation-positive held-out outcomes | harmful `13`, neutral `4`, positive `5` |
| d7 validation-positive false-positive ratio | `59.09%` |
| d7 validation-vs-heldout Pearson r | `-0.452872` |
| harmful candidate seeds blocked by selected mode | `17/17` |
| d3 selected oracle recovery mean | `10.38%` |
| d5 selected oracle recovery mean | `6.25%` |
| d7 selected oracle recovery mean | `0.14%` |
| d7 candidate-oracle recovery mean | `86.84%` |

## Thesis Insertion Plan

| chapter | figure |
| --- | --- |
| Chapter 4 Proposed Method | Figure 1 and Figure 2 |
| Chapter 6 Results | Figure 3 |
| Chapter 7 D7 Limitation Analysis | Figure 4, Figure 5, and Figure 6 |

## Remaining Figure Work

- Optionally replace PNGs with same-basename PDFs if the final university
  template requires vector figures.
- Adjust numbering if the school template requires Korean labels such as
  `그림 1`.
- Keep the source SVGs, PNG compile inputs, and summary JSON in the
  reproducibility appendix.
