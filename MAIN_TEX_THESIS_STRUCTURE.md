# Main.tex Thesis Structure

This document records the logical flow now implemented in `main.tex`.

## 2026-05-28 Thesis File Status

- The submitted English title is fixed and must be kept:
  `Design and Evaluation of a Transition-Aware Neural Pre-Decoder for Surface-Code Quantum Error Correction`.
- `main.tex` is the active Korean polished draft.
- `main_en.tex` is the active English polished draft.
- Both drafts target RevTeX/XeLaTeX and use the same bibliography and figure
  package.
- `showkeys` has been removed for final-output safety.
- The Korean draft now uses `surface code` consistently for the code name.
- Both drafts clarify that `logical_class4`, logical frame, and logical
  decision refer to the same four-class PyMatching logical decision target.
- Both drafts clarify that held-out `stage_c_corr` local-edit search is post
  hoc oracle analysis only and is not used for training, motif selection,
  threshold selection, or seed-level adoption.

## Role of `main.tex`

`main.tex` is now the working LaTeX manuscript file. The previous neutron
detector example text has been replaced with the transition-aware neural
predecoder thesis content.

2026-05-27 manuscript versioning update:

- `legacy_archive/main_ko_reference.tex` preserves the Korean manuscript state before the
  final style-polish pass.
- `main.tex` is the active polished Korean draft with the English thesis title
  and v2 figure package.
- `main_en.tex` is the English submission draft translated and rewritten from
  the polished structure, rather than overwriting the preserved Korean
  reference.

The file keeps the original `revtex4-2` two-column style, but adds packages
needed for the current manuscript:

- `kotex` for Korean text
- `amsmath` for math expressions
- `booktabs` for tables
- `graphicx` for figures
- `xurl` for URL line breaking
- `geometry` for A4 thesis margins

## Logical Flow

The manuscript is organized around one central argument:

> Neural pre-decoding is useful as a PyMatching front-end at d3/d5, while d7
> reveals a selector-ranking and generalization bottleneck rather than a lack
> of local-edit candidates.

## Section Plan

| section | role in argument |
| --- | --- |
| Abstract | States the hybrid predecoder structure, d3/d5 gains, and d7 selector limitation |
| Introduction | Introduces surface-code decoding and motivates pre-decoding instead of replacing PyMatching |
| Background | Explains surface-code syndrome decoding, PyMatching, and why direct neural replacement is not the final path |
| Related-work context table | Places this work against AlphaQubit, NVIDIA Ising-Decoding, near-term neural decoding, and Google Willow decoding context without claiming head-to-head equivalence |
| Dataset and Noise Setting | Fixes Stage A/B training-validation and Stage C held-out evaluation boundary |
| Proposed Method | Describes the 36-channel input, 3D trunk, local motif candidates, benefit/harm selector, and safety fallback |
| Experimental Setup | Defines raw PyMatching, selected predecoder, candidate branch, oracle, and held-out metric |
| Results | Presents d3/d5/d7 main table, d3/d5 robustness, paired statistics, the d5 seed-level fallback table, and selected-vs-candidate behavior |
| D7 Limitation Analysis | Shows oracle headroom, validation-to-held-out mismatch, harmful candidates, and rejected d7 paths |
| Discussion | States claim boundaries: d3/d5 selected-mode improvement, conservative d5 adoption, d7 selector generalization limitation, and Stage A/B/C noise-family scope |
| Conclusion | Summarizes d3/d5 success and d7 selector bottleneck |
| Reproducibility Note | Points to evidence documents and the final consistency check |

## Figures

The active manuscript figures are the matplotlib v2 outputs under
`artifacts/figures/predecoder_v2/`. `main.tex` now uses the
`\predecoderfigure` macro. For each figure, the macro includes a v2 `.pdf`
first if present and otherwise includes the v2 `.png`. The older
`artifacts/figures/predecoder/` package remains as a fallback only.

Figures mapped in `main.tex`:

- Figure 1: `fig1_predecoder_pipeline_v2.png`
- Figure 2: `fig2_model_architecture_v2.png`
- Figure 3: `fig3_main_accuracy_comparison_v2.png`
- Figure 4: `fig4_d7_oracle_gap_false_positive_v2.png`
- Figure 5: `fig5_d7_validation_heldout_scatter_v2.png`
- Figure 6: `fig6_oracle_recovery_distribution_v2.png`

Before final PDF submission, compile on Overleaf with XeLaTeX. The current v2
package already includes both PDF and PNG outputs.

V2 figure generation helper:

```powershell
python tools/build_predecoder_figures_matplotlib.py
```

The helper uses matplotlib with the non-interactive `Agg` backend and does not
require Inkscape, SVG conversion, or shell escape.

Overleaf package helper:

```powershell
python tools/prepare_overleaf_package.py
python tools/prepare_overleaf_package.py --out-dir artifacts/overleaf_predecoder_package_en --zip artifacts/overleaf_predecoder_package_en.zip --main-file main_en.tex
```

This creates the Korean package `artifacts/overleaf_predecoder_package.zip` and
the English package `artifacts/overleaf_predecoder_package_en.zip`. Upload the
appropriate zip to Overleaf, set the compiler to XeLaTeX, and compile
`main.tex` or `main_en.tex`.

Static manuscript check:

```powershell
python tools/check_main_tex_static.py
python tools/check_main_tex_static.py --main-tex main_en.tex --figure-dir artifacts\figures\predecoder_v2 --out artifacts\eval\nn\main_en_tex_static_check_summary.json
```

This verifies labels, references, citations, required figure sources, required
claim-boundary terms, and stale evaluation-report wording. It is not a
replacement for final PDF compilation, but it catches the main manuscript
integration errors while `pdflatex`/`xelatex` are unavailable.

## Claim Boundary Preserved

The current `main.tex` keeps the evaluation-report constraints:

- d3 is stated as uniformly positive over seed `0..7`.
- d5 is stated as positive-mean and selected-safe, not uniformly positive.
- d5 has a seed-level fallback table showing that seeds `2` and `3` supply the
  selected-mode gain while harmful candidate-only seeds `4` and `6` are blocked.
- d7 is stated as a controlled scaling limitation, not solved recovery.
- Noise-family generalization is limited to Stage A/B training-validation to
  Stage C held-out evaluation.
- Stage D/E details are not part of the main quantitative claim in the current
  manuscript wording. Keep the paper focused on Stage A/B/C unless the user
  explicitly asks for implementation-scope appendix material.

## Remaining LaTeX Work

1. Confirm the school-required compiler and class options.
2. Compile the Overleaf package with XeLaTeX.
3. Confirm `main.tex` picks up PNG figures through `\predecoderfigure`.
4. Add final author/affiliation metadata if required.
5. Compile and fix typography, table width, and figure placement.
6. Add any school-required cover page, acknowledgments, or Korean abstract
   format.
7. Verify the cross-paper comparison bibliography formatting during final
   compile.
