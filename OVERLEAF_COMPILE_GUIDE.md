# Overleaf Compile Guide

This is the upload and compile checklist for the active thesis manuscripts:

- `main.tex`: polished Korean draft with English title.
- `main_en.tex`: English submission draft.

## Compiler

Use **XeLaTeX** on Overleaf.

Reason:

- `main.tex` uses `kotex` for Korean text.
- `main_en.tex` is English-only, but XeLaTeX is still safe and consistent with
  the Korean package.
- Figures are included with `graphicx` from generated PDF/PNG files, so
  Overleaf does not need the `svg` package, Inkscape, or shell escape.

English draft clean-log notes:

- `main_en.tex` uses `fontspec` with TeX Gyre Termes under XeLaTeX instead of
  the older `mathptmx` text font path. This avoids `TU/ptm` font-shape
  substitution warnings.
- `hyperref` is intentionally omitted from `main_en.tex` because the
  `revtex4-2`/`nameref` label-hook warning is harmless but noisy on recent
  Overleaf TeX Live versions. Cross-references and citations still compile
  normally, but they are not PDF hyperlinks.
- Long file paths in the reproducibility note are summarized rather than kept
  as unbreakable inline `\texttt{...}` strings.

## Files To Upload

For the Korean draft, use:

```text
artifacts/overleaf_predecoder_package.zip
```

For the English draft, use:

```text
artifacts/overleaf_predecoder_package_en.zip
```

The package preserves this structure:

```text
main.tex or main_en.tex
main.bib
artifacts/
  figures/
    predecoder_v2/
      fig1_predecoder_pipeline_v2.pdf
      fig1_predecoder_pipeline_v2.png
      ...
      fig6_oracle_recovery_distribution_v2.pdf
      fig6_oracle_recovery_distribution_v2.png
```

The manuscript figure macro uses this priority:

```text
PDF first -> PNG second
```

The SVG files may also be present in the package as editable source figures,
but the LaTeX file does not compile through SVG.

## Package Command

From the repository root:

```powershell
python tools/build_predecoder_figures_matplotlib.py
python tools/prepare_overleaf_package.py
python tools/prepare_overleaf_package.py --out-dir artifacts/overleaf_predecoder_package_en --zip artifacts/overleaf_predecoder_package_en.zip --main-file main_en.tex
```

This creates:

```text
artifacts/overleaf_predecoder_package/
artifacts/overleaf_predecoder_package.zip
artifacts/overleaf_predecoder_package_en/
artifacts/overleaf_predecoder_package_en.zip
```

Upload the appropriate zip to Overleaf, set the compiler to XeLaTeX, and
compile `main.tex` or `main_en.tex`. Overleaf should run BibTeX automatically
after the first LaTeX pass because references are stored in `main.bib`.

## Before Upload

Run:

```powershell
python tools/check_main_tex_static.py
python tools/check_main_tex_static.py --main-tex main_en.tex --figure-dir artifacts\figures\predecoder_v2 --out artifacts\eval\nn\main_en_tex_static_check_summary.json
python tools/build_final_result_consistency_summary.py
python -m unittest discover -s tests -v
```

Expected status:

- `main.tex` static check: `pass=True`, failed errors `0`, failed warnings `0`
- `main_en.tex` static check: `pass=True`, failed errors `0`, failed warnings `0`
- final result consistency: `37/37` pass
- regression tests: `18/18 OK`

## If Compilation Still Fails

Check these items first:

- Overleaf compiler is set to XeLaTeX, not pdfLaTeX.
- `artifacts/figures/predecoder_v2/*_v2.pdf` or `*_v2.png` exists in the
  Overleaf file tree.
- `main.bib` exists at the Overleaf project root.
- The selected main file is at the Overleaf project root, not inside a nested
  folder.
- The uploaded zip is the regenerated
  `artifacts/overleaf_predecoder_package.zip` or
  `artifacts/overleaf_predecoder_package_en.zip`.
