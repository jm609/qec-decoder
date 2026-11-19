# Overleaf compile guide

The current manuscripts are `main_d7_en.tex` and `main_d7_ko.tex`.
Both contain an inline bibliography and use the five frozen PDFs in
`artifacts/figures/d7_paper_evidence/20261117_final/`.
Those figure assets are tracked so a fresh clone can build a package.

## Create a package

From the repository root in the Python environment:

```powershell
python tools/prepare_overleaf_package.py --main-file main_d7_en.tex --out-dir artifacts/overleaf_d7_en --zip artifacts/overleaf_d7_en.zip
python tools/prepare_overleaf_package.py --main-file main_d7_ko.tex --out-dir artifacts/overleaf_d7_ko --zip artifacts/overleaf_d7_ko.zip
```

Use new output paths to retain an earlier package. The ZIP preserves the
repository-relative figure paths used by the manuscript.

## Overleaf settings

1. Upload the ZIP as a project.
2. Set the main document to `main_d7_en.tex` or `main_d7_ko.tex`.
3. Select **XeLaTeX**.
4. Use **Recompile from scratch** when replacing a previous project.

No external `.bib` file, model checkpoint, or dataset is needed.
The Korean version uses `kotex`. The English version uses `microtype`
character protrusion with font expansion disabled for XeTeX.

Keep `\usepackage{array}[=2016-10-06]` before the table-related packages.
This retains the older array interface required by REVTeX 4.2 on recent
TeX Live releases. Removing it previously caused table compilation errors.

## Warnings and verification

The English typesetting revision makes long numeric lists breakable, reflows
paragraphs identified in the supplied log, and aligns bibliography entries
ragged right. Diagnostic warnings are not suppressed.

A `nameref` warning about a changed `\label` definition may remain with REVTeX
and recent LaTeX. Check reference numbers and PDF links. An `Underfull \hbox`
message reports loose line spacing; inspect paragraphs with high badness.

The supplied log for the previous English package completed 19 XeTeX pages
without a fatal error. The typesetting revision has static checks, but its
exact warning count and rendered layout have not been verified locally.

- [Overleaf explanation of box warnings](https://www.overleaf.com/learn/how-to/Understanding_underfull_and_overfull_box_warnings)
- [Overleaf compatibility notes](https://www.overleaf.com/blog/tex-live-2025-is-now-available)
- [Microtype engine support](https://ctan.org/pkg/microtype)

## Earlier manuscripts

`main.tex` and `main_en.tex` are earlier drafts using `main.bib` and a separate
six-figure package. The script retains that route when a legacy main file is
supplied. Its figures are local generated artifacts, separate from the current
five-figure publication set.
