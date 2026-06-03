"""Static checks for the active thesis manuscript.

This is a lightweight substitute for a LaTeX compile check on machines where
pdflatex/xelatex are not installed. It checks references, citations, required
paper-facing labels, figure source availability, and stale evaluation wording.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MAIN_TEX = Path("main.tex")
DEFAULT_FIGURE_DIR = Path("artifacts/figures/predecoder_v2")
DEFAULT_OUTPUT = Path("artifacts/eval/nn/main_tex_static_check_summary.json")

REQUIRED_LABELS = {
    "tab:related-work",
    "tab:noise-family",
    "fig:pipeline",
    "fig:architecture",
    "tab:main-results",
    "fig:main-accuracy",
    "tab:d3d5-robustness",
    "tab:paired-stats",
    "tab:d5-fallback",
    "fig:d7-oracle",
    "tab:d7-mismatch",
    "fig:d7-scatter",
    "fig:oracle-recovery",
}

REQUIRED_BIBITEMS = {
    "pymatching",
    "stim",
    "surfacecode",
    "alphaqubit",
    "nvidiaising",
    "neartermnn",
    "willowthreshold",
}

REQUIRED_SECTIONS = {
    "서론",
    "배경 및 관련 연구",
    "데이터셋 및 노이즈 환경",
    "제안 방법",
    "실험 설정",
    "실험 결과",
    "d7 한계 분석",
    "논의",
    "결론",
}

ENGLISH_REQUIRED_SECTIONS = {
    "Introduction",
    "Background and Related Work",
    "Dataset and Noise Setting",
    "Proposed Method",
    "Experimental Setup",
    "Results",
    "D7 Limitation Analysis",
    "Discussion",
    "Conclusion",
}

STALE_PATTERNS = {
    "old_evaluation_score": r"8\.4\s*/\s*10",
    "d5_uniformly_positive_claim": r"d5[^.\n]{0,120}uniformly positive",
    "stage_de_vague_not_included": r"Stage D[^.\n]{0,80}not quantitatively evaluated",
    "strong_d5_significance_korean": r"d5[^.\n]{0,80}통계적으로 유의",
}


def _find_all(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text, flags=re.MULTILINE)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _read_braced_groups(text: str, offset: int, count: int) -> list[str] | None:
    groups: list[str] = []
    pos = offset
    for _ in range(count):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            return None
        pos += 1
        start = pos
        depth = 1
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth:
            return None
        groups.append(text[start : pos - 1])
    return groups


def _fixed_display_labels(text: str) -> set[str]:
    labels: set[str] = set()
    for command in ("fixedwidefigure",):
        for match in re.finditer(rf"\\{command}", text):
            groups = _read_braced_groups(text, match.end(), 3)
            if groups:
                labels.add(groups[1].strip())
    for env in ("fixedtable", "fixedwidetable"):
        for match in re.finditer(rf"\\begin\{{{env}\}}", text):
            groups = _read_braced_groups(text, match.end(), 2)
            if groups:
                labels.add(groups[0].strip())
    return labels


def _bib_database_keys(main_tex: Path, text: str) -> set[str]:
    keys: set[str] = set()
    for group in _find_all(r"\\bibliography\{([^}]+)\}", text):
        for raw_name in group.split(","):
            name = raw_name.strip()
            if not name:
                continue
            bib_path = main_tex.parent / f"{name}.bib"
            if not bib_path.exists():
                continue
            bib_text = bib_path.read_text(encoding="utf-8")
            keys.update(_find_all(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bib_text))
    return keys


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    severity: str = "error",
    detail: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "severity": severity,
            "pass": bool(passed),
            "detail": detail,
        }
    )


def build_summary(main_tex: Path, figure_dir: Path) -> dict[str, Any]:
    text = main_tex.read_text(encoding="utf-8")

    labels = set(_find_all(r"\\label\{([^}]+)\}", text)) | _fixed_display_labels(text)
    refs = set(_find_all(r"\\(?:ref|pageref)\{([^}]+)\}", text))
    raw_cites = _find_all(r"\\cite\{([^}]+)\}", text)
    cites = {key.strip() for group in raw_cites for key in group.split(",") if key.strip()}
    bibitems = set(_find_all(r"\\bibitem\{([^}]+)\}", text)) | _bib_database_keys(main_tex, text)
    figure_basenames = _find_all(r"\\predecoderfigure\{([^}]+)\}", text)
    sections = set(_find_all(r"\\section\*?\{([^}]+)\}", text))

    checks: list[dict[str, Any]] = []

    missing_refs = sorted(refs - labels)
    _check(checks, "all_references_have_labels", not missing_refs, detail=missing_refs)

    missing_cites = sorted(cites - bibitems)
    _check(checks, "all_citations_have_bibitems", not missing_cites, detail=missing_cites)

    missing_labels = sorted(REQUIRED_LABELS - labels)
    _check(checks, "required_labels_present", not missing_labels, detail=missing_labels)

    missing_bibitems = sorted(REQUIRED_BIBITEMS - bibitems)
    _check(checks, "required_bibitems_present", not missing_bibitems, detail=missing_bibitems)

    missing_korean_sections = sorted(REQUIRED_SECTIONS - sections)
    missing_english_sections = sorted(ENGLISH_REQUIRED_SECTIONS - sections)
    sections_pass = not missing_korean_sections or not missing_english_sections
    _check(
        checks,
        "required_sections_present",
        sections_pass,
        detail={
            "missing_korean_sections": missing_korean_sections,
            "missing_english_sections": missing_english_sections,
        },
    )

    figure_rows: list[dict[str, Any]] = []
    missing_figure_assets: list[str] = []
    missing_latex_ready_figures: list[str] = []
    missing_pdf_png_fallbacks: list[str] = []
    for basename in figure_basenames:
        svg_paths = [
            figure_dir / f"{basename}.svg",
            figure_dir / f"{basename}_v2.svg",
        ]
        pdf_paths = [
            figure_dir / f"{basename}.pdf",
            figure_dir / f"{basename}_v2.pdf",
        ]
        png_paths = [
            figure_dir / f"{basename}.png",
            figure_dir / f"{basename}_v2.png",
        ]
        row = {
            "basename": basename,
            "svg": any(path.exists() for path in svg_paths),
            "pdf": any(path.exists() for path in pdf_paths),
            "png": any(path.exists() for path in png_paths),
        }
        figure_rows.append(row)
        if not row["svg"] and not row["pdf"] and not row["png"]:
            missing_figure_assets.append(basename)
        if not row["pdf"] and not row["png"] and not row["svg"]:
            missing_latex_ready_figures.append(basename)
        if not row["pdf"] and not row["png"]:
            missing_pdf_png_fallbacks.append(basename)

    _check(
        checks,
        "predecoder_figure_sources_exist",
        not missing_figure_assets,
        detail=missing_figure_assets,
    )
    _check(
        checks,
        "predecoder_figures_have_latex_input",
        not missing_latex_ready_figures,
        detail=missing_latex_ready_figures,
    )
    _check(
        checks,
        "predecoder_figures_missing_pdf_png_fallback",
        not missing_pdf_png_fallbacks,
        severity="warning",
        detail=missing_pdf_png_fallbacks,
    )

    uses_svg_package = "\\usepackage{svg}" in text or "\\includesvg" in text
    has_latex_ready_figures = not missing_pdf_png_fallbacks
    _check(
        checks,
        "overleaf_uses_graphicx_ready_figures",
        has_latex_ready_figures and not uses_svg_package,
        detail=(
            "main.tex should compile on Overleaf without SVG/Inkscape by using "
            "PDF/PNG figure files through graphicx."
        ),
    )
    fragile_table_matches = []
    for pattern in (r"\\begin\{tabular\}\{[^\n]*p\{", r"\\arraybackslash", r"\\usepackage\{array\}"):
        for match in re.finditer(pattern, text, flags=re.MULTILINE):
            fragile_table_matches.append(
                {
                    "line": _line_number(text, match.start()),
                    "text": match.group(0),
                }
            )
    _check(
        checks,
        "no_overleaf_fragile_table_column_specs",
        not fragile_table_matches,
        detail=fragile_table_matches,
    )

    stale_matches: dict[str, list[dict[str, Any]]] = {}
    for name, pattern in STALE_PATTERNS.items():
        matches = []
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            matches.append(
                {
                    "line": _line_number(text, match.start()),
                    "text": match.group(0),
                }
            )
        stale_matches[name] = matches
        _check(checks, f"no_stale_phrase.{name}", not matches, detail=matches)

    required_cross_paper_cites = {
        "alphaqubit",
        "nvidiaising",
        "neartermnn",
        "willowthreshold",
    }
    missing_cross_paper_cites = sorted(required_cross_paper_cites - cites)
    _check(
        checks,
        "cross_paper_table_cites_required_sources",
        not missing_cross_paper_cites,
        detail=missing_cross_paper_cites,
    )

    required_terms = {
        "candidate safety policy": ("후보 우선 안전", "safety rule", "adoption rule"),
        "stage abc scope": ("Stage A",),
        "structured noise scope": ("correlated stray",),
        "d5 fallback table": ("tab:d5-fallback",),
    }
    missing_terms = sorted(
        key for key, terms in required_terms.items() if not any(term in text for term in terms)
    )
    _check(checks, "required_claim_boundary_terms_present", not missing_terms, detail=missing_terms)

    failed_errors = [
        check["name"]
        for check in checks
        if check["severity"] == "error" and not check["pass"]
    ]
    failed_warnings = [
        check["name"]
        for check in checks
        if check["severity"] == "warning" and not check["pass"]
    ]

    return {
        "schema_version": "main_tex_static_check.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "main_tex": str(main_tex),
        "figure_dir": str(figure_dir),
        "pass": not failed_errors,
        "num_checks": len(checks),
        "num_failed_errors": len(failed_errors),
        "num_failed_warnings": len(failed_warnings),
        "failed_errors": failed_errors,
        "failed_warnings": failed_warnings,
        "counts": {
            "labels": len(labels),
            "refs": len(refs),
            "cites": len(cites),
            "bibitems": len(bibitems),
            "predecoder_figures": len(figure_basenames),
        },
        "figures": figure_rows,
        "stale_matches": stale_matches,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(args.main_tex, args.figure_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"main.tex static check: pass={summary['pass']} "
        f"failed_errors={summary['num_failed_errors']} "
        f"failed_warnings={summary['num_failed_warnings']} "
        f"out={args.out}"
    )
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
