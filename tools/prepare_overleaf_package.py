"""Prepare a compact Overleaf upload package for the thesis manuscript."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("artifacts/overleaf_predecoder_package")
DEFAULT_ZIP = Path("artifacts/overleaf_predecoder_package.zip")
FIGURE_DIR = Path("artifacts/figures/predecoder_v2")
FIGURE_BASENAMES = [
    "fig1_predecoder_pipeline",
    "fig2_model_architecture",
    "fig3_main_accuracy_comparison",
    "fig4_d7_oracle_gap_false_positive",
    "fig5_d7_validation_heldout_scatter",
    "fig6_oracle_recovery_distribution",
]
ROOT_FILES = [
    Path("main.tex"),
    Path("main.bib"),
    Path("OVERLEAF_COMPILE_GUIDE.md"),
]
OPTIONAL_AUDIT_FILES = [
    Path("artifacts/eval/nn/main_tex_static_check_summary.json"),
    Path("artifacts/eval/nn/sedp_final_result_consistency_check.json"),
    Path("artifacts/figures/predecoder_v2/predecoder_v2_figure_summary.json"),
]


def _copy_file(src: Path, out_dir: Path, copied: list[str], missing: list[str]) -> None:
    if not src.exists():
        missing.append(src.as_posix())
        return
    dest = out_dir / src
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    copied.append(dest.as_posix())


def _write_package_readme(out_dir: Path, main_file: Path) -> None:
    readme = out_dir / "README_OVERLEAF_PACKAGE.md"
    readme.write_text(
        f"""# Overleaf Predecoder Thesis Package

Set the Overleaf compiler to **XeLaTeX** and compile `{main_file.as_posix()}`. Bibliography
entries are in `main.bib`; Overleaf should run BibTeX automatically.

The figure inclusion priority in `{main_file.as_posix()}` is:

```text
PDF -> PNG
```

The package includes generated matplotlib PNG/PDF figures, so Overleaf does not need the
`svg` package, Inkscape, or shell escape for figure conversion. SVG sources may
also be included for auditability, but `{main_file.as_posix()}` does not depend on them.

See `OVERLEAF_COMPILE_GUIDE.md` for the full checklist.
""",
        encoding="utf-8",
    )


def _zip_dir(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))


def build_package(out_dir: Path, zip_path: Path, main_file: Path = Path("main.tex")) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []

    root_files = [main_file, Path("main.bib"), Path("OVERLEAF_COMPILE_GUIDE.md")]
    for src in root_files:
        _copy_file(src, out_dir, copied, missing_required)

    for basename in FIGURE_BASENAMES:
        found_latex_ready = False
        for suffix in (".pdf", ".png", ".svg"):
            candidates = [
                FIGURE_DIR / f"{basename}_v2{suffix}",
                FIGURE_DIR / f"{basename}{suffix}",
            ]
            for src in candidates:
                if src.exists():
                    _copy_file(src, out_dir, copied, missing_required)
                    if suffix in {".pdf", ".png"}:
                        found_latex_ready = True
        if not found_latex_ready:
            missing_required.append((FIGURE_DIR / f"{basename}_v2.png").as_posix())

    optional_audit_files = list(OPTIONAL_AUDIT_FILES)
    if main_file.name == "main_en.tex":
        optional_audit_files.insert(0, Path("artifacts/eval/nn/main_en_tex_static_check_summary.json"))
    for src in optional_audit_files:
        _copy_file(src, out_dir, copied, missing_optional)

    _write_package_readme(out_dir, main_file)
    copied.append((out_dir / "README_OVERLEAF_PACKAGE.md").as_posix())

    summary = {
        "schema_version": "overleaf_predecoder_package.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": out_dir.as_posix(),
        "zip_path": zip_path.as_posix(),
        "compiler": "XeLaTeX",
        "main_file": main_file.as_posix(),
        "figure_priority": ["pdf", "png"],
        "copied_files": copied,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "pass": not missing_required,
    }
    summary_path = out_dir / "overleaf_package_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _zip_dir(out_dir, zip_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--main-file", type=Path, default=Path("main.tex"))
    args = parser.parse_args()

    summary = build_package(args.out_dir, args.zip, args.main_file)
    print(
        json.dumps(
            {
                "pass": summary["pass"],
                "out_dir": summary["out_dir"],
                "zip_path": summary["zip_path"],
                "num_files": len(summary["copied_files"]),
                "missing_required": summary["missing_required"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
