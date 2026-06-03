"""Check final Markdown result tables against consolidated evidence JSON."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONSOLIDATED = Path(
    "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json"
)
DEFAULT_RESULT_TABLES = Path("PREDECODER_FINAL_RESULT_TABLES.md")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Missing section {marker!r}")
    next_start = text.find("\n## ", start + len(marker))
    if next_start < 0:
        return text[start:]
    return text[start:next_start]


def _table_rows(section_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    if rows and not rows[0][0].startswith("d") and rows[0][0] not in {"metric"}:
        rows = rows[1:]
    elif rows and rows[0][0] in {"distance", "metric"}:
        rows = rows[1:]
    return rows


def _clean(value: str) -> str:
    return value.replace("`", "").replace(",", "").strip()


def _parse_float(value: str) -> float:
    cleaned = _clean(value)
    cleaned = cleaned.replace("+", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return float(cleaned)


def _parse_int_pair(value: str) -> tuple[int, int]:
    cleaned = _clean(value)
    match = re.search(r"(\d+)\s*/\s*(\d+)", cleaned)
    if not match:
        raise ValueError(f"Expected int pair in {value!r}")
    return int(match.group(1)), int(match.group(2))


def _mode_counts(value: str) -> dict[str, int]:
    cleaned = _clean(value)
    counts: dict[str, int] = {}
    local = re.search(r"local selector\s+(\d+)/(\d+)", cleaned)
    if local:
        counts["local_motif_selector"] = int(local.group(1))
    raw = re.search(r"raw no-edit\s+(\d+)/(\d+)", cleaned)
    if raw:
        counts["raw_no_edit"] = int(raw.group(1))
    return counts


def _check_float(
    checks: list[dict[str, Any]],
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    delta = observed - expected
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "delta": delta,
            "tolerance": tolerance,
            "pass": abs(delta) <= tolerance,
        }
    )


def _check_equal(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "pass": observed == expected,
        }
    )


def _distance_results(consolidated: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["distance"]): row for row in consolidated["distance_results"]}


def _paper_results(consolidated: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["distance"]): row for row in consolidated["paper_result_table"]}


def build_summary(consolidated_path: Path, result_tables_path: Path) -> dict[str, Any]:
    consolidated = _load_json(consolidated_path)
    text = result_tables_path.read_text(encoding="utf-8")
    paper = _paper_results(consolidated)
    by_distance = _distance_results(consolidated)

    checks: list[dict[str, Any]] = []

    main_rows = _table_rows(_section(text, "Main Accuracy Table"))
    for row in main_rows:
        distance = row[0]
        expected = paper[distance]
        _check_equal(checks, f"{distance}.seeds", _clean(row[1]), str(expected["seeds"]))
        _check_float(
            checks,
            f"{distance}.raw_pymatching_accuracy",
            _parse_float(row[2]),
            float(expected["raw_pymatching_accuracy"]),
            5e-9,
        )
        _check_float(
            checks,
            f"{distance}.selected_predecoder_accuracy",
            _parse_float(row[3]),
            float(expected["selected_predecoder_accuracy"]),
            5e-9,
        )
        _check_float(
            checks,
            f"{distance}.candidate_branch_accuracy",
            _parse_float(row[4]),
            float(expected["candidate_branch_accuracy"]),
            5e-9,
        )
        _check_float(
            checks,
            f"{distance}.target_local_edit_oracle_accuracy",
            _parse_float(row[5]),
            float(expected["target_local_edit_oracle_accuracy"]),
            5e-9,
        )
        _check_float(
            checks,
            f"{distance}.selected_delta_over_raw",
            _parse_float(row[6]),
            float(expected["selected_delta_over_raw"]),
            5e-9,
        )
        _check_float(
            checks,
            f"{distance}.oracle_recovery_percent",
            _parse_float(row[7]),
            float(expected["selected_oracle_recovery_fraction"]) * 100.0,
            0.01,
        )

    behavior_rows = _table_rows(_section(text, "Selected-Mode Behavior"))
    for row in behavior_rows:
        distance = row[0]
        expected = by_distance[distance]
        _check_equal(
            checks,
            f"{distance}.selected_modes",
            _mode_counts(row[1]),
            dict(expected["mode_counts"]),
        )
        _check_equal(
            checks,
            f"{distance}.selected_improved_harmed",
            _parse_int_pair(row[2]),
            (
                int(expected["selected_improved_total"]),
                int(expected["selected_harmed_total"]),
            ),
        )
        _check_equal(
            checks,
            f"{distance}.candidate_improved_harmed",
            _parse_int_pair(row[3]),
            (
                int(expected["candidate_improved_total"]),
                int(expected["candidate_harmed_total"]),
            ),
        )

    d7_gap = consolidated["d7_oracle_gap"]
    d7_distance = by_distance["d7"]
    d7_metrics = {
        row[0]: row[1] for row in _table_rows(_section(text, "D7 Oracle Gap"))
    }
    _check_float(
        checks,
        "d7_gap.mean_selected_delta",
        _parse_float(d7_metrics["mean selected delta"]),
        float(d7_gap["mean_selected_delta_over_no_edit"]),
        5e-9,
    )
    _check_float(
        checks,
        "d7_gap.mean_actual_candidate_delta",
        _parse_float(d7_metrics["mean actual candidate delta"]),
        float(d7_gap["mean_candidate_delta_over_no_edit"]),
        5e-9,
    )
    _check_float(
        checks,
        "d7_gap.target_local_edit_oracle_delta",
        _parse_float(d7_metrics["target local-edit oracle delta"]),
        float(d7_distance["target_local_edit_oracle_delta"]),
        5e-9,
    )
    _check_float(
        checks,
        "d7_gap.learned_candidate_oracle_delta",
        _parse_float(d7_metrics["learned candidate-oracle delta"]),
        float(d7_gap["mean_candidate_oracle_delta_over_no_edit"]),
        5e-9,
    )
    _check_float(
        checks,
        "d7_gap.candidate_to_oracle_gap",
        _parse_float(d7_metrics["candidate-to-oracle gap"]),
        float(d7_gap["mean_candidate_to_oracle_gap"]),
        5e-9,
    )

    actual_candidate = d7_metrics["actual candidate outcomes"]
    _check_equal(
        checks,
        "d7_gap.actual_candidate_outcomes",
        {
            "positive": int(re.search(r"positive `?(\d+)", actual_candidate).group(1)),
            "neutral": int(re.search(r"neutral `?(\d+)", actual_candidate).group(1)),
            "harmful": int(re.search(r"harmful `?(\d+)", actual_candidate).group(1)),
        },
        dict(d7_gap["candidate_delta_class_counts"]),
    )
    oracle_outcomes = d7_metrics["oracle outcomes"]
    _check_equal(
        checks,
        "d7_gap.oracle_outcomes",
        {"positive": int(re.search(r"positive `?(\d+)", oracle_outcomes).group(1))},
        dict(d7_gap["oracle_delta_class_counts"]),
    )

    failed = [check for check in checks if not check["pass"]]
    return {
        "schema_version": "predecoder_final_result_consistency.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "consolidated_evidence": consolidated_path.as_posix(),
        "result_tables": result_tables_path.as_posix(),
        "num_checks": len(checks),
        "num_failed": len(failed),
        "pass": not failed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, default=DEFAULT_CONSOLIDATED)
    parser.add_argument("--result-tables", type=Path, default=DEFAULT_RESULT_TABLES)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_final_result_consistency_check.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.consolidated, args.result_tables)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "pass": summary["pass"],
                "num_checks": summary["num_checks"],
                "num_failed": summary["num_failed"],
            },
            indent=2,
        )
    )
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
