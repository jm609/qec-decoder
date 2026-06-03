"""Build a taxonomy of harmful d7 candidate-edit behavior."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = Path("artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Expected a rows list")
    return [row for row in rows if isinstance(row, dict)]


def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def _safe_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None:
        return default
    return int(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": float(min(values)), "max": float(max(values))}


def _seed_list(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))]


def _top_rows(rows: list[dict[str, Any]], key: str, *, reverse: bool, limit: int = 8) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: _safe_float(row, key), reverse=reverse)
    out = []
    for row in sorted_rows[:limit]:
        out.append(
            {
                "seed": int(row["seed"]),
                "candidate_delta_over_no_edit": _safe_float(row, "candidate_delta_over_no_edit"),
                "validation_delta_over_no_edit": _safe_float(row, "validation_delta_over_no_edit"),
                "candidate_oracle_delta_over_no_edit": _safe_float(
                    row, "candidate_oracle_delta_over_no_edit"
                ),
                "candidate_to_oracle_gap": _safe_float(row, "candidate_to_oracle_gap"),
                "candidate_improved": _safe_int(row, "candidate_improved"),
                "candidate_harmed": _safe_int(row, "candidate_harmed"),
                "candidate_fraction_with_any_selected_edit": _safe_float(
                    row, "candidate_fraction_with_any_selected_edit"
                ),
                "candidate_mean_selected_edit_weight": _safe_float(
                    row, "candidate_mean_selected_edit_weight"
                ),
                "adoption_margin": _safe_float(row, "adoption_margin"),
                "adoption_nonzero": _safe_int(row, "adoption_nonzero"),
                "adoption_reason": str(row.get("adoption_reason")),
                "validation_delta_class": str(row.get("validation_delta_class")),
                "candidate_delta_class": str(row.get("candidate_delta_class")),
                "selected_mode": str(row.get("selected_mode")),
            }
        )
    return out


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "seeds": _seed_list(rows),
        "mean_candidate_delta_over_no_edit": _mean(
            [_safe_float(row, "candidate_delta_over_no_edit") for row in rows]
        ),
        "mean_validation_delta_over_no_edit": _mean(
            [_safe_float(row, "validation_delta_over_no_edit") for row in rows]
        ),
        "mean_candidate_oracle_delta_over_no_edit": _mean(
            [_safe_float(row, "candidate_oracle_delta_over_no_edit") for row in rows]
        ),
        "mean_candidate_to_oracle_gap": _mean(
            [_safe_float(row, "candidate_to_oracle_gap") for row in rows]
        ),
        "mean_candidate_improved": _mean([float(_safe_int(row, "candidate_improved")) for row in rows]),
        "mean_candidate_harmed": _mean([float(_safe_int(row, "candidate_harmed")) for row in rows]),
        "mean_candidate_fraction_with_any_selected_edit": _mean(
            [_safe_float(row, "candidate_fraction_with_any_selected_edit") for row in rows]
        ),
        "mean_candidate_selected_edit_weight": _mean(
            [_safe_float(row, "candidate_mean_selected_edit_weight") for row in rows]
        ),
        "validation_delta_range": _range(
            [_safe_float(row, "validation_delta_over_no_edit") for row in rows]
        ),
        "candidate_delta_range": _range(
            [_safe_float(row, "candidate_delta_over_no_edit") for row in rows]
        ),
        "adoption_reason_counts": dict(Counter(str(row.get("adoption_reason")) for row in rows)),
        "selected_mode_counts": dict(Counter(str(row.get("selected_mode")) for row in rows)),
        "top_by_harm": _top_rows(rows, "candidate_delta_over_no_edit", reverse=False),
        "top_by_validation_delta": _top_rows(rows, "validation_delta_over_no_edit", reverse=True),
        "top_by_oracle_delta": _top_rows(rows, "candidate_oracle_delta_over_no_edit", reverse=True),
    }


def _taxonomy_entry(
    *,
    name: str,
    definition: str,
    rows: list[dict[str, Any]],
    interpretation: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "definition": definition,
        "interpretation": interpretation,
        "stats": _group_stats(rows),
    }


def build_summary(input_path: Path) -> dict[str, Any]:
    data = _load_json(input_path)
    rows = _rows(data)
    harmful = [row for row in rows if row.get("candidate_delta_class") == "harmful"]
    positive = [row for row in rows if row.get("candidate_delta_class") == "positive"]
    neutral = [row for row in rows if row.get("candidate_delta_class") == "neutral"]
    validation_positive = [row for row in rows if row.get("validation_delta_class") == "positive"]
    validation_false_positive = [
        row
        for row in rows
        if row.get("validation_delta_class") == "positive"
        and row.get("candidate_delta_class") == "harmful"
    ]
    validation_true_positive = [
        row
        for row in rows
        if row.get("validation_delta_class") == "positive"
        and row.get("candidate_delta_class") == "positive"
    ]
    validation_positive_neutral = [
        row
        for row in rows
        if row.get("validation_delta_class") == "positive"
        and row.get("candidate_delta_class") == "neutral"
    ]
    validation_nonpositive_harmful = [
        row
        for row in rows
        if row.get("validation_delta_class") != "positive"
        and row.get("candidate_delta_class") == "harmful"
    ]
    high_oracle_harmful = [
        row
        for row in harmful
        if _safe_float(row, "candidate_oracle_delta_over_no_edit") >= 0.12
    ]
    broad_over_edit_harmful = [
        row
        for row in harmful
        if _safe_float(row, "candidate_fraction_with_any_selected_edit") >= 0.02
    ]
    sparse_harmful = [
        row
        for row in harmful
        if _safe_float(row, "candidate_fraction_with_any_selected_edit") < 0.005
    ]
    severe_harmful = [
        row for row in harmful if _safe_float(row, "candidate_delta_over_no_edit") <= -0.01
    ]
    sentinel_block = {8, 13, 17, 18, 32, 33, 53, 54}
    sentinel_false_positive = [row for row in rows if int(row["seed"]) in sentinel_block]

    crosstab: dict[str, dict[str, int]] = {}
    for row in rows:
        v = str(row.get("validation_delta_class"))
        c = str(row.get("candidate_delta_class"))
        crosstab.setdefault(v, {})
        crosstab[v][c] = crosstab[v].get(c, 0) + 1

    taxonomy = [
        _taxonomy_entry(
            name="validation_false_positive_harmful",
            definition="validation delta is positive, but held-out candidate delta is harmful",
            rows=validation_false_positive,
            interpretation=(
                "This is the main d7 harmful type: validation evidence looks usable, "
                "but held-out stage_c_corr candidate behavior is harmful."
            ),
        ),
        _taxonomy_entry(
            name="validation_nonpositive_harmful",
            definition="validation is neutral/nonpositive and held-out candidate delta is harmful",
            rows=validation_nonpositive_harmful,
            interpretation=(
                "These are less dangerous for adoption because validation already gives "
                "little support; selected-mode fallback blocks them."
            ),
        ),
        _taxonomy_entry(
            name="high_oracle_harmful",
            definition="held-out candidate branch is harmful despite candidate-oracle delta >= 0.12",
            rows=high_oracle_harmful,
            interpretation=(
                "These seeds prove the issue is wrong candidate ranking, not absence of "
                "useful local-edit candidates."
            ),
        ),
        _taxonomy_entry(
            name="broad_over_edit_harmful",
            definition="harmful candidate branch emits edits on at least 2% of held-out shots",
            rows=broad_over_edit_harmful,
            interpretation=(
                "Many harmful seeds are broad over-edit cases: the candidate branch touches "
                "too many shots and accumulates more harm than benefit."
            ),
        ),
        _taxonomy_entry(
            name="sparse_harmful",
            definition="harmful candidate branch emits edits on less than 0.5% of held-out shots",
            rows=sparse_harmful,
            interpretation=(
                "Sparse harmful seeds show that even low edit frequency can be harmful "
                "when the selected local edits are misranked."
            ),
        ),
        _taxonomy_entry(
            name="severe_harmful",
            definition="held-out candidate delta is <= -0.01",
            rows=severe_harmful,
            interpretation=(
                "The worst harmful cases are large enough to justify conservative selected-mode fallback."
            ),
        ),
        _taxonomy_entry(
            name="sentinel_false_positive_block_set",
            definition="recommended d7 block sentinel seeds",
            rows=sentinel_false_positive,
            interpretation=(
                "Future d7 objectives must keep this group blocked while preserving true positives."
            ),
        ),
    ]

    validation_positive_count = len(validation_positive)
    false_positive_count = len(validation_false_positive)
    true_positive_count = len(validation_true_positive)
    false_positive_ratio = (
        None
        if validation_positive_count == 0
        else float(false_positive_count / validation_positive_count)
    )

    return {
        "schema_version": "predecoder_d7_harmful_edit_taxonomy.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": input_path.as_posix(),
        "num_rows": len(rows),
        "summary_counts": {
            "candidate_positive": len(positive),
            "candidate_neutral": len(neutral),
            "candidate_harmful": len(harmful),
            "validation_positive": validation_positive_count,
            "validation_true_positive": true_positive_count,
            "validation_false_positive_harmful": false_positive_count,
            "validation_positive_neutral": len(validation_positive_neutral),
            "validation_nonpositive_harmful": len(validation_nonpositive_harmful),
            "high_oracle_harmful": len(high_oracle_harmful),
            "broad_over_edit_harmful": len(broad_over_edit_harmful),
            "sparse_harmful": len(sparse_harmful),
            "severe_harmful": len(severe_harmful),
            "false_positive_ratio_among_validation_positive": false_positive_ratio,
        },
        "validation_vs_candidate_crosstab": crosstab,
        "harmful_overall": _group_stats(harmful),
        "validation_positive_false_vs_true": {
            "false_positive_harmful": _group_stats(validation_false_positive),
            "true_positive": _group_stats(validation_true_positive),
            "neutral": _group_stats(validation_positive_neutral),
        },
        "taxonomy": taxonomy,
        "interpretation": [
            "D7 harmful edits are dominated by validation-positive false positives.",
            "Among validation-positive seeds, harmful held-out candidates are more common than true positives.",
            "Several harmful seeds still have high oracle headroom, so the failure is selector ranking/generalization.",
            "Selected-mode fallback blocks the harmful candidate branches; broad d7 adoption remains unsafe.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.input)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "schema_version": summary["schema_version"],
                "candidate_harmful": summary["summary_counts"]["candidate_harmful"],
                "validation_false_positive_harmful": summary["summary_counts"][
                    "validation_false_positive_harmful"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
