"""Build a compact hyperparameter sensitivity table from existing artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "predecoder_hyperparameter_sensitivity.v1"
DEFAULT_JSON_OUT = Path("artifacts/eval/nn/sedp_hyperparameter_sensitivity_summary.json")


@dataclass(frozen=True, slots=True)
class SensitivitySpec:
    parameter: str
    value: float
    artifact: Path
    verdict: str


DEFAULT_SPECS = (
    SensitivitySpec(
        parameter="selector_identity_margin_loss_weight",
        value=0.25,
        artifact=Path(
            "artifacts/eval/nn/sedp_d7_candidatefirst_idmargin025_diagselect_selection_compare_seed0_2_5.json"
        ),
        verdict="too weak: admits a harmful selected seed0 branch",
    ),
    SensitivitySpec(
        parameter="selector_identity_margin_loss_weight",
        value=0.5,
        artifact=Path(
            "artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_selection_compare_seed0_2_5.json"
        ),
        verdict="best sentinel balance among checked values",
    ),
    SensitivitySpec(
        parameter="selector_identity_margin_loss_weight",
        value=1.0,
        artifact=Path(
            "artifacts/eval/nn/sedp_d7_candidatefirst_idmargin10_diagselect_selection_compare_seed0_2_5.json"
        ),
        verdict="too conservative: suppresses true-positive seed2 recovery",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _seed_delta(rows: list[dict[str, Any]], seed: int, key: str) -> float | None:
    for row in rows:
        if int(row["seed"]) == int(seed):
            value = row.get(key)
            return None if value is None else float(value)
    return None


def _seed_mode(rows: list[dict[str, Any]], seed: int) -> str | None:
    for row in rows:
        if int(row["seed"]) == int(seed):
            return None if row.get("mode") is None else str(row["mode"])
    return None


def _class_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {
        "positive": sum(1 for value in values if value > 0.0),
        "neutral": sum(1 for value in values if value == 0.0),
        "harmful": sum(1 for value in values if value < 0.0),
    }


def _summarize_spec(spec: SensitivitySpec) -> dict[str, Any]:
    data = _load_json(spec.artifact)
    rows = [row for row in data.get("rows") or [] if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No rows in {spec.artifact}")
    seeds = [int(row["seed"]) for row in rows]
    return {
        "parameter": spec.parameter,
        "value": spec.value,
        "artifact": spec.artifact.as_posix(),
        "seeds": seeds,
        "mean_selected_delta_over_no_edit": float(data["mean_eval_selected_delta_over_no_edit"]),
        "mean_candidate_delta_over_no_edit": float(data["mean_eval_candidate_delta_over_no_edit"]),
        "mode_counts": _mode_counts(rows),
        "selected_delta_class_counts": _class_counts(rows, "eval_selected_delta_over_no_edit"),
        "candidate_delta_class_counts": _class_counts(rows, "eval_candidate_delta_over_no_edit"),
        "seed0_selected_delta": _seed_delta(rows, 0, "eval_selected_delta_over_no_edit"),
        "seed2_selected_delta": _seed_delta(rows, 2, "eval_selected_delta_over_no_edit"),
        "seed5_selected_delta": _seed_delta(rows, 5, "eval_selected_delta_over_no_edit"),
        "seed0_candidate_delta": _seed_delta(rows, 0, "eval_candidate_delta_over_no_edit"),
        "seed2_candidate_delta": _seed_delta(rows, 2, "eval_candidate_delta_over_no_edit"),
        "seed5_candidate_delta": _seed_delta(rows, 5, "eval_candidate_delta_over_no_edit"),
        "seed0_mode": _seed_mode(rows, 0),
        "seed2_mode": _seed_mode(rows, 2),
        "seed5_mode": _seed_mode(rows, 5),
        "verdict": spec.verdict,
    }


def build_summary(specs: tuple[SensitivitySpec, ...]) -> dict[str, Any]:
    rows = [_summarize_spec(spec) for spec in specs]
    best = max(rows, key=lambda row: float(row["mean_selected_delta_over_no_edit"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "d7 sentinel seeds 0,2,5",
        "purpose": (
            "Small sensitivity table for selector identity-margin loss weight. "
            "This is not a full hyperparameter search; it documents why the active "
            "d7 follow-up used 0.5 before adding stricter selected-mode guards."
        ),
        "rows": rows,
        "best_by_mean_selected_delta": {
            "parameter": best["parameter"],
            "value": best["value"],
            "mean_selected_delta_over_no_edit": best["mean_selected_delta_over_no_edit"],
        },
        "interpretation": [
            "Weight 0.25 is too weak because it selects a harmful seed0 branch.",
            "Weight 0.5 preserves seed2, recovers seed0, and blocks seed5 in this sentinel check.",
            "Weight 1.0 is too conservative because it blocks all selected recovery.",
            "The table supports the active 0.5 setting as a local compromise, but it does not solve full d7 generalization.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    summary = build_summary(DEFAULT_SPECS)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "schema_version": SCHEMA_VERSION,
                "rows": len(summary["rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
