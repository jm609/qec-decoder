"""Build the ablation/failure-path synthesis used by the thesis writeup."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_COMPARISON = Path("artifacts/eval/nn/sedp_baseline_comparison_summary.json")
CONSOLIDATED_EVIDENCE = Path(
    "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json"
)
D7_ADOPTION_GRID = Path("artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json")
D7_BOTTLENECK = Path("artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _index_by_distance(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        distance = str(row.get("distance"))
        value = row.get(key)
        if value is not None:
            out[distance] = float(value)
    return out


def _find_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("name", "")).startswith(prefix)]


def _direct_rows_with_delta(
    rows: list[dict[str, Any]], pymatching_by_distance: dict[str, float], prefix: str
) -> list[dict[str, Any]]:
    out = []
    for row in _find_rows(rows, prefix):
        distance = str(row["distance"])
        acc = float(row["stage_c_accuracy"])
        pymatching = pymatching_by_distance.get(distance)
        out.append(
            {
                "name": row["name"],
                "distance": distance,
                "stage_c_accuracy": acc,
                "stage_c_macro_f1": float(row["stage_c_macro_f1"]),
                "pymatching_refresh_accuracy": pymatching,
                "delta_vs_pymatching_refresh": None if pymatching is None else acc - pymatching,
                "artifact": row["artifact"],
            }
        )
    return out


def _result_by_distance(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["distance"]): row for row in rows}


def _metric_row(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def build_summary(root: Path) -> dict[str, Any]:
    baseline = _load_json(root / BASELINE_COMPARISON)
    consolidated = _load_json(root / CONSOLIDATED_EVIDENCE)
    adoption_grid = _load_json(root / D7_ADOPTION_GRID)
    bottleneck = _load_json(root / D7_BOTTLENECK)

    direct_rows = list(baseline["direct_neural_baseline_rows"])
    pymatching_rows = list(baseline["pymatching_refresh_rows"])
    predecoder_rows = list(consolidated["paper_result_table"])
    predecoder_by_distance = _result_by_distance(predecoder_rows)
    pymatching_by_distance = _index_by_distance(pymatching_rows, "logical_class4_accuracy")

    flfd_rows = _direct_rows_with_delta(direct_rows, pymatching_by_distance, "FLFD-small")
    m3d_rows = _direct_rows_with_delta(direct_rows, pymatching_by_distance, "M3D-FLFD")
    rectcnn = baseline["rectcnn_readiness_row"]
    d7_gap = consolidated["d7_oracle_gap"]
    grid_best = (adoption_grid.get("best_results") or [{}])[0]
    rejected_d7 = list(consolidated.get("recent_rejected_d7_objectives") or [])

    architecture_decisions = [
        {
            "decision_point": "Use a standalone direct neural logical_class4 decoder",
            "tested_path": "FLFD-small direct classifier",
            "evidence": flfd_rows,
            "verdict": "reject as final model family",
            "reason": (
                "It partially learns at d3 but falls below PyMatching and collapses with "
                "distance, especially at d7."
            ),
            "paper_use": "context/negative baseline",
        },
        {
            "decision_point": "Fix direct classification by making the dense trunk multiscale",
            "tested_path": "M3D-FLFD direct classifier",
            "evidence": m3d_rows,
            "verdict": "reject as final model family",
            "reason": (
                "The multiscale direct classifier does not close the d5 gap; the stronger "
                "d5 run collapses further."
            ),
            "paper_use": "ablation against another dense direct decoder",
        },
        {
            "decision_point": "Use RectCNN as the main comparison",
            "tested_path": "RectCNN readiness d3",
            "evidence": {
                "num_examples": rectcnn.get("num_examples"),
                "accuracy": rectcnn.get("accuracy"),
                "macro_f1": rectcnn.get("macro_f1"),
                "artifact": rectcnn.get("artifact"),
            },
            "verdict": "retain only as readiness/context baseline",
            "reason": "The available artifact is a 24-shot readiness run, not a final comparable result.",
            "paper_use": "architecture context only",
        },
        {
            "decision_point": "Use neural pre-decoding rather than neural replacement",
            "tested_path": "patch-head local motif predecoder plus PyMatching",
            "evidence": {
                "d3": _metric_row(
                    predecoder_by_distance["d3"],
                    [
                        "raw_pymatching_accuracy",
                        "selected_predecoder_accuracy",
                        "selected_delta_over_raw",
                        "target_local_edit_oracle_accuracy",
                    ],
                ),
                "d5": _metric_row(
                    predecoder_by_distance["d5"],
                    [
                        "raw_pymatching_accuracy",
                        "selected_predecoder_accuracy",
                        "selected_delta_over_raw",
                        "target_local_edit_oracle_accuracy",
                    ],
                ),
            },
            "verdict": "retain as the final proposed structure",
            "reason": "It is the first path that gives selected-mode gains over raw PyMatching at d3 and d5.",
            "paper_use": "main method and main result",
        },
    ]

    d7_failure_decisions = [
        {
            "decision_point": "Explain d7 as candidate-set exhaustion",
            "evidence": {
                "mean_candidate_delta_over_no_edit": d7_gap["mean_candidate_delta_over_no_edit"],
                "mean_candidate_oracle_delta_over_no_edit": d7_gap[
                    "mean_candidate_oracle_delta_over_no_edit"
                ],
                "oracle_delta_class_counts": d7_gap["oracle_delta_class_counts"],
                "candidate_delta_class_counts": d7_gap["candidate_delta_class_counts"],
            },
            "verdict": "reject",
            "reason": "Every checked d7 seed has positive oracle headroom, so candidate coverage is not the blocker.",
        },
        {
            "decision_point": "Solve d7 by scalar selected-mode adoption tuning",
            "evidence": {
                "num_policies_checked": adoption_grid.get("num_policies_checked"),
                "num_passing_policies": adoption_grid.get("num_passing_policies"),
                "best_mean_selected_delta_all_rows": grid_best.get(
                    "mean_selected_delta_all_rows"
                ),
                "best_harmful_block_adopted": grid_best.get("harmful_block_adopted"),
            },
            "verdict": "reject",
            "reason": "No monotone adoption-threshold policy passed the preserve/recover/block sentinel gate.",
        },
    ]

    for row in rejected_d7:
        d7_failure_decisions.append(
            {
                "decision_point": f"Use d7 objective: {row['name']}",
                "evidence": {
                    "seed": row.get("seed"),
                    "num_rows": row.get("num_rows"),
                    "selected_delta_over_no_edit": row.get("selected_delta_over_no_edit"),
                    "candidate_delta_over_no_edit": row.get("candidate_delta_over_no_edit"),
                    "mean_selected_delta_over_no_edit": row.get(
                        "mean_selected_delta_over_no_edit"
                    ),
                    "mean_candidate_delta_over_no_edit": row.get(
                        "mean_candidate_delta_over_no_edit"
                    ),
                    "artifact": row.get("artifact"),
                },
                "verdict": "reject",
                "reason": (
                    "It fails the targeted d7 sentinel logic: either a false-positive "
                    "gate remains open or an existing true-positive seed is destroyed."
                ),
            }
        )

    final_structure_rationale = [
        "PyMatching remains the fair final baseline and final decoder.",
        "The neural component is useful as a local edit selector, not as a standalone logical classifier.",
        "Candidate-first selected-mode safety is necessary because harmful local edits can reduce PyMatching accuracy.",
        "D3 and d5 provide the positive main result; d7 is a controlled selector-ranking/generalization limitation.",
        "The next research-writing step should synthesize these ablations, not add another feature branch.",
    ]

    return {
        "schema_version": "predecoder_ablation_failure_synthesis.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "baseline_comparison": BASELINE_COMPARISON.as_posix(),
            "consolidated_evidence": CONSOLIDATED_EVIDENCE.as_posix(),
            "d7_adoption_grid": D7_ADOPTION_GRID.as_posix(),
            "d7_bottleneck": D7_BOTTLENECK.as_posix(),
        },
        "architecture_decisions": architecture_decisions,
        "d7_failure_decisions": d7_failure_decisions,
        "d7_bottleneck_summary": bottleneck.get("summary"),
        "final_structure_rationale": final_structure_rationale,
        "paper_claim": (
            "The ablation trail supports a neural predecoder plus PyMatching "
            "rather than a standalone neural decoder or another scalar d7 "
            "adoption sweep."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_ablation_failure_synthesis_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "schema_version": summary["schema_version"],
                "architecture_decisions": len(summary["architecture_decisions"]),
                "d7_failure_decisions": len(summary["d7_failure_decisions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
