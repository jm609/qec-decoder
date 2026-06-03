"""Build a compact baseline-comparison summary for the predecoder writeup."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_FAMILY = "stage_c_corr"

DIRECT_BASELINE_SPECS = [
    {
        "name": "FLFD-small d3",
        "distance": "d3",
        "artifact": "artifacts/eval/nn/flfd_small_2k_manifest/experiment_summary.json",
        "role": "direct neural logical-classifier context baseline",
    },
    {
        "name": "FLFD-small d5",
        "distance": "d5",
        "artifact": "artifacts/eval/nn/flfd_small_d5_2k_manifest/experiment_summary.json",
        "role": "direct neural logical-classifier context baseline",
    },
    {
        "name": "FLFD-small d7",
        "distance": "d7",
        "artifact": "artifacts/eval/nn/flfd_small_d7_2k_manifest/experiment_summary.json",
        "role": "direct neural logical-classifier context baseline",
    },
    {
        "name": "M3D-FLFD d3",
        "distance": "d3",
        "artifact": "artifacts/eval/nn/m3d_flfd_d3_2k_manifest/experiment_summary.json",
        "role": "multiscale direct neural logical-classifier context baseline",
    },
    {
        "name": "M3D-FLFD d5",
        "distance": "d5",
        "artifact": "artifacts/eval/nn/m3d_flfd_d5_2k_manifest/experiment_summary.json",
        "role": "multiscale direct neural logical-classifier context baseline",
    },
    {
        "name": "M3D-FLFD d5 stronger",
        "distance": "d5",
        "artifact": "artifacts/eval/nn/m3d_flfd_d5_2k_stronger/experiment_summary.json",
        "role": "stronger multiscale direct neural ablation",
    },
]

PYMATCHING_REFRESH_SPECS = {
    "d3": "artifacts/eval/pymatching/d3_2k_class4_refresh.json",
    "d5": "artifacts/eval/pymatching/d5_2k_class4_refresh.json",
    "d7": "artifacts/eval/pymatching/d7_2k_class4_refresh.json",
}

RECTCNN_READINESS_SPEC = {
    "name": "RectCNN readiness d3",
    "distance": "d3",
    "artifact": "artifacts/eval/nn/class4_rectcnn_stagea_eval_stagec.json",
    "role": "paper-style CNN readiness baseline; not a main comparable result",
}

PREDECODER_CONSOLIDATED = (
    "artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _family_metrics(summary: dict[str, Any], family: str) -> dict[str, Any]:
    for row in summary.get("evaluation", {}).get("per_family", []):
        if isinstance(row, dict) and row.get("family") == family:
            return row
    raise KeyError(f"Family {family!r} not found in evaluation.per_family")


def _direct_baseline_row(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    path = root / spec["artifact"]
    summary = _load_json(path)
    metrics = _family_metrics(summary, EVAL_FAMILY)
    training = summary.get("training") or {}
    manifest = summary.get("manifest") or {}
    mixed = training.get("mixed_test_split_metrics") or {}
    return {
        "name": spec["name"],
        "distance": spec["distance"],
        "decoder": summary.get("decoder"),
        "role": spec["role"],
        "artifact": spec["artifact"],
        "manifest_path": manifest.get("path"),
        "train_families": training.get("train_families"),
        "eval_family": EVAL_FAMILY,
        "stage_c_accuracy": _as_float(metrics.get("accuracy")),
        "stage_c_macro_f1": _as_float(metrics.get("macro_f1")),
        "stage_c_balanced_accuracy": _as_float(metrics.get("balanced_accuracy")),
        "stage_c_target_class_histogram": metrics.get("target_class_histogram"),
        "stage_c_predicted_class_histogram": metrics.get("predicted_class_histogram"),
        "mixed_test_accuracy": _as_float(mixed.get("accuracy")),
        "mixed_test_macro_f1": _as_float(mixed.get("macro_f1")),
    }


def _pymatching_refresh_row(root: Path, distance: str, artifact: str) -> dict[str, Any]:
    summary = _load_json(root / artifact)
    family_summary = summary.get("by_family", {}).get(EVAL_FAMILY) or {}
    metrics = family_summary.get("metrics") or {}
    manifest_summary = summary.get("manifest_summary") or {}
    return {
        "distance": distance,
        "decoder": "pymatching",
        "role": "raw PyMatching on direct class4 refresh manifest",
        "artifact": artifact,
        "manifest_path": summary.get("manifest_path"),
        "eval_family": EVAL_FAMILY,
        "shots": manifest_summary.get("shots"),
        "logical_class4_accuracy": _as_float(metrics.get("logical_class4_accuracy")),
        "frame_error_rate": _as_float(metrics.get("frame_error_rate")),
        "estimated_ler_per_cycle": _as_float(metrics.get("estimated_ler_per_cycle")),
    }


def _rectcnn_readiness_row(root: Path) -> dict[str, Any]:
    spec = RECTCNN_READINESS_SPEC
    summary = _load_json(root / spec["artifact"])
    metrics = summary.get("metrics") or {}
    return {
        "name": spec["name"],
        "distance": spec["distance"],
        "decoder": summary.get("decoder"),
        "role": spec["role"],
        "artifact": spec["artifact"],
        "eval_family": summary.get("family"),
        "num_examples": metrics.get("num_examples"),
        "accuracy": _as_float(metrics.get("accuracy")),
        "macro_f1": _as_float(metrics.get("macro_f1")),
        "target_class_histogram": metrics.get("target_class_histogram"),
        "predicted_class_histogram": metrics.get("predicted_class_histogram"),
        "comparison_note": (
            "Only a 24-shot readiness artifact; use as architecture smoke "
            "context, not as a main numerical baseline."
        ),
    }


def _predecoder_rows(root: Path) -> list[dict[str, Any]]:
    summary = _load_json(root / PREDECODER_CONSOLIDATED)
    target_manifests = {
        row["distance"]: row.get("target_manifest")
        for row in summary.get("distance_results", [])
        if isinstance(row, dict) and row.get("distance")
    }
    rows = []
    for row in summary.get("paper_result_table", []):
        distance = str(row["distance"])
        rows.append(
            {
                "distance": distance,
                "role": "main same-artifact predecoder-vs-raw-PyMatching result",
                "target_manifest": target_manifests.get(distance),
                "eval_family": EVAL_FAMILY,
                "seeds": row.get("seeds"),
                "raw_pymatching_accuracy": _as_float(row.get("raw_pymatching_accuracy")),
                "selected_predecoder_accuracy": _as_float(
                    row.get("selected_predecoder_accuracy")
                ),
                "candidate_branch_accuracy": _as_float(row.get("candidate_branch_accuracy")),
                "target_local_edit_oracle_accuracy": _as_float(
                    row.get("target_local_edit_oracle_accuracy")
                ),
                "selected_delta_over_raw": _as_float(row.get("selected_delta_over_raw")),
                "selected_modes": row.get("selected_modes"),
            }
        )
    return rows


def build_summary(root: Path) -> dict[str, Any]:
    direct_rows = [
        _direct_baseline_row(root, spec) for spec in DIRECT_BASELINE_SPECS
    ]
    pymatching_rows = [
        _pymatching_refresh_row(root, distance, artifact)
        for distance, artifact in PYMATCHING_REFRESH_SPECS.items()
    ]
    predecoder_rows = _predecoder_rows(root)
    rectcnn_row = _rectcnn_readiness_row(root)

    return {
        "schema_version": "predecoder_baseline_comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_family": EVAL_FAMILY,
        "comparison_scope": {
            "main_fair_comparison": (
                "Use predecoder_rows: selected predecoder and raw no-edit "
                "PyMatching are evaluated on the same predecoder target artifacts."
            ),
            "context_comparison": (
                "Direct FLFD/M3D/RectCNN rows are earlier direct neural decoder "
                "baselines and may use different manifests or readiness-scale "
                "artifacts; they motivate the predecoder path but should not be "
                "overclaimed as a strict head-to-head result."
            ),
        },
        "predecoder_rows": predecoder_rows,
        "pymatching_refresh_rows": pymatching_rows,
        "direct_neural_baseline_rows": direct_rows,
        "rectcnn_readiness_row": rectcnn_row,
        "conclusions": [
            "Raw PyMatching is the primary fair baseline for the final predecoder claim.",
            "Direct neural logical classifiers underperform PyMatching in the distance ladder and motivate the predecoder-plus-PyMatching design.",
            "The proposed model should be described as a neural predecoder, not as an end-to-end neural replacement for PyMatching.",
            "RectCNN is retained only as a paper-style architecture smoke/readiness baseline.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_baseline_comparison_summary.json"),
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
                "direct_rows": len(summary["direct_neural_baseline_rows"]),
                "predecoder_rows": len(summary["predecoder_rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
