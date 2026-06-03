"""Build a noise-family analysis summary from existing predecoder artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _count_signs(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(1 for value in values if value > 0.0),
        "neutral": sum(1 for value in values if value == 0.0),
        "harmful": sum(1 for value in values if value < 0.0),
    }


def _successful_distance_family_rows(robustness: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in robustness.get("distance_results") or []:
        distance = str(result["distance"])
        validation_summary = result.get("validation_family_summary") or {}
        for family, family_summary in validation_summary.items():
            rows.append(
                {
                    "distance": distance,
                    "family": str(family),
                    "role": "train_validation",
                    "mean_selected_delta_over_raw": family_summary.get(
                        "mean_val_selected_delta_over_no_edit"
                    ),
                    "positive_seed_count": int(
                        family_summary.get("positive_selected_seed_count") or 0
                    ),
                    "neutral_seed_count": int(
                        family_summary.get("neutral_selected_seed_count") or 0
                    ),
                    "harmful_seed_count": int(
                        family_summary.get("harmful_selected_seed_count") or 0
                    ),
                    "notes": "Validation-family selected-mode behavior before held-out stage_c_corr evaluation.",
                }
            )
        rows.append(
            {
                "distance": distance,
                "family": result.get("eval_family"),
                "role": "heldout_eval",
                "mean_selected_delta_over_raw": result.get("mean_selected_delta_over_raw"),
                "positive_seed_count": int(result.get("positive_seed_count") or 0),
                "neutral_seed_count": int(result.get("neutral_seed_count") or 0),
                "harmful_seed_count": int(result.get("harmful_seed_count") or 0),
                "notes": "Held-out claim family.",
            }
        )
    return rows


def _d7_contrast_rows(d7_compare: dict[str, Any]) -> dict[str, Any]:
    rows = list(d7_compare.get("rows") or [])
    validation_deltas = [
        float(row.get("validation_delta_over_no_edit") or 0.0)
        for row in rows
        if row.get("validation_delta_over_no_edit") is not None
    ]
    heldout_selected_deltas = [
        float(row.get("eval_selected_delta_over_no_edit") or 0.0)
        for row in rows
        if row.get("eval_selected_delta_over_no_edit") is not None
    ]
    heldout_candidate_deltas = [
        float(row.get("eval_candidate_delta_over_no_edit") or 0.0)
        for row in rows
        if row.get("eval_candidate_delta_over_no_edit") is not None
    ]
    validation_positive_heldout_harmful = sum(
        1
        for row in rows
        if float(row.get("validation_delta_over_no_edit") or 0.0) > 0.0
        and float(row.get("eval_candidate_delta_over_no_edit") or 0.0) < 0.0
    )
    validation_positive_heldout_positive = sum(
        1
        for row in rows
        if float(row.get("validation_delta_over_no_edit") or 0.0) > 0.0
        and float(row.get("eval_candidate_delta_over_no_edit") or 0.0) > 0.0
    )
    return {
        "distance": "d7",
        "source_compare": "artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json",
        "num_seeds": len(rows),
        "eval_family": d7_compare.get("eval_family"),
        "mean_validation_delta_over_raw": _mean(validation_deltas),
        "validation_sign_counts": _count_signs(validation_deltas),
        "mean_heldout_selected_delta_over_raw": _mean(heldout_selected_deltas),
        "heldout_selected_sign_counts": _count_signs(heldout_selected_deltas),
        "mean_heldout_candidate_delta_over_raw": _mean(heldout_candidate_deltas),
        "heldout_candidate_sign_counts": _count_signs(heldout_candidate_deltas),
        "validation_positive_heldout_candidate_harmful": validation_positive_heldout_harmful,
        "validation_positive_heldout_candidate_positive": validation_positive_heldout_positive,
        "notes": (
            "D7 is included as a contrast: validation-positive candidate evidence "
            "does not reliably transfer to held-out stage_c_corr."
        ),
    }


def build_summary(root: Path) -> dict[str, Any]:
    robustness_path = root / "sedp_d3_d5_robustness_summary.json"
    d7_compare_path = (
        root
        / "sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json"
    )
    robustness = _load_json(robustness_path)
    d7_compare = _load_json(d7_compare_path)
    return {
        "schema_version": "predecoder_noise_family_analysis.v1",
        "successful_distance_family_rows": _successful_distance_family_rows(robustness),
        "d7_contrast": _d7_contrast_rows(d7_compare),
        "conclusion": (
            "D3 has positive selected-mode behavior in both validation noise families "
            "and held-out stage_c_corr. D5 is noisier across validation families, but "
            "selected-mode adoption remains non-harmful on held-out stage_c_corr. D7 "
            "shows the scaling failure: validation-positive evidence often does not "
            "transfer to held-out candidate behavior."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/eval/nn"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_noise_family_analysis_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"json_out": args.json_out.as_posix(), "schema_version": summary["schema_version"]}, indent=2))


if __name__ == "__main__":
    main()
