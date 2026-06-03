"""Build a compact robustness summary for the successful d3/d5 predecoder runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SPECS = {
    "d3": {
        "compare": "sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json",
    },
    "d5": {
        "compare": "sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_7.json",
    },
}


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


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)))


def _accuracy(section: dict[str, Any] | None, family: str) -> float | None:
    if not isinstance(section, dict) or family not in section:
        return None
    metrics = (section.get(family) or {}).get("edited_pymatching") or {}
    value = metrics.get("accuracy")
    return None if value is None else float(value)


def _change_count(section: dict[str, Any] | None, family: str, key: str) -> int | None:
    if not isinstance(section, dict) or family not in section:
        return None
    summary = (section.get(family) or {}).get("change_summary") or {}
    value = summary.get(key)
    return None if value is None else int(value)


def _selected_validation_by_family(summary: dict[str, Any]) -> dict[str, Any] | None:
    training = summary.get("training") or {}
    mode = str(training.get("selected_inference_mode") or "")
    if mode == "raw_no_edit":
        return training.get("best_val_no_edit_by_family")
    if mode in {"candidate_selector", "local_motif_selector"}:
        return training.get("best_val_candidate_selector_by_family")
    if mode == "global_policy":
        return training.get("best_val_global_policy_by_family")
    if mode == "motif_vocab":
        return training.get("best_val_motif_vocab_by_family")
    if mode == "action_motif":
        return training.get("best_val_action_motif_by_family")
    if mode == "local_motif":
        return training.get("best_val_local_motif_by_family")
    return None


def _validation_family_rows(compare_row: dict[str, Any], validation_families: list[str]) -> list[dict[str, Any]]:
    run_dir = Path(str(compare_row["run_dir"]))
    summary = _load_json(run_dir / "experiment_summary.json")
    training = summary.get("training") or {}
    no_edit_by_family = training.get("best_val_no_edit_by_family")
    candidate_by_family = training.get("best_val_candidate_selector_by_family")
    selected_by_family = _selected_validation_by_family(summary)
    rows: list[dict[str, Any]] = []
    for family in validation_families:
        no_edit = _accuracy(no_edit_by_family, family)
        candidate = _accuracy(candidate_by_family, family)
        selected = _accuracy(selected_by_family, family)
        rows.append(
            {
                "family": family,
                "val_no_edit_accuracy": no_edit,
                "val_candidate_accuracy": candidate,
                "val_selected_accuracy": selected,
                "val_candidate_delta_over_no_edit": None
                if no_edit is None or candidate is None
                else candidate - no_edit,
                "val_selected_delta_over_no_edit": None
                if no_edit is None or selected is None
                else selected - no_edit,
                "candidate_improved": _change_count(
                    candidate_by_family, family, "num_improved_over_baseline"
                ),
                "candidate_harmed": _change_count(
                    candidate_by_family, family, "num_harmed_vs_baseline"
                ),
            }
        )
    return rows


def _summarize_distance(distance: str, compare_path: Path) -> dict[str, Any]:
    data = _load_json(compare_path)
    rows = list(data.get("rows") or [])
    validation_families = [str(item) for item in data.get("validation_families") or []]
    seed_rows: list[dict[str, Any]] = []
    for row in rows:
        seed_row = {
            "seed": int(row["seed"]),
            "mode": str(row.get("mode")),
            "best_epoch": row.get("best_epoch"),
            "eval_no_edit_accuracy": float(row["eval_no_edit_accuracy"]),
            "eval_selected_accuracy": float(row["eval_selected_accuracy"]),
            "eval_candidate_accuracy": float(row["eval_candidate_accuracy"]),
            "eval_selected_delta_over_no_edit": float(row["eval_selected_delta_over_no_edit"]),
            "eval_candidate_delta_over_no_edit": float(row["eval_candidate_delta_over_no_edit"]),
            "selected_improved": int(row.get("selected_improved") or 0),
            "selected_harmed": int(row.get("selected_harmed") or 0),
            "candidate_improved": int(row.get("candidate_improved") or 0),
            "candidate_harmed": int(row.get("candidate_harmed") or 0),
            "validation_family_rows": _validation_family_rows(row, validation_families),
        }
        seed_rows.append(seed_row)

    selected_deltas = [row["eval_selected_delta_over_no_edit"] for row in seed_rows]
    positive_rows = [row for row in seed_rows if row["eval_selected_delta_over_no_edit"] > 0.0]
    neutral_rows = [row for row in seed_rows if row["eval_selected_delta_over_no_edit"] == 0.0]
    harmful_rows = [row for row in seed_rows if row["eval_selected_delta_over_no_edit"] < 0.0]
    local_rows = [row for row in seed_rows if row["mode"] == "local_motif_selector"]

    family_summary: dict[str, dict[str, Any]] = {}
    for family in validation_families:
        family_rows = [
            item
            for row in seed_rows
            for item in row["validation_family_rows"]
            if item["family"] == family
        ]
        selected_family_deltas = [
            float(item["val_selected_delta_over_no_edit"])
            for item in family_rows
            if item["val_selected_delta_over_no_edit"] is not None
        ]
        candidate_family_deltas = [
            float(item["val_candidate_delta_over_no_edit"])
            for item in family_rows
            if item["val_candidate_delta_over_no_edit"] is not None
        ]
        family_summary[family] = {
            "mean_val_selected_delta_over_no_edit": _mean(selected_family_deltas),
            "mean_val_candidate_delta_over_no_edit": _mean(candidate_family_deltas),
            "min_val_selected_delta_over_no_edit": min(selected_family_deltas)
            if selected_family_deltas
            else None,
            "positive_selected_seed_count": sum(1 for value in selected_family_deltas if value > 0.0),
            "neutral_selected_seed_count": sum(1 for value in selected_family_deltas if value == 0.0),
            "harmful_selected_seed_count": sum(1 for value in selected_family_deltas if value < 0.0),
        }

    return {
        "distance": distance,
        "source_compare": compare_path.as_posix(),
        "eval_family": data.get("eval_family"),
        "validation_families": validation_families,
        "num_seeds": len(seed_rows),
        "seeds": [row["seed"] for row in seed_rows],
        "mean_raw_pymatching_accuracy": _mean([row["eval_no_edit_accuracy"] for row in seed_rows]),
        "mean_selected_predecoder_accuracy": _mean(
            [row["eval_selected_accuracy"] for row in seed_rows]
        ),
        "mean_candidate_branch_accuracy": _mean(
            [row["eval_candidate_accuracy"] for row in seed_rows]
        ),
        "mean_selected_delta_over_raw": _mean(selected_deltas),
        "stdev_selected_delta_over_raw": _stdev(selected_deltas),
        "min_selected_delta_over_raw": min(selected_deltas) if selected_deltas else None,
        "max_selected_delta_over_raw": max(selected_deltas) if selected_deltas else None,
        "positive_seed_count": len(positive_rows),
        "neutral_seed_count": len(neutral_rows),
        "harmful_seed_count": len(harmful_rows),
        "local_selector_seed_count": len(local_rows),
        "raw_no_edit_seed_count": sum(1 for row in seed_rows if row["mode"] == "raw_no_edit"),
        "selected_improved_total": sum(row["selected_improved"] for row in seed_rows),
        "selected_harmed_total": sum(row["selected_harmed"] for row in seed_rows),
        "candidate_improved_total": sum(row["candidate_improved"] for row in seed_rows),
        "candidate_harmed_total": sum(row["candidate_harmed"] for row in seed_rows),
        "validation_family_summary": family_summary,
        "seed_rows": seed_rows,
    }


def build_summary(root: Path) -> dict[str, Any]:
    distance_results = []
    for distance, spec in DEFAULT_SPECS.items():
        distance_results.append(_summarize_distance(distance, root / spec["compare"]))

    recommendation = (
        "The seed-expanded d3/d5 seed0..7 evidence remains positive in selected "
        "mode, but the wider seed set lowers the mean gains. The claim should "
        "therefore emphasize selected-mode safety and conservative improvement "
        "rather than overstate statistical significance."
    )
    return {
        "schema_version": "predecoder_d3_d5_robustness.v1",
        "distance_results": distance_results,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/eval/nn"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_d3_d5_robustness_summary.json"),
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
