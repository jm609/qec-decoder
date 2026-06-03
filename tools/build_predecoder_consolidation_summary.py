from __future__ import annotations

"""
Build a compact consolidation summary for the transition-aware pre-decoder line.

The script reads already-generated comparison artifacts and writes one JSON
summary that can be cited from status documents without re-running training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import datetime as dt
import json


SCHEMA_VERSION = "predecoder_consolidated_evidence.v2"


@dataclass(frozen=True, slots=True)
class CompareSpec:
    name: str
    distance: str
    artifact: Path
    row_mode_key: str
    target_manifest: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _mode_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get(key) or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _target_oracle_stats(path: Path, eval_family: str) -> dict[str, float | None]:
    data = _load_json(path)
    stats = ((data.get("oracle_stats_by_family") or {}).get(eval_family) or {})
    baseline = stats.get("baseline_pymatching_logical_class4_accuracy")
    oracle = stats.get("oracle_pymatching_logical_class4_accuracy_after_edit_targets")
    baseline_f = None if baseline is None else float(baseline)
    oracle_f = None if oracle is None else float(oracle)
    return {
        "target_baseline_accuracy": baseline_f,
        "target_local_edit_oracle_accuracy": oracle_f,
        "target_local_edit_oracle_delta": (
            None if baseline_f is None or oracle_f is None else float(oracle_f - baseline_f)
        ),
    }


def _safe_add(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left + right)


def _recovery_fraction(delta: float | None, oracle_delta: float | None) -> float | None:
    if delta is None or oracle_delta is None or abs(float(oracle_delta)) < 1e-12:
        return None
    return float(delta / oracle_delta)


def _summarize_compare(spec: CompareSpec) -> dict[str, Any]:
    data = _load_json(spec.artifact)
    rows = list(data.get("rows") or [])
    eval_family = str(data.get("eval_family") or "stage_c_corr")
    selected_deltas = [
        float(row["eval_selected_delta_over_no_edit"])
        for row in rows
        if row.get("eval_selected_delta_over_no_edit") is not None
    ]
    candidate_deltas = [
        float(row["eval_candidate_delta_over_no_edit"])
        for row in rows
        if row.get("eval_candidate_delta_over_no_edit") is not None
    ]
    row_baselines = [
        float(row["eval_no_edit_accuracy"])
        for row in rows
        if row.get("eval_no_edit_accuracy") is not None
    ]
    target_oracle = _target_oracle_stats(spec.target_manifest, eval_family)
    baseline_accuracy = _mean(row_baselines)
    if baseline_accuracy is None:
        baseline_accuracy = target_oracle["target_baseline_accuracy"]
    mean_selected_delta = _mean(selected_deltas)
    mean_candidate_delta = _mean(candidate_deltas)
    target_oracle_delta = target_oracle["target_local_edit_oracle_delta"]
    return {
        "name": spec.name,
        "distance": spec.distance,
        "artifact": spec.artifact.as_posix(),
        "target_manifest": spec.target_manifest.as_posix(),
        "eval_family": eval_family,
        "num_seeds": len(rows),
        "seed_min": min((int(row["seed"]) for row in rows), default=None),
        "seed_max": max((int(row["seed"]) for row in rows), default=None),
        "mean_no_edit_accuracy": baseline_accuracy,
        "mean_selected_accuracy": _safe_add(baseline_accuracy, mean_selected_delta),
        "mean_candidate_accuracy": _safe_add(baseline_accuracy, mean_candidate_delta),
        "target_local_edit_oracle_accuracy": target_oracle["target_local_edit_oracle_accuracy"],
        "target_local_edit_oracle_delta": target_oracle_delta,
        "mean_selected_delta_over_no_edit": mean_selected_delta,
        "mean_candidate_delta_over_no_edit": mean_candidate_delta,
        "selected_target_oracle_recovery_fraction": _recovery_fraction(
            mean_selected_delta,
            target_oracle_delta,
        ),
        "candidate_target_oracle_recovery_fraction": _recovery_fraction(
            mean_candidate_delta,
            target_oracle_delta,
        ),
        "mode_counts": _mode_counts(rows, spec.row_mode_key),
        "selected_improved_total": sum(int(row.get("selected_improved") or 0) for row in rows),
        "selected_harmed_total": sum(int(row.get("selected_harmed") or 0) for row in rows),
        "candidate_improved_total": sum(int(row.get("candidate_improved") or 0) for row in rows),
        "candidate_harmed_total": sum(int(row.get("candidate_harmed") or 0) for row in rows),
    }


def _summarize_oracle(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    summary = dict(data.get("summary") or {})
    return {
        "artifact": path.as_posix(),
        "eval_family": data.get("eval_family"),
        "num_rows": summary.get("num_rows"),
        "mean_selected_delta_over_no_edit": summary.get("mean_selected_delta_over_no_edit"),
        "mean_candidate_delta_over_no_edit": summary.get("mean_candidate_delta_over_no_edit"),
        "mean_candidate_oracle_delta_over_no_edit": summary.get(
            "mean_candidate_oracle_delta_over_no_edit"
        ),
        "mean_candidate_to_oracle_gap": summary.get("mean_candidate_to_oracle_gap"),
        "candidate_delta_class_counts": summary.get("candidate_delta_class_counts"),
        "oracle_delta_class_counts": summary.get("oracle_delta_class_counts"),
        "local_selected_count": summary.get("local_selected_count"),
        "local_selected_seeds": summary.get("local_selected_seeds"),
        "positive_oracle_harmful_candidate_count": summary.get(
            "positive_oracle_harmful_candidate_count"
        ),
        "positive_oracle_neutral_candidate_count": summary.get(
            "positive_oracle_neutral_candidate_count"
        ),
    }


def _summarize_recent_rejection(name: str, path: Path) -> dict[str, Any]:
    data = _load_json(path)
    rows = list(data.get("rows") or [])
    row = rows[0] if rows else {}
    return {
        "name": name,
        "artifact": path.as_posix(),
        "num_rows": len(rows),
        "mean_selected_delta_over_no_edit": data.get("mean_eval_selected_delta_over_no_edit"),
        "mean_candidate_delta_over_no_edit": data.get("mean_eval_candidate_delta_over_no_edit"),
        "seed": row.get("seed"),
        "adoption_reason": row.get("adoption_reason"),
        "selected_delta_over_no_edit": row.get("eval_selected_delta_over_no_edit"),
        "candidate_delta_over_no_edit": row.get("eval_candidate_delta_over_no_edit"),
    }


def build_summary(root: Path) -> dict[str, Any]:
    d3 = CompareSpec(
        name="d3_candidatefirst_seed0_7",
        distance="d3",
        artifact=root / "sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json",
        row_mode_key="mode",
        target_manifest=Path("artifacts/datasets/predecoder_targets_d3_2k_router1k/manifest.json"),
    )
    d5 = CompareSpec(
        name="d5_candidatefirst_seed0_7",
        distance="d5",
        artifact=root / "sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_7.json",
        row_mode_key="mode",
        target_manifest=Path("artifacts/datasets/predecoder_targets_d5_2k_router1k/manifest.json"),
    )
    d7_support = CompareSpec(
        name="d7_support_guard_seed0_57",
        distance="d7",
        artifact=(
            root
            / "sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json"
        ),
        row_mode_key="selected_mode",
        target_manifest=Path("artifacts/datasets/predecoder_targets_d7_2k_router1k/manifest.json"),
    )
    oracle_path = root / "sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json"
    crossfam_weak = (
        root
        / "sedp_d7_candidatefirst_idmargin05_crossfam025_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json"
    )
    crossfam_strong = (
        root
        / "sedp_d7_candidatefirst_idmargin05_crossfam10_m05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed54.json"
    )
    compatpair_topk = (
        root
        / "sedp_d7_candidatefirst_idmargin05_compatpair_topk_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_selection_compare_seed2_54.json"
    )

    distance_results = [
        _summarize_compare(d3),
        _summarize_compare(d5),
        _summarize_compare(d7_support),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "eval_family": "stage_c_corr",
        "distance_results": distance_results,
        "paper_result_table": [
            {
                "distance": row["distance"],
                "seeds": f"{row['seed_min']}..{row['seed_max']}",
                "raw_pymatching_accuracy": row["mean_no_edit_accuracy"],
                "selected_predecoder_accuracy": row["mean_selected_accuracy"],
                "candidate_branch_accuracy": row["mean_candidate_accuracy"],
                "target_local_edit_oracle_accuracy": row["target_local_edit_oracle_accuracy"],
                "selected_delta_over_raw": row["mean_selected_delta_over_no_edit"],
                "selected_oracle_recovery_fraction": row[
                    "selected_target_oracle_recovery_fraction"
                ],
                "selected_modes": row["mode_counts"],
            }
            for row in distance_results
        ],
        "d7_oracle_gap": _summarize_oracle(oracle_path),
        "recent_rejected_d7_objectives": [
            _summarize_recent_rejection("cross_family_positive_negative_0.25_0.5", crossfam_weak),
            _summarize_recent_rejection("cross_family_positive_negative_1.0_0.5", crossfam_strong),
            _summarize_recent_rejection("candidate_compatibility_pairwise_topk", compatpair_topk),
        ],
        "conclusions": [
            "d3 and d5 have selected-mode gains over raw PyMatching under candidate-first safety.",
            "d7 support-guard selected mode is safe over seeds 0..57 but recovers only sparse learned gains.",
            "d7 candidate-oracle headroom remains high, so the blocker is selector ranking/generalization rather than candidate coverage.",
            "The latest cross-family hard positive-vs-negative objective failed the seed54 false-positive gate.",
            "The candidate-compatibility pairwise top-k gate blocks seed54 but destroys the seed2 true-positive candidate branch.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/eval/nn"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"json_out": args.json_out.as_posix(), "schema_version": SCHEMA_VERSION}, indent=2))


if __name__ == "__main__":
    main()
