"""Summarize candidate-oracle headroom for pre-decoder seed sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _accuracy(section: dict[str, Any] | None, family: str, key: str) -> float | None:
    result = _get_nested(section or {}, (family, key, "accuracy"))
    return _as_float(result)


def _change_summary(section: dict[str, Any] | None, family: str) -> dict[str, Any]:
    summary = _get_nested(section or {}, (family, "change_summary"))
    return summary if isinstance(summary, dict) else {}


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _mean(values: list[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return mean(kept) if kept else None


def _quantiles(values: list[float | None]) -> dict[str, float] | None:
    kept = sorted(float(value) for value in values if value is not None)
    if not kept:
        return None

    def at(frac: float) -> float:
        if len(kept) == 1:
            return kept[0]
        pos = frac * (len(kept) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(kept) - 1)
        weight = pos - lo
        return kept[lo] * (1.0 - weight) + kept[hi] * weight

    return {
        "min": kept[0],
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "max": kept[-1],
    }


def _classify_delta(delta: float | None) -> str:
    if delta is None:
        return "missing"
    if delta > 0.0:
        return "positive"
    if delta < 0.0:
        return "harmful"
    return "neutral"


def _load_seed_row(row: dict[str, Any], eval_family: str) -> dict[str, Any]:
    run_dir_raw = row.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw:
        raise ValueError(f"Missing run_dir for seed row: {row}")
    run_dir = Path(run_dir_raw)
    summary_path = run_dir / "experiment_summary.json"
    summary = _load_json(summary_path)

    no_edit = summary.get("eval_families_no_edit")
    candidate = summary.get("eval_families_candidate_selector")
    selected = summary.get("eval_families")
    candidate_summary = _change_summary(candidate, eval_family)

    baseline_accuracy = _accuracy(no_edit, eval_family, "baseline_pymatching")
    candidate_accuracy = _accuracy(candidate, eval_family, "edited_pymatching")
    selected_accuracy = _accuracy(selected, eval_family, "edited_pymatching")
    candidate_oracle_accuracy = _as_float(candidate_summary.get("selector_candidate_oracle_accuracy"))

    candidate_delta = _delta(candidate_accuracy, baseline_accuracy)
    # For post-hoc/simulated policy summaries, the row value is the selected
    # policy delta even when the source run evaluated a less guarded mode.
    row_selected_delta = _as_float(row.get("eval_selected_delta_over_no_edit"))
    selected_delta = row_selected_delta
    if selected_delta is None:
        selected_delta = _delta(selected_accuracy, baseline_accuracy)
    if selected_delta is not None and baseline_accuracy is not None:
        selected_accuracy = float(baseline_accuracy) + float(selected_delta)
    oracle_delta = _delta(candidate_oracle_accuracy, baseline_accuracy)

    return {
        "seed": _as_int(row.get("seed")),
        "run_dir": run_dir.as_posix(),
        "selected_mode": row.get("selected_mode") or row.get("mode"),
        "adoption_reason": row.get("adoption_reason"),
        "adoption_margin": _as_float(row.get("adoption_margin")),
        "adoption_nonzero": _as_int(row.get("adoption_nonzero")),
        "baseline_accuracy": baseline_accuracy,
        "selected_accuracy": selected_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "candidate_oracle_accuracy": candidate_oracle_accuracy,
        "selected_delta_over_no_edit": selected_delta,
        "candidate_delta_over_no_edit": candidate_delta,
        "candidate_oracle_delta_over_no_edit": oracle_delta,
        "candidate_to_oracle_gap": (
            None
            if candidate_oracle_accuracy is None or candidate_accuracy is None
            else float(candidate_oracle_accuracy) - float(candidate_accuracy)
        ),
        "candidate_delta_class": _classify_delta(candidate_delta),
        "oracle_delta_class": _classify_delta(oracle_delta),
        "candidate_improved": _as_int(candidate_summary.get("num_improved_over_baseline")),
        "candidate_harmed": _as_int(candidate_summary.get("num_harmed_vs_baseline")),
        "candidate_mean_candidates_per_shot": _as_float(
            candidate_summary.get("selector_mean_candidates_per_shot")
        ),
        "candidate_mean_selected_edit_weight": _as_float(
            candidate_summary.get("selector_mean_selected_edit_weight")
        ),
        "candidate_fraction_with_any_selected_edit": _as_float(
            candidate_summary.get("selector_fraction_with_any_selected_edit")
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_deltas = [row["candidate_delta_over_no_edit"] for row in rows]
    selected_deltas = [row["selected_delta_over_no_edit"] for row in rows]
    oracle_deltas = [row["candidate_oracle_delta_over_no_edit"] for row in rows]
    gaps = [row["candidate_to_oracle_gap"] for row in rows]

    by_candidate_class: dict[str, int] = {}
    by_oracle_class: dict[str, int] = {}
    for row in rows:
        by_candidate_class[row["candidate_delta_class"]] = (
            by_candidate_class.get(row["candidate_delta_class"], 0) + 1
        )
        by_oracle_class[row["oracle_delta_class"]] = by_oracle_class.get(row["oracle_delta_class"], 0) + 1

    positive_oracle_harmful_candidate = [
        row["seed"]
        for row in rows
        if row["oracle_delta_class"] == "positive" and row["candidate_delta_class"] == "harmful"
    ]
    positive_oracle_neutral_candidate = [
        row["seed"]
        for row in rows
        if row["oracle_delta_class"] == "positive" and row["candidate_delta_class"] == "neutral"
    ]

    return {
        "num_rows": len(rows),
        "seed_min": min(row["seed"] for row in rows if row["seed"] is not None),
        "seed_max": max(row["seed"] for row in rows if row["seed"] is not None),
        "mean_selected_delta_over_no_edit": _mean(selected_deltas),
        "mean_candidate_delta_over_no_edit": _mean(candidate_deltas),
        "mean_candidate_oracle_delta_over_no_edit": _mean(oracle_deltas),
        "mean_candidate_to_oracle_gap": _mean(gaps),
        "candidate_delta_quantiles": _quantiles(candidate_deltas),
        "candidate_oracle_delta_quantiles": _quantiles(oracle_deltas),
        "candidate_to_oracle_gap_quantiles": _quantiles(gaps),
        "candidate_delta_class_counts": by_candidate_class,
        "oracle_delta_class_counts": by_oracle_class,
        "positive_oracle_harmful_candidate_count": len(positive_oracle_harmful_candidate),
        "positive_oracle_harmful_candidate_seeds": positive_oracle_harmful_candidate,
        "positive_oracle_neutral_candidate_count": len(positive_oracle_neutral_candidate),
        "positive_oracle_neutral_candidate_seeds": positive_oracle_neutral_candidate,
        "local_selected_count": sum(1 for row in rows if row.get("selected_mode") == "local_motif_selector"),
        "local_selected_seeds": [
            row["seed"] for row in rows if row.get("selected_mode") == "local_motif_selector"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", type=Path, required=True)
    parser.add_argument("--eval-family", default="stage_c_corr")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    compare = _load_json(args.compare_json)
    raw_rows = compare.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"Missing rows list in {args.compare_json}")

    rows = [_load_seed_row(row, str(args.eval_family)) for row in raw_rows if isinstance(row, dict)]
    rows.sort(key=lambda row: (-1 if row["seed"] is None else row["seed"]))
    payload = {
        "schema_version": "predecoder_candidate_oracle_analysis.v1",
        "compare_json": args.compare_json.as_posix(),
        "eval_family": str(args.eval_family),
        "summary": _summarize(rows),
        "rows": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
