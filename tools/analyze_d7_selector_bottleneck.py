"""Build a targeted d7 selector-bottleneck diagnostic.

The script consumes the current support-guard selection comparison and the
candidate-oracle analysis. It does not rerun training. Its purpose is to select
the next d7 sentinel groups before adding another objective or running more
seed extensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_COMPARE_JSON = Path(
    "artifacts/eval/nn/"
    "sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_"
    "posminnz5_plateauguard_mixed_selection_compare_seed0_57.json"
)
DEFAULT_ORACLE_JSON = Path(
    "artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json"
)
DEFAULT_JSON_OUT = Path(
    "artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json"
)


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


def _classify_delta(value: float | None) -> str:
    if value is None:
        return "missing"
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "harmful"
    return "neutral"


def _mean(values: list[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return mean(kept) if kept else None


def _seed(row: dict[str, Any]) -> int:
    value = _as_int(row.get("seed"))
    if value is None:
        raise ValueError(f"Missing seed in row: {row}")
    return value


def _merge_rows(compare: dict[str, Any], oracle: dict[str, Any]) -> list[dict[str, Any]]:
    compare_rows = {
        _seed(row): row
        for row in compare.get("rows", [])
        if isinstance(row, dict)
    }
    oracle_rows = {
        _seed(row): row
        for row in oracle.get("rows", [])
        if isinstance(row, dict)
    }
    seeds = sorted(set(compare_rows) | set(oracle_rows))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        c = compare_rows.get(seed, {})
        o = oracle_rows.get(seed, {})
        candidate_delta = _as_float(
            o.get("candidate_delta_over_no_edit", c.get("eval_candidate_delta_over_no_edit"))
        )
        selected_delta = _as_float(
            o.get("selected_delta_over_no_edit", c.get("eval_selected_delta_over_no_edit"))
        )
        oracle_delta = _as_float(o.get("candidate_oracle_delta_over_no_edit"))
        validation_delta = _as_float(c.get("validation_delta_over_no_edit"))
        row = {
            "seed": seed,
            "selected_mode": o.get("selected_mode") or c.get("selected_mode"),
            "adoption_reason": o.get("adoption_reason") or c.get("adoption_reason"),
            "adoption_margin": _as_float(o.get("adoption_margin", c.get("adoption_margin"))),
            "adoption_nonzero": _as_int(o.get("adoption_nonzero", c.get("adoption_nonzero"))),
            "validation_delta_over_no_edit": validation_delta,
            "validation_delta_class": _classify_delta(validation_delta),
            "validation_improved": _as_int(c.get("validation_improved")),
            "validation_harmed": _as_int(c.get("validation_harmed")),
            "selected_delta_over_no_edit": selected_delta,
            "candidate_delta_over_no_edit": candidate_delta,
            "candidate_delta_class": _classify_delta(candidate_delta),
            "candidate_oracle_delta_over_no_edit": oracle_delta,
            "candidate_to_oracle_gap": _as_float(o.get("candidate_to_oracle_gap")),
            "candidate_improved": _as_int(o.get("candidate_improved", c.get("candidate_improved"))),
            "candidate_harmed": _as_int(o.get("candidate_harmed", c.get("candidate_harmed"))),
            "candidate_fraction_with_any_selected_edit": _as_float(
                o.get("candidate_fraction_with_any_selected_edit")
            ),
            "candidate_mean_selected_edit_weight": _as_float(
                o.get("candidate_mean_selected_edit_weight")
            ),
        }
        rows.append(row)
    return rows


def _seeds(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["seed"]) for row in rows]


def _sort_rows(rows: list[dict[str, Any]], key: str, *, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float("-inf") if row.get(key) is None else float(row[key]),
            -int(row["seed"]),
        ),
        reverse=reverse,
    )


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "seed",
        "selected_mode",
        "adoption_reason",
        "adoption_margin",
        "adoption_nonzero",
        "validation_delta_over_no_edit",
        "validation_improved",
        "validation_harmed",
        "candidate_delta_over_no_edit",
        "candidate_improved",
        "candidate_harmed",
        "candidate_oracle_delta_over_no_edit",
        "candidate_to_oracle_gap",
    ]
    return {key: row.get(key) for key in keys}


def _crosstab(rows: list[dict[str, Any]], left_key: str, right_key: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        left = str(row.get(left_key))
        right = str(row.get(right_key))
        out.setdefault(left, {})
        out[left][right] = out[left].get(right, 0) + 1
    return out


def _summarize(rows: list[dict[str, Any]], *, top_k: int, high_oracle_delta: float) -> dict[str, Any]:
    selected_positive = [
        row
        for row in rows
        if row.get("selected_mode") == "local_motif_selector"
        and (_as_float(row.get("selected_delta_over_no_edit")) or 0.0) > 0.0
    ]
    missed_candidate_positive = [
        row
        for row in rows
        if (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) > 0.0
        and (_as_float(row.get("selected_delta_over_no_edit")) or 0.0) <= 0.0
    ]
    harmful_candidate_blocked = [
        row
        for row in rows
        if (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) < 0.0
        and (_as_float(row.get("selected_delta_over_no_edit")) or 0.0) == 0.0
    ]
    neutral_candidate_with_oracle = [
        row
        for row in rows
        if (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) == 0.0
        and (_as_float(row.get("candidate_oracle_delta_over_no_edit")) or 0.0) > 0.0
    ]
    validation_false_positive = [
        row
        for row in rows
        if (_as_float(row.get("validation_delta_over_no_edit")) or 0.0) > 0.0
        and (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) < 0.0
    ]
    validation_true_positive = [
        row
        for row in rows
        if (_as_float(row.get("validation_delta_over_no_edit")) or 0.0) > 0.0
        and (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) > 0.0
    ]
    eval_positive_validation_nonpositive = [
        row
        for row in rows
        if (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) > 0.0
        and (_as_float(row.get("validation_delta_over_no_edit")) or 0.0) <= 0.0
    ]
    high_oracle_missed = [
        row
        for row in rows
        if (_as_float(row.get("candidate_oracle_delta_over_no_edit")) or 0.0)
        >= float(high_oracle_delta)
        and (_as_float(row.get("candidate_delta_over_no_edit")) or 0.0) <= 0.0
    ]

    return {
        "num_rows": len(rows),
        "mean_selected_delta_over_no_edit": _mean(
            [_as_float(row.get("selected_delta_over_no_edit")) for row in rows]
        ),
        "mean_candidate_delta_over_no_edit": _mean(
            [_as_float(row.get("candidate_delta_over_no_edit")) for row in rows]
        ),
        "mean_candidate_oracle_delta_over_no_edit": _mean(
            [_as_float(row.get("candidate_oracle_delta_over_no_edit")) for row in rows]
        ),
        "validation_vs_eval_candidate_crosstab": _crosstab(
            rows,
            "validation_delta_class",
            "candidate_delta_class",
        ),
        "selected_positive": {
            "count": len(selected_positive),
            "seeds": _seeds(selected_positive),
        },
        "missed_candidate_positive": {
            "count": len(missed_candidate_positive),
            "seeds": _seeds(missed_candidate_positive),
            "top_by_candidate_delta": [
                _compact_row(row)
                for row in _sort_rows(
                    missed_candidate_positive,
                    "candidate_delta_over_no_edit",
                )[:top_k]
            ],
        },
        "harmful_candidate_blocked": {
            "count": len(harmful_candidate_blocked),
            "seeds": _seeds(harmful_candidate_blocked),
            "top_by_harm": [
                _compact_row(row)
                for row in _sort_rows(
                    harmful_candidate_blocked,
                    "candidate_delta_over_no_edit",
                    reverse=False,
                )[:top_k]
            ],
        },
        "neutral_candidate_with_oracle": {
            "count": len(neutral_candidate_with_oracle),
            "seeds": _seeds(neutral_candidate_with_oracle),
        },
        "validation_false_positive": {
            "count": len(validation_false_positive),
            "seeds": _seeds(validation_false_positive),
            "top_by_validation_delta": [
                _compact_row(row)
                for row in _sort_rows(
                    validation_false_positive,
                    "validation_delta_over_no_edit",
                )[:top_k]
            ],
        },
        "validation_true_positive": {
            "count": len(validation_true_positive),
            "seeds": _seeds(validation_true_positive),
        },
        "eval_positive_validation_nonpositive": {
            "count": len(eval_positive_validation_nonpositive),
            "seeds": _seeds(eval_positive_validation_nonpositive),
        },
        "high_oracle_missed": {
            "threshold": float(high_oracle_delta),
            "count": len(high_oracle_missed),
            "seeds": _seeds(high_oracle_missed),
            "top_by_oracle_delta": [
                _compact_row(row)
                for row in _sort_rows(
                    high_oracle_missed,
                    "candidate_oracle_delta_over_no_edit",
                )[:top_k]
            ],
        },
        "recommended_sentinel_sets": {
            "preserve_true_positives": _seeds(selected_positive),
            "recover_missed_positives": _seeds(
                _sort_rows(missed_candidate_positive, "candidate_delta_over_no_edit")[:top_k]
            ),
            "block_validation_false_positives": _seeds(
                _sort_rows(validation_false_positive, "validation_delta_over_no_edit")[:top_k]
            ),
            "inspect_high_oracle_misses": _seeds(
                _sort_rows(high_oracle_missed, "candidate_oracle_delta_over_no_edit")[:top_k]
            ),
        },
        "interpretation": [
            "d7 has positive oracle headroom in every checked seed, but selected adoption recovers only sparse true positives.",
            "The next d7 objective should be tested on preserve/recover/block sentinel groups before any broad seed extension.",
            "A useful next objective must improve candidate ranking on missed-positive or high-oracle seeds while preserving seeds 2 and 11 and blocking validation false positives.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", type=Path, default=DEFAULT_COMPARE_JSON)
    parser.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--high-oracle-delta", type=float, default=0.12)
    args = parser.parse_args()

    compare = _load_json(args.compare_json)
    oracle = _load_json(args.oracle_json)
    rows = _merge_rows(compare, oracle)
    payload = {
        "schema_version": "d7_selector_bottleneck_targeted_summary.v1",
        "compare_json": args.compare_json.as_posix(),
        "oracle_json": args.oracle_json.as_posix(),
        "summary": _summarize(
            rows,
            top_k=int(args.top_k),
            high_oracle_delta=float(args.high_oracle_delta),
        ),
        "rows": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
