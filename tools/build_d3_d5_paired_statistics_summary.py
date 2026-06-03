"""Build compact paired statistics for the d3/d5 selected deltas."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "predecoder_d3_d5_paired_statistics.v1"
DEFAULT_CI_JSON = Path("artifacts/eval/nn/sedp_d3_d5_seed0_7_bootstrap_ci_summary.json")
DEFAULT_JSON_OUT = Path("artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _exact_sign_test(deltas: list[float]) -> dict[str, Any]:
    nonzero = [value for value in deltas if value != 0.0]
    n = len(nonzero)
    positives = sum(1 for value in nonzero if value > 0.0)
    negatives = sum(1 for value in nonzero if value < 0.0)
    if n == 0:
        return {
            "num_nonzero": 0,
            "num_positive": 0,
            "num_negative": 0,
            "one_sided_positive_p": 1.0,
            "two_sided_p": 1.0,
        }

    total = 2**n
    one_sided_count = sum(math.comb(n, k) for k in range(positives, n + 1))
    lower_count = sum(math.comb(n, k) for k in range(0, positives + 1))
    upper_count = one_sided_count
    two_sided = min(1.0, 2.0 * min(lower_count / total, upper_count / total))
    return {
        "num_nonzero": n,
        "num_positive": positives,
        "num_negative": negatives,
        "one_sided_positive_p": one_sided_count / total,
        "two_sided_p": two_sided,
    }


def _exact_sign_flip_mean_test(deltas: list[float]) -> dict[str, Any]:
    nonzero = [float(value) for value in deltas if value != 0.0]
    n = len(nonzero)
    observed_sum = sum(nonzero)
    observed_mean = sum(deltas) / len(deltas) if deltas else 0.0
    if n == 0:
        return {
            "num_nonzero": 0,
            "observed_mean": observed_mean,
            "one_sided_positive_p": 1.0,
            "two_sided_p": 1.0,
        }

    total = 2**n
    one_sided_count = 0
    two_sided_count = 0
    observed_abs_sum = abs(observed_sum)
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        signed_sum = sum(sign * value for sign, value in zip(signs, nonzero))
        if signed_sum >= observed_sum:
            one_sided_count += 1
        if abs(signed_sum) >= observed_abs_sum:
            two_sided_count += 1
    return {
        "num_nonzero": n,
        "observed_mean": observed_mean,
        "one_sided_positive_p": one_sided_count / total,
        "two_sided_p": two_sided_count / total,
    }


def _summarize_distance(row: dict[str, Any]) -> dict[str, Any]:
    seed_rows = [seed_row for seed_row in row.get("seed_rows") or [] if isinstance(seed_row, dict)]
    deltas = [float(seed_row["eval_selected_delta_over_no_edit"]) for seed_row in seed_rows]
    seed_delta_rows = [
        {
            "seed": int(seed_row["seed"]),
            "mode": str(seed_row["mode"]),
            "selected_delta": float(seed_row["eval_selected_delta_over_no_edit"]),
            "candidate_delta": float(seed_row["eval_candidate_delta_over_no_edit"]),
            "adoption_reason": str(seed_row["adoption_reason"]),
        }
        for seed_row in seed_rows
    ]
    sign = _exact_sign_test(deltas)
    sign_flip = _exact_sign_flip_mean_test(deltas)
    distance = str(row["distance"])
    if distance == "d3":
        interpretation = (
            "All nonzero seed-level selected deltas are positive; the exact sign and "
            "sign-flip tests support a positive d3 effect over the checked seeds."
        )
    elif distance == "d5":
        interpretation = (
            "Only two seeds have nonzero selected deltas and both are positive, while "
            "six seeds fall back to raw no-edit. Treat d5 as positive-mean and "
            "non-harmful, not as a strong statistical significance claim."
        )
    else:
        interpretation = "Paired seed-level selected-delta summary."
    return {
        "distance": distance,
        "num_seeds": len(deltas),
        "mean_selected_delta": float(row["selected_delta"]["mean"]),
        "bootstrap_ci_95_low": float(row["selected_delta"]["bootstrap_ci_95_low"]),
        "bootstrap_ci_95_high": float(row["selected_delta"]["bootstrap_ci_95_high"]),
        "selected_delta_class_counts": dict(row["selected_delta"]["class_counts"]),
        "mode_counts": dict(row["mode_counts"]),
        "sign_test_excluding_zero_deltas": sign,
        "exact_sign_flip_mean_test_excluding_zero_deltas": sign_flip,
        "seed_delta_rows": seed_delta_rows,
        "interpretation": interpretation,
    }


def build_summary(ci_json: Path) -> dict[str, Any]:
    data = _load_json(ci_json)
    distance_results = [
        _summarize_distance(row)
        for row in data.get("distance_results") or []
        if isinstance(row, dict) and row.get("distance") in {"d3", "d5"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_artifact": ci_json.as_posix(),
        "purpose": (
            "Exact paired seed-level tests for selected predecoder deltas over raw "
            "PyMatching. These tests supplement, but do not replace, the bootstrap "
            "confidence intervals."
        ),
        "distance_results": distance_results,
        "interpretation": [
            "d3 is uniformly positive over the eight checked seeds.",
            "d5 has a positive mean and no harmful selected seed, but six zero fallback seeds make the statistical claim conservative.",
            "Use these values to avoid overstating d5 as a strong significance result.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-json", type=Path, default=DEFAULT_CI_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    summary = build_summary(args.ci_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "schema_version": SCHEMA_VERSION,
                "distances": [row["distance"] for row in summary["distance_results"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
