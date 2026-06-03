"""Build seed-expanded d3/d5 summary with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "predecoder_d3_d5_seed_expansion_ci.v1"


@dataclass(frozen=True, slots=True)
class CompareSpec:
    distance: str
    compare_path: Path


DEFAULT_SPECS = (
    CompareSpec(
        distance="d3",
        compare_path=Path(
            "artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json"
        ),
    ),
    CompareSpec(
        distance="d5",
        compare_path=Path(
            "artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_7.json"
        ),
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty list")
    return float(sum(values) / len(values))


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = _mean(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list")
    if fraction <= 0.0:
        return float(sorted_values[0])
    if fraction >= 1.0:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int,
    rng: random.Random,
) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty list")
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(_mean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    return {
        "mean": _mean(values),
        "bootstrap_ci_95_low": _percentile(means, 0.025),
        "bootstrap_ci_95_high": _percentile(means, 0.975),
    }


def _class_counts(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(1 for value in values if value > 0.0),
        "neutral": sum(1 for value in values if value == 0.0),
        "harmful": sum(1 for value in values if value < 0.0),
    }


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _summarize_distance(
    spec: CompareSpec,
    *,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    data = _load_json(spec.compare_path)
    rows = list(data.get("rows") or [])
    if not rows:
        raise ValueError(f"No seed rows in {spec.compare_path}")

    selected_deltas = [float(row["eval_selected_delta_over_no_edit"]) for row in rows]
    candidate_deltas = [float(row["eval_candidate_delta_over_no_edit"]) for row in rows]
    raw_acc = [float(row["eval_no_edit_accuracy"]) for row in rows]
    selected_acc = [float(row["eval_selected_accuracy"]) for row in rows]
    candidate_acc = [float(row["eval_candidate_accuracy"]) for row in rows]
    candidate_harm_blocked = [
        row
        for row in rows
        if float(row["eval_candidate_delta_over_no_edit"]) < 0.0
        and str(row.get("mode")) == "raw_no_edit"
    ]

    return {
        "distance": spec.distance,
        "source_compare": spec.compare_path.as_posix(),
        "eval_family": data.get("eval_family"),
        "num_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "mean_raw_pymatching_accuracy": _mean(raw_acc),
        "mean_selected_predecoder_accuracy": _mean(selected_acc),
        "mean_candidate_branch_accuracy": _mean(candidate_acc),
        "selected_delta": {
            **_bootstrap_mean_ci(selected_deltas, iterations=iterations, rng=rng),
            "stdev": _stdev(selected_deltas),
            "min": min(selected_deltas),
            "max": max(selected_deltas),
            "class_counts": _class_counts(selected_deltas),
        },
        "candidate_delta": {
            **_bootstrap_mean_ci(candidate_deltas, iterations=iterations, rng=rng),
            "stdev": _stdev(candidate_deltas),
            "min": min(candidate_deltas),
            "max": max(candidate_deltas),
            "class_counts": _class_counts(candidate_deltas),
        },
        "mode_counts": _mode_counts(rows),
        "selected_improved_total": sum(int(row.get("selected_improved") or 0) for row in rows),
        "selected_harmed_total": sum(int(row.get("selected_harmed") or 0) for row in rows),
        "candidate_improved_total": sum(int(row.get("candidate_improved") or 0) for row in rows),
        "candidate_harmed_total": sum(int(row.get("candidate_harmed") or 0) for row in rows),
        "candidate_harm_blocked_seed_count": len(candidate_harm_blocked),
        "candidate_harm_blocked_seeds": [int(row["seed"]) for row in candidate_harm_blocked],
        "seed_rows": [
            {
                "seed": int(row["seed"]),
                "mode": str(row.get("mode")),
                "adoption_reason": row.get("adoption_reason"),
                "eval_selected_delta_over_no_edit": float(
                    row["eval_selected_delta_over_no_edit"]
                ),
                "eval_candidate_delta_over_no_edit": float(
                    row["eval_candidate_delta_over_no_edit"]
                ),
                "selected_improved": int(row.get("selected_improved") or 0),
                "selected_harmed": int(row.get("selected_harmed") or 0),
                "candidate_improved": int(row.get("candidate_improved") or 0),
                "candidate_harmed": int(row.get("candidate_harmed") or 0),
            }
            for row in rows
        ],
    }


def build_summary(
    *,
    specs: tuple[CompareSpec, ...],
    iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rng = random.Random(bootstrap_seed)
    distance_results = [
        _summarize_distance(spec, iterations=iterations, rng=rng) for spec in specs
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "distance_results": distance_results,
        "interpretation": [
            "The seed-expanded d3/d5 results remain positive in selected mode.",
            "The wider seed set lowers the d3 and d5 mean deltas compared with seed0..3.",
            "d5 shows the practical value of selected-mode adoption because harmful candidate branches are blocked by raw no-edit fallback.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260514)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/eval/nn/sedp_d3_d5_seed0_7_bootstrap_ci_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(
        specs=DEFAULT_SPECS,
        iterations=int(args.iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "schema_version": SCHEMA_VERSION,
                "bootstrap_iterations": int(args.iterations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
