"""Search simple d7 sentinel adoption policies over existing comparison rows.

This is a diagnostic, not a new decoder. It asks whether the current d7
candidate branch could pass the preserve/recover/block sentinel gate by
changing only validation-side adoption thresholds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_JSON = Path(
    "artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json"
)
DEFAULT_JSON_OUT = Path(
    "artifacts/eval/nn/sedp_d7_sentinel_adoption_grid_summary.json"
)


@dataclass(frozen=True, slots=True)
class Policy:
    min_validation_delta: float
    min_nonzero: int
    max_validation_harmed: int
    min_validation_improved: int
    min_validation_net: int
    min_margin: float
    max_margin: float


@dataclass(frozen=True, slots=True)
class PolicyResult:
    policy: Policy
    adopted_seeds: list[int]
    preserve_selected: list[int]
    preserve_missed: list[int]
    recover_selected: list[int]
    recover_missed: list[int]
    harmful_block_adopted: list[int]
    harmful_blocked: list[int]
    adopted_harmful_all: list[int]
    mean_selected_delta_all_rows: float
    gate_pass: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    return int(value)


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _unique_sorted(values: list[float]) -> list[float]:
    return sorted({round(float(v), 12) for v in values})


def _policy_grid(rows: list[dict[str, Any]], sentinel_seeds: set[int]) -> list[Policy]:
    sentinel_rows = [row for row in rows if int(row["seed"]) in sentinel_seeds]
    observed_deltas = [
        _as_float(row.get("validation_delta_over_no_edit"))
        for row in sentinel_rows
        if row.get("validation_delta_over_no_edit") is not None
    ]
    observed_margins = [
        _as_float(row.get("adoption_margin"))
        for row in sentinel_rows
        if row.get("adoption_margin") is not None
    ]
    delta_grid = _unique_sorted(
        [
            0.0,
            1e-9,
            0.001,
            0.003,
            0.005,
            0.006,
            0.00649,
            0.007,
            0.009,
            *observed_deltas,
        ]
    )
    min_margin_grid = _unique_sorted([0.0, 1.0, 1.25, *observed_margins])
    max_margin_grid = _unique_sorted([1.25, 1.5, 2.0, 99.0, *observed_margins])
    policies: list[Policy] = []
    for min_delta in delta_grid:
        for min_nonzero in (0, 1, 2, 5, 6):
            for max_harmed in (0, 1, 2, 3):
                for min_improved in (0, 1, 2, 4):
                    for min_net in (-2, 0, 1, 2):
                        for min_margin in min_margin_grid:
                            for max_margin in max_margin_grid:
                                if float(min_margin) > float(max_margin):
                                    continue
                                policies.append(
                                    Policy(
                                        min_validation_delta=float(min_delta),
                                        min_nonzero=int(min_nonzero),
                                        max_validation_harmed=int(max_harmed),
                                        min_validation_improved=int(min_improved),
                                        min_validation_net=int(min_net),
                                        min_margin=float(min_margin),
                                        max_margin=float(max_margin),
                                    )
                                )
    return policies


def _adopts(row: dict[str, Any], policy: Policy) -> bool:
    val_delta = _as_float(row.get("validation_delta_over_no_edit"))
    nonzero = _as_int(row.get("adoption_nonzero"))
    harmed = _as_int(row.get("validation_harmed"))
    improved = _as_int(row.get("validation_improved"))
    margin = _as_float(row.get("adoption_margin"))
    return (
        val_delta >= float(policy.min_validation_delta)
        and nonzero >= int(policy.min_nonzero)
        and harmed <= int(policy.max_validation_harmed)
        and improved >= int(policy.min_validation_improved)
        and (improved - harmed) >= int(policy.min_validation_net)
        and margin >= float(policy.min_margin)
        and margin <= float(policy.max_margin)
    )


def _evaluate_policy(
    rows: list[dict[str, Any]],
    policy: Policy,
    *,
    preserve: set[int],
    recover: set[int],
    block: set[int],
    min_recover: int,
) -> PolicyResult:
    row_by_seed = {int(row["seed"]): row for row in rows}
    adopted: list[int] = []
    selected_deltas: list[float] = []
    for row in rows:
        seed = int(row["seed"])
        candidate_delta = _as_float(row.get("candidate_delta_over_no_edit"))
        if _adopts(row, policy):
            adopted.append(seed)
            selected_deltas.append(candidate_delta)
        else:
            selected_deltas.append(0.0)

    preserve_selected = [
        seed
        for seed in sorted(preserve)
        if seed in row_by_seed
        and seed in adopted
        and _as_float(row_by_seed[seed].get("candidate_delta_over_no_edit")) > 0.0
    ]
    recover_selected = [
        seed
        for seed in sorted(recover)
        if seed in row_by_seed
        and seed in adopted
        and _as_float(row_by_seed[seed].get("candidate_delta_over_no_edit")) > 0.0
    ]
    harmful_block_adopted = [
        seed
        for seed in sorted(block)
        if seed in row_by_seed
        and seed in adopted
        and _as_float(row_by_seed[seed].get("candidate_delta_over_no_edit")) < 0.0
    ]
    adopted_harmful_all = [
        seed
        for seed in adopted
        if _as_float(row_by_seed[seed].get("candidate_delta_over_no_edit")) < 0.0
    ]
    gate_pass = (
        len(preserve_selected) == len(preserve)
        and len(recover_selected) >= int(min_recover)
        and not harmful_block_adopted
    )
    return PolicyResult(
        policy=policy,
        adopted_seeds=sorted(adopted),
        preserve_selected=preserve_selected,
        preserve_missed=sorted(seed for seed in preserve if seed not in preserve_selected),
        recover_selected=recover_selected,
        recover_missed=sorted(seed for seed in recover if seed not in recover_selected),
        harmful_block_adopted=harmful_block_adopted,
        harmful_blocked=sorted(seed for seed in block if seed not in harmful_block_adopted),
        adopted_harmful_all=sorted(adopted_harmful_all),
        mean_selected_delta_all_rows=(
            float(sum(selected_deltas) / len(selected_deltas)) if selected_deltas else 0.0
        ),
        gate_pass=gate_pass,
    )


def _result_sort_key(result: PolicyResult) -> tuple[Any, ...]:
    return (
        int(result.gate_pass),
        len(result.preserve_selected),
        len(result.recover_selected),
        -len(result.harmful_block_adopted),
        -len(result.adopted_harmful_all),
        result.mean_selected_delta_all_rows,
        -len(result.adopted_seeds),
    )


def _compact_result(result: PolicyResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["policy"] = asdict(result.policy)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--preserve-seeds", default="2,11")
    parser.add_argument("--recover-seeds", default="0,28,43,45")
    parser.add_argument("--block-seeds", default="13,17,33,54,53,32,8,18")
    parser.add_argument("--min-recover", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    payload = _load_json(args.summary_json)
    rows = [dict(row) for row in payload.get("rows", []) if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No rows found in {args.summary_json}")

    preserve = set(_parse_int_list(args.preserve_seeds))
    recover = set(_parse_int_list(args.recover_seeds))
    block = set(_parse_int_list(args.block_seeds))
    sentinel_seeds = set(preserve) | set(recover) | set(block)
    policies = _policy_grid(rows, sentinel_seeds)
    results = [
        _evaluate_policy(
            rows,
            policy,
            preserve=preserve,
            recover=recover,
            block=block,
            min_recover=int(args.min_recover),
        )
        for policy in policies
    ]
    results.sort(key=_result_sort_key, reverse=True)
    passing = [result for result in results if result.gate_pass]
    best = results[: int(args.top_k)]
    out = {
        "schema_version": "d7_sentinel_adoption_grid.v1",
        "summary_json": args.summary_json.as_posix(),
        "num_rows": len(rows),
        "num_policies_checked": len(results),
        "sentinel_sets": {
            "preserve": sorted(preserve),
            "recover": sorted(recover),
            "block": sorted(block),
            "min_recover": int(args.min_recover),
        },
        "num_passing_policies": len(passing),
        "best_results": [_compact_result(result) for result in best],
        "best_passing_results": [
            _compact_result(result) for result in passing[: int(args.top_k)]
        ],
        "interpretation": [
            (
                "At least one simple adoption threshold policy passes the sentinel gate."
                if passing
                else "No simple monotone adoption-threshold policy passes the sentinel gate."
            ),
            "If no policy passes, d7 needs a selector-ranking change rather than another adoption-threshold sweep.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "num_policies_checked": len(results),
                "num_passing_policies": len(passing),
                "best_result": _compact_result(best[0]) if best else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
