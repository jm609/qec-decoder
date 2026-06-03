from __future__ import annotations

"""
simulate_predecoder_adoption_policy.py

Post-hoc simulator for syndrome-edit pre-decoder adoption policies.

This does not mutate experiment artifacts. It reads existing experiment
summaries and asks which branch would be selected under a conservative
candidate-first policy.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json


SELECTOR_MODES = {"candidate_selector", "local_motif_selector", "local_motif_router"}


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    name: str
    template: str
    seeds: list[int]


@dataclass(frozen=True, slots=True)
class PolicyThresholds:
    strong_delta: float
    positive_delta: float
    positive_margin_floor: float
    positive_max_harmed: int
    positive_max_margin: float
    positive_min_nonzero: int
    positive_plateau_guard: bool
    positive_family_min_delta: float
    positive_min_family_count: int
    positive_max_family_harmed: int
    tie_min_delta: float
    tie_margin_floor: float
    tie_min_nonzero: int
    allow_global: bool
    global_min_delta: float


@dataclass(frozen=True, slots=True)
class AdoptionRow:
    suite: str
    seed: int
    run_dir: str
    original_mode: str
    original_selector_margin: float | None
    policy_mode: str
    policy_reason: str
    val_no_edit_accuracy: float
    val_global_delta: float | None
    val_candidate_delta: float | None
    val_candidate_nonzero: int | None
    val_candidate_improved: int | None
    val_candidate_harmed: int | None
    val_candidate_family_deltas: dict[str, float] | None
    val_candidate_family_nonzero: dict[str, int] | None
    val_candidate_family_improved: dict[str, int] | None
    val_candidate_family_harmed: dict[str, int] | None
    eval_no_edit_accuracy: float
    eval_original_delta: float | None
    eval_policy_delta: float
    eval_candidate_delta: float | None
    eval_global_delta: float | None


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for item in text.split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed must be provided")
    return seeds


def _parse_suite(text: str) -> SuiteSpec:
    parts = text.split(",", 2)
    if len(parts) != 3:
        raise ValueError(
            "Suite must be NAME,RUN_DIR_TEMPLATE,SEEDS; "
            f"got {text!r}"
        )
    name, template, seeds_text = (part.strip() for part in parts)
    if not name or not template:
        raise ValueError(f"Invalid suite specification: {text!r}")
    return SuiteSpec(name=name, template=template, seeds=_parse_seeds(seeds_text))


def _run_dir(root: Path, template: str, seed: int) -> Path:
    rendered = template.format(seed=seed)
    path = Path(rendered)
    if path.is_absolute():
        return path
    return root / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mean_accuracy(by_family: dict[str, Any] | None, families: list[str]) -> float | None:
    if by_family is None:
        return None
    values = []
    for family in families:
        if family not in by_family:
            raise ValueError(f"Missing validation family {family!r}")
        values.append(float(by_family[family]["edited_pymatching"]["accuracy"]))
    if not values:
        return None
    return sum(values) / len(values)


def _accuracy(section: dict[str, Any] | None, family: str) -> float | None:
    if section is None or family not in section:
        return None
    return float(section[family]["edited_pymatching"]["accuracy"])


def _nonzero_count(result: dict[str, Any]) -> int:
    change_summary = result.get("change_summary") or {}
    histogram = change_summary.get("predicted_edit_weight_histogram") or {}
    total = 0
    for key, value in histogram.items():
        if str(key) != "0":
            total += int(value)
    return int(total)


def _sum_change(by_family: dict[str, Any] | None, families: list[str], key: str) -> int | None:
    if by_family is None:
        return None
    total = 0
    for family in families:
        if family not in by_family:
            raise ValueError(f"Missing validation family {family!r}")
        change_summary = by_family[family].get("change_summary") or {}
        total += int(change_summary.get(key) or 0)
    return int(total)


def _sum_nonzero(by_family: dict[str, Any] | None, families: list[str]) -> int | None:
    if by_family is None:
        return None
    return int(sum(_nonzero_count(by_family[family]) for family in families))


def _family_candidate_stats(
    *,
    no_edit_by_family: dict[str, Any] | None,
    candidate_by_family: dict[str, Any] | None,
    families: list[str],
) -> tuple[dict[str, float], dict[str, int], dict[str, int], dict[str, int]]:
    if no_edit_by_family is None or candidate_by_family is None:
        return {}, {}, {}, {}
    deltas: dict[str, float] = {}
    nonzero: dict[str, int] = {}
    improved: dict[str, int] = {}
    harmed: dict[str, int] = {}
    for family in families:
        if family not in no_edit_by_family or family not in candidate_by_family:
            raise ValueError(f"Missing validation family {family!r}")
        no_edit_acc = float(no_edit_by_family[family]["edited_pymatching"]["accuracy"])
        candidate_acc = float(candidate_by_family[family]["edited_pymatching"]["accuracy"])
        change_summary = candidate_by_family[family].get("change_summary") or {}
        deltas[family] = float(candidate_acc - no_edit_acc)
        nonzero[family] = _nonzero_count(candidate_by_family[family])
        improved[family] = int(change_summary.get("num_improved_over_baseline") or 0)
        harmed[family] = int(change_summary.get("num_harmed_vs_baseline") or 0)
    return deltas, nonzero, improved, harmed


def _selector_margin_profile(by_family: dict[str, Any] | None, families: list[str]) -> list[dict[str, Any]]:
    if by_family is None:
        return []
    for family in families:
        if family not in by_family:
            continue
        result = by_family.get(family) or {}
        decision = result.get("decision") or {}
        profile = decision.get("selector_margin_profile")
        if isinstance(profile, list):
            return [dict(row) for row in profile if isinstance(row, dict)]
    return []


def _delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return float(value) - float(base)


def _original_eval_accuracy(summary: dict[str, Any], eval_family: str) -> float | None:
    return _accuracy(summary.get("eval_families"), eval_family)


def _choose_policy_mode(
    *,
    candidate_delta: float | None,
    candidate_nonzero: int | None,
    candidate_harmed: int | None,
    selector_margin: float | None,
    selector_margin_profile: list[dict[str, Any]],
    candidate_family_deltas: dict[str, float] | None,
    candidate_family_harmed: dict[str, int] | None,
    global_delta: float | None,
    thresholds: PolicyThresholds,
) -> tuple[str, str]:
    cand_delta = None if candidate_delta is None else float(candidate_delta)
    cand_nonzero = int(candidate_nonzero or 0)
    cand_harmed = int(candidate_harmed or 0)
    margin = float(selector_margin or 0.0)
    if cand_delta is not None and cand_nonzero > 0:
        if cand_delta >= float(thresholds.strong_delta):
            return "local_motif_selector", "candidate_strong_validation_delta"
        positive_harm_guard_failed = (
            int(thresholds.positive_max_harmed) >= 0
            and cand_delta >= float(thresholds.positive_delta)
            and cand_harmed > int(thresholds.positive_max_harmed)
        )
        positive_margin_guard_failed = (
            float(thresholds.positive_max_margin) >= 0.0
            and cand_delta >= float(thresholds.positive_delta)
            and margin > float(thresholds.positive_max_margin)
        )
        positive_support_guard_failed = (
            int(thresholds.positive_min_nonzero) > 0
            and cand_delta >= float(thresholds.positive_delta)
            and cand_nonzero < int(thresholds.positive_min_nonzero)
        )
        positive_plateau_guard_failed = False
        if bool(thresholds.positive_plateau_guard) and cand_delta >= float(thresholds.positive_delta):
            for row in selector_margin_profile:
                row_margin = row.get("selector_emit_margin")
                row_delta = row.get("validation_delta_over_no_edit")
                if row_margin is None or row_delta is None:
                    continue
                if (
                    float(row_margin) > margin + 1e-12
                    and float(row_delta) >= float(thresholds.positive_delta)
                ):
                    positive_plateau_guard_failed = True
                    break
        positive_family_guard_failed = False
        if cand_delta >= float(thresholds.positive_delta):
            if int(thresholds.positive_min_family_count) > 0:
                family_deltas = candidate_family_deltas or {}
                family_positive_count = sum(
                    1
                    for delta in family_deltas.values()
                    if float(delta) >= float(thresholds.positive_family_min_delta)
                )
                if family_positive_count < int(thresholds.positive_min_family_count):
                    positive_family_guard_failed = True
            if int(thresholds.positive_max_family_harmed) >= 0:
                family_harmed = candidate_family_harmed or {}
                if any(
                    int(value) > int(thresholds.positive_max_family_harmed)
                    for value in family_harmed.values()
                ):
                    positive_family_guard_failed = True
        positive_guard_failed = (
            positive_harm_guard_failed
            or positive_margin_guard_failed
            or positive_support_guard_failed
            or positive_plateau_guard_failed
            or positive_family_guard_failed
        )
        if (
            cand_delta >= float(thresholds.positive_delta)
            and margin >= float(thresholds.positive_margin_floor)
            and not positive_guard_failed
        ):
            return "local_motif_selector", "candidate_positive_delta_with_margin"
        if (
            cand_delta >= float(thresholds.tie_min_delta)
            and margin >= float(thresholds.tie_margin_floor)
            and cand_nonzero >= int(thresholds.tie_min_nonzero)
            and not positive_guard_failed
        ):
            return "local_motif_selector", "candidate_tie_with_high_margin_evidence"
        if positive_harm_guard_failed:
            return "raw_no_edit", "candidate_positive_delta_harm_guard"
        if positive_margin_guard_failed:
            return "raw_no_edit", "candidate_positive_delta_margin_guard"
        if positive_plateau_guard_failed:
            return "raw_no_edit", "candidate_positive_delta_plateau_guard"
        if positive_family_guard_failed:
            return "raw_no_edit", "candidate_positive_delta_family_guard"
        if positive_support_guard_failed:
            return "raw_no_edit", "candidate_positive_delta_support_guard"

    if (
        bool(thresholds.allow_global)
        and global_delta is not None
        and float(global_delta) >= float(thresholds.global_min_delta)
    ):
        return "global_policy", "global_delta_clears_guard"
    return "raw_no_edit", "default_no_edit"


def _row_from_summary(
    *,
    suite: str,
    seed: int,
    run_dir: Path,
    summary: dict[str, Any],
    validation_families: list[str],
    eval_family: str,
    thresholds: PolicyThresholds,
) -> AdoptionRow:
    training = summary["training"]
    selector_training = training.get("selector_training") or {}
    original_mode = str(training.get("selected_inference_mode", ""))
    selector_margin_raw = selector_training.get("selected_selector_emit_margin")
    selector_margin = None if selector_margin_raw is None else float(selector_margin_raw)

    val_no_edit = _mean_accuracy(training.get("best_val_no_edit_by_family"), validation_families)
    val_no_edit_by_family = training.get("best_val_no_edit_by_family")
    val_global = _mean_accuracy(training.get("best_val_global_policy_by_family"), validation_families)
    val_candidate_by_family = training.get("best_val_candidate_selector_by_family")
    val_candidate = _mean_accuracy(val_candidate_by_family, validation_families)
    if val_no_edit is None:
        raise ValueError(f"Missing no-edit validation metric for {run_dir}")

    global_delta = _delta(val_global, val_no_edit)
    candidate_delta = _delta(val_candidate, val_no_edit)
    candidate_nonzero = _sum_nonzero(val_candidate_by_family, validation_families)
    candidate_improved = _sum_change(
        val_candidate_by_family,
        validation_families,
        "num_improved_over_baseline",
    )
    candidate_harmed = _sum_change(
        val_candidate_by_family,
        validation_families,
        "num_harmed_vs_baseline",
    )
    (
        candidate_family_deltas,
        candidate_family_nonzero,
        candidate_family_improved,
        candidate_family_harmed,
    ) = _family_candidate_stats(
        no_edit_by_family=val_no_edit_by_family,
        candidate_by_family=val_candidate_by_family,
        families=validation_families,
    )
    candidate_margin_profile = _selector_margin_profile(val_candidate_by_family, validation_families)

    policy_mode, policy_reason = _choose_policy_mode(
        candidate_delta=candidate_delta,
        candidate_nonzero=candidate_nonzero,
        candidate_harmed=candidate_harmed,
        selector_margin=selector_margin,
        selector_margin_profile=candidate_margin_profile,
        candidate_family_deltas=candidate_family_deltas,
        candidate_family_harmed=candidate_family_harmed,
        global_delta=global_delta,
        thresholds=thresholds,
    )

    eval_no_edit = _accuracy(summary.get("eval_families_no_edit"), eval_family)
    eval_candidate = _accuracy(summary.get("eval_families_candidate_selector"), eval_family)
    eval_global = _accuracy(summary.get("eval_families_global_policy"), eval_family)
    eval_original = _original_eval_accuracy(summary, eval_family)
    if eval_no_edit is None:
        raise ValueError(f"Missing no-edit eval metric for {run_dir}")
    if policy_mode in SELECTOR_MODES:
        policy_eval = eval_candidate
    elif policy_mode == "global_policy":
        policy_eval = eval_global
    else:
        policy_eval = eval_no_edit
    if policy_eval is None:
        raise ValueError(f"Missing eval branch {policy_mode!r} for {run_dir}")

    return AdoptionRow(
        suite=str(suite),
        seed=int(seed),
        run_dir=run_dir.as_posix(),
        original_mode=original_mode,
        original_selector_margin=selector_margin,
        policy_mode=policy_mode,
        policy_reason=policy_reason,
        val_no_edit_accuracy=float(val_no_edit),
        val_global_delta=global_delta,
        val_candidate_delta=candidate_delta,
        val_candidate_nonzero=candidate_nonzero,
        val_candidate_improved=candidate_improved,
        val_candidate_harmed=candidate_harmed,
        val_candidate_family_deltas=candidate_family_deltas or None,
        val_candidate_family_nonzero=candidate_family_nonzero or None,
        val_candidate_family_improved=candidate_family_improved or None,
        val_candidate_family_harmed=candidate_family_harmed or None,
        eval_no_edit_accuracy=float(eval_no_edit),
        eval_original_delta=_delta(eval_original, eval_no_edit),
        eval_policy_delta=float(policy_eval) - float(eval_no_edit),
        eval_candidate_delta=_delta(eval_candidate, eval_no_edit),
        eval_global_delta=_delta(eval_global, eval_no_edit),
    )


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _summaries(rows: list[AdoptionRow]) -> list[dict[str, Any]]:
    suites = sorted({row.suite for row in rows})
    out: list[dict[str, Any]] = []
    for suite in suites:
        suite_rows = [row for row in rows if row.suite == suite]
        out.append(
            {
                "suite": suite,
                "num_rows": len(suite_rows),
                "mean_original_delta": _mean([row.eval_original_delta for row in suite_rows]),
                "mean_policy_delta": _mean([row.eval_policy_delta for row in suite_rows]),
                "mean_candidate_delta": _mean([row.eval_candidate_delta for row in suite_rows]),
                "policy_mode_counts": {
                    mode: sum(1 for row in suite_rows if row.policy_mode == mode)
                    for mode in sorted({row.policy_mode for row in suite_rows})
                },
            }
        )
    return out


def _print_summary(rows: list[AdoptionRow]) -> None:
    for summary in _summaries(rows):
        print(
            f"{summary['suite']}: original={summary['mean_original_delta']:+.9f} "
            f"policy={summary['mean_policy_delta']:+.9f} "
            f"candidate={summary['mean_candidate_delta']:+.9f} "
            f"modes={summary['policy_mode_counts']}"
        )
    print("rows:")
    for row in rows:
        print(
            f"{row.suite} seed={row.seed} original={row.original_mode} "
            f"margin={row.original_selector_margin} policy={row.policy_mode} "
            f"reason={row.policy_reason} val_cand={row.val_candidate_delta:+.9f} "
            f"nz={row.val_candidate_nonzero} eval_policy={row.eval_policy_delta:+.9f} "
            f"eval_candidate={row.eval_candidate_delta:+.9f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/eval/nn"))
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        help="NAME,RUN_DIR_TEMPLATE,SEEDS. Example: d3,sedp_d3_seed{seed},0-3",
    )
    parser.add_argument("--eval-family", default="stage_c_corr")
    parser.add_argument("--validation-families", default=None)
    parser.add_argument("--strong-delta", type=float, default=0.02)
    parser.add_argument("--positive-delta", type=float, default=0.005)
    parser.add_argument("--positive-margin-floor", type=float, default=0.5)
    parser.add_argument("--positive-max-harmed", type=int, default=2)
    parser.add_argument("--positive-max-margin", type=float, default=1.5)
    parser.add_argument("--positive-min-nonzero", type=int, default=0)
    parser.add_argument("--positive-plateau-guard", action="store_true")
    parser.add_argument("--positive-family-min-delta", type=float, default=0.0)
    parser.add_argument("--positive-min-family-count", type=int, default=0)
    parser.add_argument("--positive-max-family-harmed", type=int, default=-1)
    parser.add_argument("--tie-min-delta", type=float, default=-1e-9)
    parser.add_argument("--tie-margin-floor", type=float, default=1.0)
    parser.add_argument("--tie-min-nonzero", type=int, default=6)
    parser.add_argument("--allow-global", action="store_true")
    parser.add_argument("--global-min-delta", type=float, default=0.01)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    suites = [_parse_suite(text) for text in args.suite]
    thresholds = PolicyThresholds(
        strong_delta=float(args.strong_delta),
        positive_delta=float(args.positive_delta),
        positive_margin_floor=float(args.positive_margin_floor),
        positive_max_harmed=int(args.positive_max_harmed),
        positive_max_margin=float(args.positive_max_margin),
        positive_min_nonzero=int(args.positive_min_nonzero),
        positive_plateau_guard=bool(args.positive_plateau_guard),
        positive_family_min_delta=float(args.positive_family_min_delta),
        positive_min_family_count=int(args.positive_min_family_count),
        positive_max_family_harmed=int(args.positive_max_family_harmed),
        tie_min_delta=float(args.tie_min_delta),
        tie_margin_floor=float(args.tie_margin_floor),
        tie_min_nonzero=int(args.tie_min_nonzero),
        allow_global=bool(args.allow_global),
        global_min_delta=float(args.global_min_delta),
    )

    rows: list[AdoptionRow] = []
    for suite in suites:
        for seed in suite.seeds:
            run_dir = _run_dir(args.root, suite.template, seed)
            summary = _load_json(run_dir / "experiment_summary.json")
            if args.validation_families:
                validation_families = [
                    item.strip()
                    for item in str(args.validation_families).split(",")
                    if item.strip()
                ]
            else:
                no_edit = summary["training"].get("best_val_no_edit_by_family")
                if not isinstance(no_edit, dict):
                    raise ValueError("Could not infer validation families")
                validation_families = sorted(no_edit)
            rows.append(
                _row_from_summary(
                    suite=suite.name,
                    seed=seed,
                    run_dir=run_dir,
                    summary=summary,
                    validation_families=validation_families,
                    eval_family=args.eval_family,
                    thresholds=thresholds,
                )
            )

    _print_summary(rows)
    if args.json_out is not None:
        payload = {
            "schema_version": "predecoder_adoption_policy_sim.v1",
            "eval_family": str(args.eval_family),
            "thresholds": asdict(thresholds),
            "summaries": _summaries(rows),
            "rows": [asdict(row) for row in rows],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
