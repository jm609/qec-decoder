from __future__ import annotations

"""
compare_predecoder_seed_sweep.py

Summarize syndrome-edit pre-decoder seed sweeps and compare model-selection
criteria against no-edit PyMatching behavior.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import argparse
import json


SELECTOR_MODES = {
    "candidate_selector",
    "local_motif_selector",
    "local_motif_router",
}


@dataclass(frozen=True, slots=True)
class SeedRow:
    seed: int
    run_dir: str
    mode: str
    best_epoch: int | None
    selector_best_epoch: int | None
    selector_epoch_selection_mode: str | None
    adoption_reason: str | None
    adoption_margin: float | None
    adoption_nonzero: int | None
    candidate_local_motif_num_patterns: int | None
    val_no_edit_metric: float | None
    val_selected_metric: float | None
    val_candidate_metric: float | None
    val_global_metric: float | None
    val_selected_delta_over_no_edit: float | None
    val_candidate_delta_over_no_edit: float | None
    eval_no_edit_accuracy: float
    eval_selected_accuracy: float
    eval_candidate_accuracy: float | None
    eval_selected_delta_over_no_edit: float
    eval_candidate_delta_over_no_edit: float | None
    selected_improved: int | None
    selected_harmed: int | None
    candidate_improved: int | None
    candidate_harmed: int | None


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
        raise ValueError("At least one seed must be provided.")
    return seeds


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _run_dir_from_template(root: Path, template: str, seed: int) -> Path:
    rendered = template.format(seed=seed)
    path = Path(rendered)
    if path.is_absolute():
        return path
    if path.parts[: len(root.parts)] == root.parts:
        return path
    return root / path


def _edited_metric(result: dict[str, Any], *, macro_f1_weight: float) -> float:
    metrics = result["edited_pymatching"]
    return float(metrics["accuracy"]) + float(macro_f1_weight) * float(metrics["macro_f1"])


def _mean_family_metric(
    by_family: dict[str, Any] | None,
    *,
    families: list[str],
    macro_f1_weight: float,
) -> float | None:
    if by_family is None:
        return None
    missing = [family for family in families if family not in by_family]
    if missing:
        raise ValueError(f"Missing validation families: {missing}")
    if not families:
        return None
    return sum(_edited_metric(by_family[family], macro_f1_weight=macro_f1_weight) for family in families) / len(families)


def _selected_val_by_family(summary: dict[str, Any]) -> dict[str, Any] | None:
    training = summary["training"]
    mode = str(training.get("selected_inference_mode", ""))
    if mode == "raw_no_edit":
        return training.get("best_val_no_edit_by_family")
    if mode in SELECTOR_MODES:
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


def _accuracy(section: dict[str, Any], family: str) -> float:
    return float(section[family]["edited_pymatching"]["accuracy"])


def _change_count(section: dict[str, Any] | None, family: str, key: str) -> int | None:
    if section is None or family not in section:
        return None
    summary = section[family].get("change_summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    return None if value is None else int(value)


def _delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return value - base


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.9f}"


def _fmt_count_pair(improved: int | None, harmed: int | None) -> str:
    if improved is None or harmed is None:
        return "n/a"
    return f"{improved}/{harmed}"


def _row_from_summary(
    *,
    seed: int,
    run_dir: Path,
    summary: dict[str, Any],
    validation_families: list[str],
    eval_family: str,
    macro_f1_weight: float,
) -> SeedRow:
    training = summary["training"]
    mode = str(training.get("selected_inference_mode", ""))
    selector_training = training.get("selector_training") or {}
    adoption_decision = training.get("selector_adoption_decision") or {}
    num_local_patterns = selector_training.get("candidate_local_motif_num_patterns")

    val_no_edit_by_family = training.get("best_val_no_edit_by_family")
    val_selected_by_family = _selected_val_by_family(summary)
    val_candidate_by_family = training.get("best_val_candidate_selector_by_family")
    val_global_by_family = training.get("best_val_global_policy_by_family")

    selected_eval = summary["eval_families"]
    no_edit_eval = summary.get("eval_families_no_edit")
    candidate_eval = summary.get("eval_families_candidate_selector")

    if no_edit_eval is None:
        no_edit_eval = selected_eval

    val_no_edit = _mean_family_metric(
        val_no_edit_by_family,
        families=validation_families,
        macro_f1_weight=macro_f1_weight,
    )
    val_selected = _mean_family_metric(
        val_selected_by_family,
        families=validation_families,
        macro_f1_weight=macro_f1_weight,
    )
    val_candidate = _mean_family_metric(
        val_candidate_by_family,
        families=validation_families,
        macro_f1_weight=macro_f1_weight,
    )
    val_global = _mean_family_metric(
        val_global_by_family,
        families=validation_families,
        macro_f1_weight=macro_f1_weight,
    )

    eval_no_edit = _accuracy(no_edit_eval, eval_family)
    eval_selected = _accuracy(selected_eval, eval_family)
    eval_candidate = None if candidate_eval is None else _accuracy(candidate_eval, eval_family)

    return SeedRow(
        seed=int(seed),
        run_dir=run_dir.as_posix(),
        mode=mode,
        best_epoch=training.get("best_epoch"),
        selector_best_epoch=selector_training.get("best_epoch"),
        selector_epoch_selection_mode=selector_training.get("selector_epoch_selection_mode"),
        adoption_reason=adoption_decision.get("reason"),
        adoption_margin=(
            None
            if adoption_decision.get("selector_emit_margin") is None
            else float(adoption_decision.get("selector_emit_margin"))
        ),
        adoption_nonzero=(
            None
            if adoption_decision.get("selector_nonzero_count") is None
            else int(adoption_decision.get("selector_nonzero_count"))
        ),
        candidate_local_motif_num_patterns=(
            None if num_local_patterns is None else int(num_local_patterns)
        ),
        val_no_edit_metric=val_no_edit,
        val_selected_metric=val_selected,
        val_candidate_metric=val_candidate,
        val_global_metric=val_global,
        val_selected_delta_over_no_edit=_delta(val_selected, val_no_edit),
        val_candidate_delta_over_no_edit=_delta(val_candidate, val_no_edit),
        eval_no_edit_accuracy=eval_no_edit,
        eval_selected_accuracy=eval_selected,
        eval_candidate_accuracy=eval_candidate,
        eval_selected_delta_over_no_edit=eval_selected - eval_no_edit,
        eval_candidate_delta_over_no_edit=_delta(eval_candidate, eval_no_edit),
        selected_improved=_change_count(selected_eval, eval_family, "num_improved_over_baseline"),
        selected_harmed=_change_count(selected_eval, eval_family, "num_harmed_vs_baseline"),
        candidate_improved=_change_count(candidate_eval, eval_family, "num_improved_over_baseline"),
        candidate_harmed=_change_count(candidate_eval, eval_family, "num_harmed_vs_baseline"),
    )


def _mean(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _best_row(rows: list[SeedRow], key: Callable[[SeedRow], float | None]) -> SeedRow | None:
    candidates = [(key(row), row) for row in rows]
    candidates = [(score, row) for score, row in candidates if score is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _selection_criteria() -> list[tuple[str, Callable[[SeedRow], float | None]]]:
    return [
        ("absolute selected validation metric", lambda row: row.val_selected_metric),
        ("selected validation delta over no-edit", lambda row: row.val_selected_delta_over_no_edit),
        ("candidate validation delta over no-edit", lambda row: row.val_candidate_delta_over_no_edit),
    ]


def _selection_summaries(rows: list[SeedRow]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for name, key in _selection_criteria():
        row = _best_row(rows, key)
        if row is None:
            summaries.append({"criterion": name, "seed": None})
            continue
        summaries.append(
            {
                "criterion": name,
                "seed": row.seed,
                "mode": row.mode,
                "score": key(row),
                "eval_selected_delta_over_no_edit": row.eval_selected_delta_over_no_edit,
                "eval_candidate_delta_over_no_edit": row.eval_candidate_delta_over_no_edit,
            }
        )
    return summaries


def _print_rows(rows: list[SeedRow], *, eval_family: str) -> None:
    print(f"Seed sweep summary for eval family: {eval_family}")
    print(
        "seed mode pre_epoch sel_epoch reason margin nonzero "
        "local_patterns "
        "val_sel-noedit val_cand-noedit "
        "eval_sel-noedit eval_cand-noedit "
        "sel_imp/harm cand_imp/harm"
    )
    for row in rows:
        print(
            f"{row.seed} {row.mode} {row.best_epoch} "
            f"{row.selector_best_epoch} {row.adoption_reason} "
            f"{row.adoption_margin} {row.adoption_nonzero} "
            f"{row.candidate_local_motif_num_patterns} "
            f"{_fmt_float(row.val_selected_delta_over_no_edit)} "
            f"{_fmt_float(row.val_candidate_delta_over_no_edit)} "
            f"{_fmt_float(row.eval_selected_delta_over_no_edit)} "
            f"{_fmt_float(row.eval_candidate_delta_over_no_edit)} "
            f"{_fmt_count_pair(row.selected_improved, row.selected_harmed)} "
            f"{_fmt_count_pair(row.candidate_improved, row.candidate_harmed)}"
        )

    print()
    print(
        "mean selected eval delta over no-edit:",
        _fmt_float(_mean([row.eval_selected_delta_over_no_edit for row in rows])),
    )
    print(
        "mean candidate eval delta over no-edit:",
        _fmt_float(_mean([row.eval_candidate_delta_over_no_edit for row in rows])),
    )

    print()
    print("selection criteria:")
    for name, key in _selection_criteria():
        row = _best_row(rows, key)
        if row is None:
            print(f"- {name}: n/a")
            continue
        print(
            f"- {name}: seed {row.seed}, mode {row.mode}, "
            f"eval selected delta {_fmt_float(row.eval_selected_delta_over_no_edit)}, "
            f"eval candidate delta {_fmt_float(row.eval_candidate_delta_over_no_edit)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/eval/nn"),
        help="Root containing run directories.",
    )
    parser.add_argument(
        "--run-dir-template",
        required=True,
        help="Run directory template. Use {seed} where the seed number belongs.",
    )
    parser.add_argument("--seeds", default="0-3", help="Comma list and/or ranges, e.g. 0-3 or 0,1,2,3.")
    parser.add_argument("--eval-family", default="stage_c_corr")
    parser.add_argument(
        "--validation-families",
        default=None,
        help="Comma-separated validation families. Defaults to best_val_no_edit_by_family keys.",
    )
    parser.add_argument("--macro-f1-weight", type=float, default=1e-3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    rows: list[SeedRow] = []
    validation_families: list[str] | None = None

    for seed in seeds:
        run_dir = _run_dir_from_template(args.root, args.run_dir_template, seed)
        summary_path = run_dir / "experiment_summary.json"
        summary = _load_summary(summary_path)
        if validation_families is None:
            if args.validation_families:
                validation_families = [item.strip() for item in args.validation_families.split(",") if item.strip()]
            else:
                no_edit = summary["training"].get("best_val_no_edit_by_family")
                if not isinstance(no_edit, dict):
                    raise ValueError(
                        "Could not infer validation families because best_val_no_edit_by_family is missing. "
                        "Pass --validation-families for older artifacts."
                    )
                validation_families = sorted(no_edit)
        rows.append(
            _row_from_summary(
                seed=seed,
                run_dir=run_dir,
                summary=summary,
                validation_families=validation_families,
                eval_family=args.eval_family,
                macro_f1_weight=float(args.macro_f1_weight),
            )
        )

    _print_rows(rows, eval_family=args.eval_family)

    if args.json_out is not None:
        payload = {
            "schema_version": "predecoder_seed_sweep_compare.v1",
            "run_dir_template": args.run_dir_template,
            "seeds": seeds,
            "eval_family": args.eval_family,
            "validation_families": validation_families,
            "macro_f1_weight": float(args.macro_f1_weight),
            "mean_eval_selected_delta_over_no_edit": _mean(
                [row.eval_selected_delta_over_no_edit for row in rows]
            ),
            "mean_eval_candidate_delta_over_no_edit": _mean(
                [row.eval_candidate_delta_over_no_edit for row in rows]
            ),
            "selection_criteria": _selection_summaries(rows),
            "rows": [asdict(row) for row in rows],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
