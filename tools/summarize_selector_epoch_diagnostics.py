from __future__ import annotations

"""
Summarize selector epoch margin diagnostics from pre-decoder experiment runs.

The input runs must have been created with
--selector-epoch-diagnostic-margin-grid. The tool reads experiment summaries and
does not mutate them.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json


@dataclass(frozen=True, slots=True)
class EpochMarginRow:
    seed: int
    run_dir: str
    epoch: int
    margin: float
    is_selector_best_epoch: bool
    selector_metric: float | None
    mean_delta_over_no_edit: float | None
    selected_nonzero: int
    improved: int
    harmed: int
    selected_positive_target: int
    selected_zero_target: int
    selected_negative_target: int
    max_best_nonzero_gap: float | None
    max_selected_gap: float | None
    family_deltas: dict[str, float | None]
    family_nonzero: dict[str, int]


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _run_dir(root: Path, template: str, seed: int) -> Path:
    rendered = template.format(seed=seed)
    path = Path(rendered)
    if path.is_absolute():
        return path
    return root / path


def _max_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _quantile_max(section: dict[str, Any], key: str) -> float | None:
    quantiles = section.get(key)
    if not isinstance(quantiles, dict):
        return None
    value = quantiles.get("max")
    return None if value is None else float(value)


def _rows_from_summary(*, seed: int, run_dir: Path, summary: dict[str, Any]) -> list[EpochMarginRow]:
    selector_training = (summary.get("training") or {}).get("selector_training") or {}
    best_epoch = int(selector_training.get("best_epoch") or -1)
    rows: list[EpochMarginRow] = []
    for epoch_record in selector_training.get("epoch_history") or []:
        epoch = int(epoch_record.get("epoch") or 0)
        by_family = epoch_record.get("val_margin_diagnostics_by_family") or {}
        if not isinstance(by_family, dict):
            continue
        margins = sorted(
            {
                str(margin)
                for family_diag in by_family.values()
                for margin in ((family_diag.get("by_margin") or {}).keys())
            },
            key=lambda x: float(x),
        )
        for margin_text in margins:
            family_deltas: dict[str, float | None] = {}
            family_nonzero: dict[str, int] = {}
            selected_nonzero = 0
            improved = 0
            harmed = 0
            selected_positive_target = 0
            selected_zero_target = 0
            selected_negative_target = 0
            best_gap_maxes: list[float | None] = []
            selected_gap_maxes: list[float | None] = []
            for family, family_diag in by_family.items():
                margin_diag = (family_diag.get("by_margin") or {}).get(margin_text) or {}
                delta = margin_diag.get("delta_over_no_edit")
                family_deltas[str(family)] = None if delta is None else float(delta)
                nonzero = int(margin_diag.get("selected_nonzero") or 0)
                family_nonzero[str(family)] = nonzero
                selected_nonzero += nonzero
                improved += int(margin_diag.get("improved") or 0)
                harmed += int(margin_diag.get("harmed") or 0)
                selected_positive_target += int(margin_diag.get("selected_positive_target") or 0)
                selected_zero_target += int(margin_diag.get("selected_zero_target") or 0)
                selected_negative_target += int(margin_diag.get("selected_negative_target") or 0)
                best_gap_maxes.append(_quantile_max(margin_diag, "best_nonzero_gap_quantiles"))
                selected_gap_maxes.append(_quantile_max(margin_diag, "selected_gap_quantiles"))
            rows.append(
                EpochMarginRow(
                    seed=int(seed),
                    run_dir=run_dir.as_posix(),
                    epoch=epoch,
                    margin=float(margin_text),
                    is_selector_best_epoch=(epoch == best_epoch),
                    selector_metric=(
                        None
                        if epoch_record.get("val_selection_metric") is None
                        else float(epoch_record.get("val_selection_metric"))
                    ),
                    mean_delta_over_no_edit=_mean_present(list(family_deltas.values())),
                    selected_nonzero=int(selected_nonzero),
                    improved=int(improved),
                    harmed=int(harmed),
                    selected_positive_target=int(selected_positive_target),
                    selected_zero_target=int(selected_zero_target),
                    selected_negative_target=int(selected_negative_target),
                    max_best_nonzero_gap=_max_present(best_gap_maxes),
                    max_selected_gap=_max_present(selected_gap_maxes),
                    family_deltas=family_deltas,
                    family_nonzero=family_nonzero,
                )
            )
    return rows


def _best_row(rows: list[EpochMarginRow], *, margin_floor: float = 0.0) -> EpochMarginRow | None:
    candidates = [
        row
        for row in rows
        if row.margin >= float(margin_floor)
        and row.mean_delta_over_no_edit is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.mean_delta_over_no_edit or 0.0),
            int(row.selected_nonzero),
            int(row.improved) - int(row.harmed),
            float(row.max_best_nonzero_gap or -1e9),
        ),
    )


def _count_rows(
    rows: list[EpochMarginRow],
    *,
    margin_floor: float = 0.0,
    min_delta: float = 0.0,
    min_nonzero: int = 1,
) -> int:
    return sum(
        1
        for row in rows
        if row.margin >= float(margin_floor)
        and row.mean_delta_over_no_edit is not None
        and float(row.mean_delta_over_no_edit) >= float(min_delta)
        and int(row.selected_nonzero) >= int(min_nonzero)
    )


def _summary(rows: list[EpochMarginRow]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_rows": len(rows),
        "positive_nonzero_rows": _count_rows(rows, min_delta=1e-12, min_nonzero=1),
        "margin1_positive_nonzero_rows": _count_rows(
            rows,
            margin_floor=1.0,
            min_delta=1e-12,
            min_nonzero=1,
        ),
        "candidate_first_strong_rows": _count_rows(rows, min_delta=0.02, min_nonzero=1),
        "candidate_first_positive_margin_rows": _count_rows(
            rows,
            margin_floor=0.5,
            min_delta=0.005,
            min_nonzero=1,
        ),
        "candidate_first_tie_high_margin_rows": _count_rows(
            rows,
            margin_floor=1.0,
            min_delta=-1e-9,
            min_nonzero=6,
        ),
    }
    best_any = _best_row(rows)
    best_margin1 = _best_row(rows, margin_floor=1.0)
    out["best_any_margin"] = None if best_any is None else asdict(best_any)
    out["best_margin_at_least_1"] = None if best_margin1 is None else asdict(best_margin1)
    return out


def _print_summary(rows: list[EpochMarginRow], *, top_k: int) -> None:
    summary = _summary(rows)
    print(
        "rows={num_rows} positive_nonzero={positive_nonzero_rows} "
        "margin1_positive_nonzero={margin1_positive_nonzero_rows} "
        "strong={candidate_first_strong_rows} "
        "positive_margin={candidate_first_positive_margin_rows} "
        "tie_high_margin={candidate_first_tie_high_margin_rows}".format(**summary)
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.mean_delta_over_no_edit or -1e9),
            int(row.selected_nonzero),
            int(row.improved) - int(row.harmed),
        ),
        reverse=True,
    )
    print("top rows:")
    for row in ranked[: int(top_k)]:
        print(
            f"seed={row.seed} epoch={row.epoch} margin={row.margin:g} "
            f"delta={row.mean_delta_over_no_edit:+.9f} "
            f"nz={row.selected_nonzero} imp/harm={row.improved}/{row.harmed} "
            f"target +/-={row.selected_positive_target}/{row.selected_negative_target} "
            f"gapmax={row.max_best_nonzero_gap}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/eval/nn"))
    parser.add_argument("--run-dir-template", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    rows: list[EpochMarginRow] = []
    for seed in _parse_seeds(str(args.seeds)):
        run_dir = _run_dir(args.root, str(args.run_dir_template), seed)
        rows.extend(
            _rows_from_summary(
                seed=seed,
                run_dir=run_dir,
                summary=_load_json(run_dir / "experiment_summary.json"),
            )
        )

    _print_summary(rows, top_k=int(args.top_k))
    if args.json_out is not None:
        payload = {
            "schema_version": "predecoder_selector_epoch_diagnostic_summary.v1",
            "run_dir_template": str(args.run_dir_template),
            "seeds": _parse_seeds(str(args.seeds)),
            "summary": _summary(rows),
            "rows": [asdict(row) for row in rows],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
