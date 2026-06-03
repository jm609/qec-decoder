"""Build seed-level oracle recovery summaries and an SVG figure."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "predecoder_oracle_recovery_distribution.v1"
DEFAULT_JSON_OUT = Path("artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json")
DEFAULT_SVG_OUT = Path("artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg")


@dataclass(frozen=True, slots=True)
class DistanceSpec:
    distance: str
    source_type: str
    source_path: Path
    manifest_path: Path


DEFAULT_SPECS = (
    DistanceSpec(
        distance="d3",
        source_type="compare",
        source_path=Path("artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json"),
        manifest_path=Path("artifacts/datasets/predecoder_targets_d3_2k_router1k/manifest.json"),
    ),
    DistanceSpec(
        distance="d5",
        source_type="compare",
        source_path=Path("artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_7.json"),
        manifest_path=Path("artifacts/datasets/predecoder_targets_d5_2k_router1k/manifest.json"),
    ),
    DistanceSpec(
        distance="d7",
        source_type="oracle",
        source_path=Path("artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json"),
        manifest_path=Path("artifacts/datasets/predecoder_targets_d7_2k_router1k/manifest.json"),
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
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


def _distribution(values: list[float]) -> dict[str, Any]:
    kept = sorted(float(value) for value in values)
    return {
        "mean": _mean(kept),
        "min": kept[0] if kept else None,
        "p25": _percentile(kept, 0.25),
        "median": _percentile(kept, 0.50),
        "p75": _percentile(kept, 0.75),
        "max": kept[-1] if kept else None,
        "positive_count": sum(1 for value in kept if value > 0.0),
        "neutral_count": sum(1 for value in kept if value == 0.0),
        "harmful_count": sum(1 for value in kept if value < 0.0),
    }


def _oracle_stats(manifest_path: Path, eval_family: str) -> dict[str, float]:
    data = _load_json(manifest_path)
    stats = (data.get("oracle_stats_by_family") or {}).get(eval_family) or {}
    baseline = float(stats["baseline_pymatching_logical_class4_accuracy"])
    oracle = float(stats["oracle_pymatching_logical_class4_accuracy_after_edit_targets"])
    return {
        "baseline_accuracy": baseline,
        "target_local_edit_oracle_accuracy": oracle,
        "target_local_edit_oracle_delta": float(oracle - baseline),
    }


def _safe_recovery(delta: float | None, oracle_delta: float) -> float | None:
    if delta is None or abs(oracle_delta) < 1e-12:
        return None
    return float(delta / oracle_delta)


def _rows_from_compare(data: dict[str, Any], oracle_delta: float) -> list[dict[str, Any]]:
    rows = []
    for row in data.get("rows") or []:
        selected_delta = float(row["eval_selected_delta_over_no_edit"])
        candidate_delta = float(row["eval_candidate_delta_over_no_edit"])
        rows.append(
            {
                "seed": int(row["seed"]),
                "selected_mode": str(row.get("mode")),
                "adoption_reason": row.get("adoption_reason"),
                "selected_delta_over_no_edit": selected_delta,
                "candidate_delta_over_no_edit": candidate_delta,
                "candidate_oracle_delta_over_no_edit": None,
                "selected_recovery_fraction": _safe_recovery(selected_delta, oracle_delta),
                "candidate_recovery_fraction": _safe_recovery(candidate_delta, oracle_delta),
                "candidate_oracle_recovery_fraction": None,
            }
        )
    return rows


def _rows_from_oracle(data: dict[str, Any], oracle_delta: float) -> list[dict[str, Any]]:
    rows = []
    for row in data.get("rows") or []:
        selected_delta = float(row["selected_delta_over_no_edit"])
        candidate_delta = float(row["candidate_delta_over_no_edit"])
        candidate_oracle_delta = float(row["candidate_oracle_delta_over_no_edit"])
        rows.append(
            {
                "seed": int(row["seed"]),
                "selected_mode": str(row.get("selected_mode")),
                "adoption_reason": row.get("adoption_reason"),
                "selected_delta_over_no_edit": selected_delta,
                "candidate_delta_over_no_edit": candidate_delta,
                "candidate_oracle_delta_over_no_edit": candidate_oracle_delta,
                "selected_recovery_fraction": _safe_recovery(selected_delta, oracle_delta),
                "candidate_recovery_fraction": _safe_recovery(candidate_delta, oracle_delta),
                "candidate_oracle_recovery_fraction": _safe_recovery(candidate_oracle_delta, oracle_delta),
            }
        )
    return rows


def _summarize_distance(spec: DistanceSpec, eval_family: str) -> dict[str, Any]:
    source = _load_json(spec.source_path)
    target = _oracle_stats(spec.manifest_path, eval_family)
    oracle_delta = target["target_local_edit_oracle_delta"]
    if spec.source_type == "compare":
        seed_rows = _rows_from_compare(source, oracle_delta)
    elif spec.source_type == "oracle":
        seed_rows = _rows_from_oracle(source, oracle_delta)
    else:
        raise ValueError(f"Unknown source_type {spec.source_type!r}")

    selected_recovery = [
        float(row["selected_recovery_fraction"])
        for row in seed_rows
        if row["selected_recovery_fraction"] is not None
    ]
    candidate_recovery = [
        float(row["candidate_recovery_fraction"])
        for row in seed_rows
        if row["candidate_recovery_fraction"] is not None
    ]
    candidate_oracle_recovery = [
        float(row["candidate_oracle_recovery_fraction"])
        for row in seed_rows
        if row["candidate_oracle_recovery_fraction"] is not None
    ]
    return {
        "distance": spec.distance,
        "source_type": spec.source_type,
        "source_artifact": spec.source_path.as_posix(),
        "target_manifest": spec.manifest_path.as_posix(),
        "eval_family": eval_family,
        "num_seeds": len(seed_rows),
        **target,
        "selected_recovery_fraction_distribution": _distribution(selected_recovery),
        "candidate_recovery_fraction_distribution": _distribution(candidate_recovery),
        "candidate_oracle_recovery_fraction_distribution": (
            None if not candidate_oracle_recovery else _distribution(candidate_oracle_recovery)
        ),
        "seed_rows": seed_rows,
    }


def build_summary(specs: tuple[DistanceSpec, ...], eval_family: str) -> dict[str, Any]:
    distance_results = [_summarize_distance(spec, eval_family) for spec in specs]
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_family": eval_family,
        "distance_results": distance_results,
        "interpretation": [
            "d3 recovers a positive but modest fraction of the target local-edit oracle gap across all checked seeds.",
            "d5 recovers oracle gap only in the adopted local-selector seeds; selected-mode fallback keeps the remaining seeds at zero recovery rather than negative recovery.",
            "d7 has high candidate-oracle recovery headroom, but selected-mode recovery remains near zero because the selector rarely ranks a safe useful edit highly enough.",
        ],
    }


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.{digits}f}%"


def _text(x: float, y: float, text: str, *, size: int = 20, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">'
        f"{html.escape(text)}</text>"
    )


def _build_svg(summary: dict[str, Any]) -> str:
    width, height = 1200, 480
    left, top = 130, 130
    bar_w, row_gap = 760, 78
    x_min, x_max = -0.20, 0.90

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * bar_w

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _text(50, 58, "Oracle-gap recovery summary", size=34, weight=700),
        f'<line x1="{sx(0):.1f}" y1="105" x2="{sx(0):.1f}" y2="370" stroke="#94a3b8" stroke-width="3"/>',
    ]

    for value, label in [(-0.20, "-20%"), (0.0, "0"), (0.30, "30%"), (0.60, "60%"), (0.90, "90%")]:
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" y1="385" x2="{x:.1f}" y2="397" stroke="#64748b" stroke-width="2"/>')
        parts.append(_text(x, 423, label, size=18, anchor="middle"))
    parts.append(_text(left + bar_w / 2, 458, "fraction of target local-edit oracle gap recovered", size=20, weight=700, anchor="middle"))

    colors = {"selected": "#2563eb", "candidate": "#f97316", "oracle": "#16a34a"}
    for idx, result in enumerate(summary["distance_results"]):
        y = top + idx * row_gap
        distance = result["distance"]
        selected = result["selected_recovery_fraction_distribution"]["mean"]
        candidate = result["candidate_recovery_fraction_distribution"]["mean"]
        selected_value = 0.0 if selected is None else float(selected)
        candidate_value = 0.0 if candidate is None else float(candidate)

        parts.append(_text(70, y + 9, distance, size=24, weight=700, anchor="end"))
        parts.append(f'<line x1="{sx(x_min):.1f}" y1="{y:.1f}" x2="{sx(x_max):.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<circle cx="{sx(selected_value):.1f}" cy="{y:.1f}" r="12" fill="{colors["selected"]}" opacity="0.95"/>')
        parts.append(_text(sx(selected_value) + 18, y + 7, _fmt_pct(selected), size=20, weight=700))
        parts.append(f'<rect x="{sx(min(0, candidate_value)):.1f}" y="{y + 22:.1f}" width="{abs(sx(candidate_value)-sx(0)):.1f}" height="16" fill="{colors["candidate"]}" opacity="0.85"/>')
        parts.append(_text(sx(candidate_value) + 18, y + 39, _fmt_pct(candidate), size=18))

        if distance == "d7":
            oracle_mean = (result["candidate_oracle_recovery_fraction_distribution"] or {}).get("mean")
            oracle_value = 0.0 if oracle_mean is None else float(oracle_mean)
            parts.append(f'<circle cx="{sx(oracle_value):.1f}" cy="{y - 25:.1f}" r="12" fill="{colors["oracle"]}" opacity="0.95"/>')
            parts.append(_text(sx(oracle_value) + 18, y - 18, f"candidate-oracle {_fmt_pct(oracle_mean)}", size=20, weight=700))

    legend_x, legend_y = 930, 150
    parts.append(f'<rect x="{legend_x}" y="{legend_y - 40}" width="220" height="140" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" rx="8"/>')
    legend = [("selected", "selected mean"), ("candidate", "candidate mean"), ("oracle", "d7 candidate-oracle")]
    for idx, (key, label) in enumerate(legend):
        y = legend_y + idx * 38
        if key == "candidate":
            parts.append(f'<rect x="{legend_x + 22}" y="{y - 12}" width="26" height="16" fill="{colors[key]}"/>')
        else:
            parts.append(f'<circle cx="{legend_x + 35}" cy="{y - 4}" r="9" fill="{colors[key]}"/>')
        parts.append(_text(legend_x + 60, y + 3, label, size=17))
    parts.append(f'<rect x="{legend_x}" y="330" width="220" height="70" fill="#fff7ed" stroke="#fed7aa" stroke-width="2" rx="8"/>')
    parts.append(_text(legend_x + 110, 360, "D7 headroom remains", size=19, weight=700, anchor="middle"))
    parts.append(_text(legend_x + 110, 388, "mostly unused", size=19, weight=700, anchor="middle"))

    return "\n".join(parts + ["</svg>\n"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-family", default="stage_c_corr")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG_OUT)
    args = parser.parse_args()

    summary = build_summary(DEFAULT_SPECS, args.eval_family)
    svg = _build_svg(summary)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.svg_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    args.svg_out.write_text(svg, encoding="utf-8")
    print(json.dumps({"json_out": args.json_out.as_posix(), "svg_out": args.svg_out.as_posix(), "schema_version": SCHEMA_VERSION}, indent=2))


if __name__ == "__main__":
    main()
