"""Build a d7 validation-vs-held-out scatter figure from existing summaries."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "predecoder_d7_validation_heldout_scatter.v1"
DEFAULT_SOURCE = Path("artifacts/eval/nn/sedp_d7_selector_bottleneck_targeted_summary.json")
DEFAULT_JSON_OUT = Path("artifacts/eval/nn/sedp_d7_validation_heldout_scatter_summary.json")
DEFAULT_SVG_OUT = Path("artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = sum((x - mx) ** 2 for x in xs)
    denom_y = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(denom_x * denom_y)
    return None if denom == 0.0 else float(numerator / denom)


def _class_for_delta(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "harmful"
    return "neutral"


def _count_classes(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(1 for value in values if value > 0.0),
        "neutral": sum(1 for value in values if value == 0.0),
        "harmful": sum(1 for value in values if value < 0.0),
    }


def _build_summary(source: Path) -> dict[str, Any]:
    data = _load_json(source)
    rows = list(data.get("rows") or [])
    if not rows:
        raise ValueError(f"No rows in {source}")

    xs = [float(row["validation_delta_over_no_edit"]) for row in rows]
    ys = [float(row["candidate_delta_over_no_edit"]) for row in rows]
    validation_positive = [row for row in rows if float(row["validation_delta_over_no_edit"]) > 0.0]
    validation_positive_harmful = [
        row for row in validation_positive if float(row["candidate_delta_over_no_edit"]) < 0.0
    ]
    validation_positive_true = [
        row for row in validation_positive if float(row["candidate_delta_over_no_edit"]) > 0.0
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "source_artifact": source.as_posix(),
        "num_rows": len(rows),
        "x_metric": "validation_delta_over_no_edit",
        "y_metric": "held_out_candidate_delta_over_no_edit",
        "validation_delta_range": [min(xs), max(xs)],
        "held_out_candidate_delta_range": [min(ys), max(ys)],
        "pearson_correlation": _pearson(xs, ys),
        "validation_delta_class_counts": _count_classes(xs),
        "held_out_candidate_delta_class_counts": _count_classes(ys),
        "validation_positive_count": len(validation_positive),
        "validation_positive_harmful_count": len(validation_positive_harmful),
        "validation_positive_neutral_count": sum(
            1 for row in validation_positive if float(row["candidate_delta_over_no_edit"]) == 0.0
        ),
        "validation_positive_true_positive_count": len(validation_positive_true),
        "validation_positive_false_positive_ratio": (
            None if not validation_positive else float(len(validation_positive_harmful) / len(validation_positive))
        ),
        "validation_positive_harmful_seeds": [int(row["seed"]) for row in validation_positive_harmful],
        "validation_positive_true_positive_seeds": [int(row["seed"]) for row in validation_positive_true],
        "rows": [
            {
                "seed": int(row["seed"]),
                "validation_delta_over_no_edit": float(row["validation_delta_over_no_edit"]),
                "held_out_candidate_delta_over_no_edit": float(row["candidate_delta_over_no_edit"]),
                "validation_delta_class": _class_for_delta(float(row["validation_delta_over_no_edit"])),
                "held_out_candidate_delta_class": _class_for_delta(float(row["candidate_delta_over_no_edit"])),
                "selected_mode": str(row.get("selected_mode")),
                "adoption_reason": str(row.get("adoption_reason")),
            }
            for row in rows
        ],
    }


def _text(x: float, y: float, text: str, *, size: int = 20, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">'
        f"{html.escape(text)}</text>"
    )


def _build_svg(summary: dict[str, Any]) -> str:
    rows = list(summary["rows"])
    width, height = 1200, 540
    left, top = 105, 92
    plot_w, plot_h = 745, 335
    x_min, x_max = -0.0004, 0.0105
    y_min, y_max = -0.0205, 0.0060

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = {"positive": "#15803d", "neutral": "#6b7280", "harmful": "#dc2626"}
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _text(50, 52, "D7 validation-to-held-out mismatch", size=34, weight=700),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>',
    ]

    for value in [0.0, 0.0025, 0.0050, 0.0075, 0.0100]:
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        parts.append(_text(x, top + plot_h + 30, f"{value:.4f}", size=16, anchor="middle"))
    for value in [-0.020, -0.015, -0.010, -0.005, 0.0, 0.005]:
        y = sy(value)
        stroke = "#94a3b8" if value == 0.0 else "#e2e8f0"
        sw = 3 if value == 0.0 else 1
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')
        parts.append(_text(left - 12, y + 6, f"{value:+.3f}", size=16, anchor="end"))

    parts.append(_text(left + plot_w / 2, height - 36, "validation delta", size=20, anchor="middle", weight=700))
    parts.append(_text(50, top - 22, "held-out candidate delta", size=20, weight=700))

    for row in rows:
        x_value = float(row["validation_delta_over_no_edit"])
        y_value = float(row["held_out_candidate_delta_over_no_edit"])
        x = sx(x_value)
        y = sy(y_value)
        if x_value == 0.0:
            x += ((int(row["seed"]) % 9) - 4) * 3.0
        heldout_class = str(row["held_out_candidate_delta_class"])
        validation_class = str(row["validation_delta_class"])
        stroke = "#111827" if validation_class == "positive" else "#ffffff"
        sw = 2.3 if validation_class == "positive" else 1.0
        r = 7.5 if validation_class == "positive" else 5.5
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{colors[heldout_class]}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="0.90"/>'
        )

    panel_x = 890
    parts.append(f'<rect x="{panel_x}" y="102" width="260" height="250" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" rx="8"/>')
    false_count = int(summary["validation_positive_harmful_count"])
    pos_count = int(summary["validation_positive_count"])
    true_count = int(summary["validation_positive_true_positive_count"])
    neutral_count = int(summary["validation_positive_neutral_count"])
    ratio = float(summary["validation_positive_false_positive_ratio"]) * 100.0
    parts.append(_text(panel_x + 130, 145, "validation-positive", size=22, weight=700, anchor="middle"))
    parts.append(_text(panel_x + 130, 205, f"{false_count}/{pos_count}", size=44, weight=700, anchor="middle"))
    parts.append(_text(panel_x + 130, 238, "held-out harmful", size=21, weight=700, anchor="middle"))
    parts.append(_text(panel_x + 130, 290, f"positive {true_count}   neutral {neutral_count}", size=18, anchor="middle"))
    parts.append(_text(panel_x + 130, 322, f"false-positive {ratio:.2f}%", size=18, anchor="middle"))

    legend_y = 390
    for idx, (label, color) in enumerate([("positive", colors["positive"]), ("neutral", colors["neutral"]), ("harmful", colors["harmful"])]):
        y = legend_y + idx * 32
        parts.append(f'<circle cx="{panel_x + 25}" cy="{y}" r="7" fill="{color}" stroke="#111827" stroke-width="1"/>')
        parts.append(_text(panel_x + 45, y + 7, label, size=18))
    parts.append(_text(panel_x + 130, 508, f"Pearson r = {float(summary['pearson_correlation']):.3f}", size=20, weight=700, anchor="middle"))

    return "\n".join(parts + ["</svg>\n"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG_OUT)
    args = parser.parse_args()

    summary = _build_summary(args.source)
    svg = _build_svg(summary)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.svg_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    args.svg_out.write_text(svg, encoding="utf-8")
    print(json.dumps({"json_out": args.json_out.as_posix(), "svg_out": args.svg_out.as_posix(), "schema_version": SCHEMA_VERSION, "num_rows": summary["num_rows"]}, indent=2))


if __name__ == "__main__":
    main()
