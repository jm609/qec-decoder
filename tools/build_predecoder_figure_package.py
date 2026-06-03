"""Build thesis-ready SVG figures for the predecoder result package."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONSOLIDATED = Path("artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json")
D7_TAXONOMY = Path("artifacts/eval/nn/sedp_d7_harmful_edit_taxonomy_summary.json")

SUPPLEMENTAL_FIGURES = [
    (
        "fig5_d7_validation_heldout_scatter.svg",
        "D7 validation-vs-heldout scatter",
        "Scatter plot of d7 validation delta against held-out candidate delta.",
    ),
    (
        "fig6_oracle_recovery_distribution.svg",
        "Oracle recovery summary",
        "Distance-level summary of selected recovery and d7 candidate-oracle headroom.",
    ),
]


@dataclass(frozen=True, slots=True)
class Figure:
    filename: str
    title: str
    caption: str
    svg: str


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        stroke: str = "#334155",
        sw: float = 2.0,
        rx: float = 8,
    ) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#475569",
        sw: float = 2.5,
        marker: bool = False,
    ) -> None:
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}" stroke-linecap="round"{marker_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int = 22,
        fill: str = "#0f172a",
        weight: int = 400,
        anchor: str = "start",
    ) -> None:
        safe = html.escape(text)
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{safe}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: list[str],
        *,
        size: int = 22,
        fill: str = "#0f172a",
        weight: int = 400,
        anchor: str = "middle",
        gap: int = 30,
    ) -> None:
        for idx, line in enumerate(lines):
            self.text(x, y + idx * gap, line, size=size, fill=fill, weight=weight, anchor=anchor)

    def circle(self, cx: float, cy: float, r: float, *, fill: str, stroke: str = "#334155") -> None:
        self.items.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.00"/>'
        )

    def to_string(self) -> str:
        body = "\n  ".join(self.items)
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  {body}
</svg>
'''


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _paper_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("paper_result_table")
    if not isinstance(rows, list):
        raise ValueError("Missing paper_result_table")
    return [row for row in rows if isinstance(row, dict)]


def _figure_1_pipeline() -> Figure:
    svg = Svg(1200, 360)
    svg.text(50, 58, "Neural pre-decoder before PyMatching", size=34, weight=700)

    boxes = [
        (55, 130, 190, 95, "#dbeafe", ["36-channel", "volume"]),
        (300, 130, 190, 95, "#dcfce7", ["3D neural", "trunk"]),
        (545, 130, 190, 95, "#fef3c7", ["local edit", "candidate"]),
        (790, 130, 190, 95, "#ede9fe", ["select edit", "or no-edit"]),
        (1035, 130, 120, 95, "#e2e8f0", ["PyMatching"]),
    ]
    for x, y, w, h, fill, lines in boxes:
        svg.rect(x, y, w, h, fill=fill)
        svg.multiline(x + w / 2, y + 37, lines, size=23, weight=700, gap=30)
    for idx in range(len(boxes) - 1):
        x, y, w, h = boxes[idx][:4]
        nx, ny, _, nh = boxes[idx + 1][:4]
        svg.line(x + w + 18, y + h / 2, nx - 18, ny + nh / 2, marker=True)

    svg.rect(170, 280, 860, 48, fill="#f8fafc", stroke="#cbd5e1", rx=6)
    svg.text(600, 312, "The neural model edits the syndrome; PyMatching still decodes the logical frame.", size=22, weight=700, anchor="middle")
    return Figure(
        filename="fig1_predecoder_pipeline.svg",
        title="Overall predecoder pipeline",
        caption="Pipeline from syndrome/noise volume to selected edit/no-edit and final PyMatching decoding.",
        svg=svg.to_string(),
    )


def _figure_2_architecture() -> Figure:
    svg = Svg(1200, 500)
    svg.text(50, 58, "SyndromeEditPreDecoder", size=34, weight=700)

    panels = [
        (60, 125, 290, 255, "#ecfdf5", "3D trunk", ["conv stem", "residual blocks", "shot feature"]),
        (455, 125, 290, 255, "#f8fafc", "Candidate builder", ["edit logits", "local motifs", "no-edit candidate"]),
        (850, 125, 290, 255, "#fff7ed", "Patch-head selector", ["local patch", "benefit/harm", "candidate score"]),
    ]
    for x, y, w, h, fill, title, rows in panels:
        svg.rect(x, y, w, h, fill=fill, stroke="#cbd5e1")
        svg.text(x + w / 2, y + 48, title, size=28, weight=700, anchor="middle")
        for idx, row in enumerate(rows):
            ry = y + 90 + idx * 48
            svg.rect(x + 35, ry, w - 70, 34, fill="#ffffff", stroke="#94a3b8", rx=5)
            svg.text(x + w / 2, ry + 25, row, size=22, weight=700, anchor="middle")
    svg.line(350, 252, 455, 252, marker=True)
    svg.line(745, 252, 850, 252, marker=True)

    svg.rect(820, 418, 350, 48, fill="#fce7f3", stroke="#cbd5e1", rx=6)
    svg.text(995, 450, "edit / no-edit", size=26, weight=700, anchor="middle")
    svg.line(995, 380, 995, 418, marker=True)
    return Figure(
        filename="fig2_model_architecture.svg",
        title="Model architecture",
        caption="The 3D trunk, candidate builder, and patch-head selector used by the neural predecoder.",
        svg=svg.to_string(),
    )


def _bar(svg: Svg, x: float, y0: float, width: float, value: float, min_v: float, max_v: float, fill: str, plot_h: float) -> float:
    h = (value - min_v) / (max_v - min_v) * plot_h
    svg.rect(x, y0 - h, width, h, fill=fill, stroke="#334155", rx=2)
    return y0 - h


def _figure_3_results(data: dict[str, Any]) -> Figure:
    rows = _paper_rows(data)
    svg = Svg(1200, 560)
    svg.text(50, 58, "Held-out accuracy by distance", size=34, weight=700)

    min_v, max_v = 0.84, 1.00
    left, top, y0 = 105, 110, 430
    plot_w, plot_h = 760, 300
    svg.line(left, y0, left + plot_w, y0, stroke="#64748b")
    svg.line(left, top, left, y0, stroke="#64748b")
    for tick in [0.84, 0.88, 0.92, 0.96, 1.00]:
        ty = y0 - (tick - min_v) / (max_v - min_v) * plot_h
        svg.line(left - 5, ty, left + plot_w, ty, stroke="#e2e8f0", sw=1.5)
        svg.text(left - 16, ty + 7, f"{tick:.2f}", size=18, fill="#475569", anchor="end")

    colors = {"raw": "#94a3b8", "selected": "#0284c7", "oracle": "#16a34a"}
    group_w, bar_w = 235, 44
    for idx, row in enumerate(rows):
        gx = left + 75 + idx * group_w
        vals = [
            ("raw", float(row["raw_pymatching_accuracy"])),
            ("selected", float(row["selected_predecoder_accuracy"])),
            ("oracle", float(row["target_local_edit_oracle_accuracy"])),
        ]
        for j, (name, value) in enumerate(vals):
            bx = gx + j * 60
            ty = _bar(svg, bx, y0, bar_w, value, min_v, max_v, colors[name], plot_h)
            svg.text(bx + bar_w / 2, ty - 9, f"{value:.3f}", size=16, weight=700, anchor="middle")
        svg.text(gx + 82, y0 + 38, str(row["distance"]), size=24, weight=700, anchor="middle")
        svg.text(gx + 82, y0 + 68, f"{float(row['selected_delta_over_raw']):+.6f}", size=18, fill="#334155", anchor="middle")

    legend_x, legend_y = 910, 155
    svg.rect(legend_x, legend_y, 240, 150, fill="#ffffff", stroke="#cbd5e1", rx=6)
    for i, (key, label) in enumerate([("raw", "raw"), ("selected", "selected"), ("oracle", "oracle")]):
        y = legend_y + 35 + i * 38
        svg.rect(legend_x + 28, y - 18, 24, 24, fill=colors[key], stroke="#334155", rx=3)
        svg.text(legend_x + 68, y, label, size=20, weight=700)
    svg.rect(910, 340, 240, 80, fill="#fff7ed", stroke="#fed7aa", rx=6)
    svg.multiline(1030, 370, ["d3/d5 improve", "d7 stays near raw"], size=20, weight=700, gap=28)

    return Figure(
        filename="fig3_main_accuracy_comparison.svg",
        title="Main accuracy comparison",
        caption="Held-out accuracies for raw PyMatching, selected predecoder, and local-edit oracle.",
        svg=svg.to_string(),
    )


def _figure_4_d7(data: dict[str, Any], taxonomy: dict[str, Any]) -> Figure:
    d7 = data["d7_oracle_gap"]
    summary = taxonomy["summary_counts"]
    svg = Svg(1200, 430)
    svg.text(50, 58, "D7 bottleneck: headroom exists, ranking fails", size=34, weight=700)

    cards = [
        ("Oracle headroom", "58/58", "seeds positive", "#dcfce7"),
        ("Validation false positive", "13/22", "held-out harmful", "#fee2e2"),
        ("Safety fallback", "17/17", "harmful blocked", "#e0f2fe"),
    ]
    for idx, (title, number, desc, fill) in enumerate(cards):
        x = 65 + idx * 375
        svg.rect(x, 125, 320, 210, fill=fill, stroke="#cbd5e1")
        svg.text(x + 160, 168, title, size=23, weight=700, anchor="middle")
        svg.text(x + 160, 245, number, size=46, weight=700, anchor="middle")
        svg.text(x + 160, 292, desc, size=22, weight=700, anchor="middle")

    svg.rect(115, 365, 970, 42, fill="#f8fafc", stroke="#cbd5e1", rx=6)
    svg.text(
        600,
        393,
        (
            f"selected {float(d7['mean_selected_delta_over_no_edit']):+.6f}  |  "
            f"candidate {float(d7['mean_candidate_delta_over_no_edit']):+.6f}  |  "
            f"oracle {float(d7['mean_candidate_oracle_delta_over_no_edit']):+.6f}"
        ),
        size=20,
        weight=700,
        anchor="middle",
    )
    return Figure(
        filename="fig4_d7_oracle_gap_false_positive.svg",
        title="D7 oracle gap and false-positive bottleneck",
        caption="D7 retains oracle headroom but learned ranking does not generalize reliably.",
        svg=svg.to_string(),
    )


def _supplemental_figure(figure_dir: Path, filename: str, title: str, caption: str) -> Figure:
    path = figure_dir / filename
    if path.exists():
        return Figure(filename=filename, title=title, caption=caption, svg=path.read_text(encoding="utf-8"))

    svg = Svg(1000, 300)
    svg.text(50, 60, title, size=30, weight=700)
    svg.rect(70, 110, 860, 110, fill="#f8fafc", stroke="#cbd5e1")
    svg.text(500, 170, f"Missing source: {filename}", size=22, weight=700, anchor="middle")
    return Figure(filename=filename, title=title, caption=caption, svg=svg.to_string())


def build_figures(consolidated_path: Path, taxonomy_path: Path, figure_dir: Path) -> list[Figure]:
    consolidated = _load(consolidated_path)
    taxonomy = _load(taxonomy_path)
    figures = [
        _figure_1_pipeline(),
        _figure_2_architecture(),
        _figure_3_results(consolidated),
        _figure_4_d7(consolidated, taxonomy),
    ]
    for filename, title, caption in SUPPLEMENTAL_FIGURES:
        figures.append(_supplemental_figure(figure_dir, filename, title, caption))
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, default=CONSOLIDATED)
    parser.add_argument("--taxonomy", type=Path, default=D7_TAXONOMY)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/figures/predecoder"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = build_figures(args.consolidated, args.taxonomy, args.out_dir)
    summary = {
        "schema_version": "predecoder_figure_package.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "consolidated": args.consolidated.as_posix(),
            "d7_taxonomy": args.taxonomy.as_posix(),
        },
        "figures": [],
    }
    for fig in figures:
        path = args.out_dir / fig.filename
        path.write_text(fig.svg, encoding="utf-8")
        summary["figures"].append({"title": fig.title, "caption": fig.caption, "path": path.as_posix()})

    summary_path = args.out_dir / "predecoder_figure_package_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": args.out_dir.as_posix(), "summary": summary_path.as_posix(), "num_figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
