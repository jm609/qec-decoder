"""Build matplotlib-based thesis figures for the predecoder project.

This builder is a v2 replacement path for the older hand-written SVG figures.
It regenerates the six thesis figures with matplotlib and writes both PNG and
PDF outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#111827",
        "text.color": "#111827",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DEFAULT_OUT_DIR = Path("artifacts/figures/predecoder_v2")
EVAL_DIR = Path("artifacts/eval/nn")
CONSOLIDATED_PATH = EVAL_DIR / "sedp_predecoder_consolidated_evidence_summary.json"
D7_TAXONOMY_PATH = EVAL_DIR / "sedp_d7_harmful_edit_taxonomy_summary.json"
D7_SCATTER_PATH = EVAL_DIR / "sedp_d7_validation_heldout_scatter_summary.json"
ORACLE_RECOVERY_PATH = EVAL_DIR / "sedp_oracle_recovery_distribution_summary.json"

COLORS = {
    "blue": "#dbeafe",
    "green": "#dcfce7",
    "amber": "#fef3c7",
    "violet": "#ede9fe",
    "pink": "#fce7f3",
    "slate": "#e2e8f0",
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#334155",
    "grid": "#cbd5e1",
    "panel": "#f8fafc",
    "white": "#ffffff",
    "soft": "#f9fafb",
    "border": "#d1d5db",
}

ACCENTS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "violet": "#7c3aed",
    "rose": "#e11d48",
    "slate": "#475569",
}

SERIES_COLORS = {
    "raw": "#64748b",
    "selected": "#2563eb",
    "candidate": "#f59e0b",
    "oracle": "#16a34a",
    "harmful": "#dc2626",
    "neutral": "#94a3b8",
    "positive": "#16a34a",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _distance_key(distance: str) -> int:
    return int(distance.lower().lstrip("d"))


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pp(value: float, *, signed: bool = True) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def _setup_axes(width: float, height: float) -> tuple[plt.Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    fill: str,
    edge: str = COLORS["line"],
    size: int = 13,
    weight: str = "bold",
    radius: float = 0.018,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=fill,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=COLORS["ink"],
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def _plain_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    weight: str = "normal",
    color: str = COLORS["ink"],
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.18,
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    color: str = COLORS["line"],
    lw: float = 1.8,
    zorder: int = 3,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _section(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    fill: str = COLORS["panel"],
) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=COLORS["grid"],
        facecolor=fill,
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.018,
        y + h - 0.035,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["muted"],
        zorder=1,
    )


def _modern_card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: str,
    accent: str,
    title_size: int = 12,
    body_size: int = 10,
) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=COLORS["border"],
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(card)
    ax.add_patch(
        Rectangle(
            (x + 0.010, y + h - 0.020),
            w - 0.020,
            0.010,
            linewidth=0,
            facecolor=accent,
            zorder=3,
        )
    )
    title_y = y + h - 0.055
    ax.text(
        x + 0.024,
        title_y,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    if body:
        ax.text(
            x + 0.024,
            title_y - 0.050,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            color=COLORS["muted"],
            linespacing=1.20,
            zorder=4,
        )


def _thin_arrow(
    ax: Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = "#94a3b8",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
    )


def _chip(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    w: float = 0.132,
    h: float = 0.050,
    accent: str = "#e5e7eb",
    size: int = 9,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=0.9,
        edgecolor=accent,
        facecolor="#ffffff",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=COLORS["ink"],
        zorder=4,
    )


def _center_card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: str = "",
    fill: str = "#ffffff",
    edge: str = COLORS["border"],
    accent: str | None = None,
    title_size: float = 10.5,
    body_size: float = 8.8,
    weight: str = "bold",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)
    if accent:
        ax.add_patch(
            Rectangle(
                (x + 0.012, y + h - 0.020),
                w - 0.024,
                0.009,
                linewidth=0,
                facecolor=accent,
                zorder=3,
            )
        )
    title_y = y + (0.70 * h if body else 0.50 * h)
    ax.text(
        x + w / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight=weight,
        color=COLORS["ink"],
        linespacing=1.15,
        zorder=4,
    )
    if body:
        ax.text(
            x + w / 2,
            y + 0.30 * h,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            linespacing=1.18,
            zorder=4,
        )


def _segmented_bar_all_counts(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    segments: list[tuple[str, int, str]],
    total: int,
) -> None:
    _plain_label(ax, x, y + h + 0.078, title, size=12, weight="bold", ha="left")
    cursor = x
    legend_x = x
    for label, count, color in segments:
        width = w * count / total if total else 0.0
        ax.add_patch(
            Rectangle(
                (cursor, y),
                width,
                h,
                linewidth=1.0,
                edgecolor="white",
                facecolor=color,
                zorder=2,
            )
        )
        text_color = "white" if color != SERIES_COLORS["neutral"] else COLORS["ink"]
        if width >= 0.075:
            _plain_label(
                ax,
                cursor + width / 2,
                y + h / 2,
                f"{count}",
                size=12,
                weight="bold",
                color=text_color,
            )
        else:
            mid = cursor + width / 2
            ax.plot([mid, mid], [y + h, y + h + 0.020], color=COLORS["muted"], linewidth=0.8)
            _plain_label(
                ax,
                mid,
                y + h + 0.035,
                f"{count}",
                size=10.2,
                weight="bold",
                color=COLORS["ink"],
            )
        cursor += width

        ax.add_patch(
            FancyBboxPatch(
                (legend_x, y - 0.060),
                0.020,
                0.020,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                linewidth=0,
                facecolor=color,
            )
        )
        _plain_label(
            ax,
            legend_x + 0.027,
            y - 0.050,
            f"{label} {count}/{total}",
            size=8.0,
            color=COLORS["muted"],
            ha="left",
        )
        legend_x += 0.130


def _save(fig: plt.Figure, out_dir: Path, basename: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{basename}.png"
    pdf = out_dir / f"{basename}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.08, dpi=220)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {"png": png.as_posix(), "pdf": pdf.as_posix()}


def build_figure_1(out_dir: Path) -> dict[str, Any]:
    fig, ax = _setup_axes(13.6, 4.8)
    _plain_label(
        ax,
        0.035,
        0.93,
        "Transition-aware neural pre-decoder pipeline",
        size=20,
        weight="bold",
        ha="left",
    )
    _plain_label(
        ax,
        0.035,
        0.870,
        "A neural front-end selects a small detector edit; PyMatching remains the final decoder.",
        size=11.2,
        color=COLORS["muted"],
        ha="left",
    )

    _center_card(
        ax,
        0.055,
        0.485,
        0.160,
        0.245,
        title="Detector volume",
        body="36 channels\nsyndrome + noise",
        fill="#eff6ff",
        edge="#bfdbfe",
        accent=ACCENTS["blue"],
        title_size=11.5,
        body_size=9.3,
    )

    predecoder = FancyBboxPatch(
        (0.285, 0.350),
        0.450,
        0.430,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.0,
        edgecolor=COLORS["border"],
        facecolor="#ffffff",
        zorder=1,
    )
    ax.add_patch(predecoder)
    _plain_label(
        ax,
        0.510,
        0.735,
        "Neural pre-decoder",
        size=12.6,
        weight="bold",
    )
    _plain_label(
        ax,
        0.510,
        0.695,
        "local edit proposal and conservative adoption",
        size=9.2,
        color=COLORS["muted"],
    )
    steps = [
        (0.315, "1", "3D trunk\nfeatures", ACCENTS["green"]),
        (0.415, "2", "local motifs\ncandidates", ACCENTS["amber"]),
        (0.515, "3", "benefit/harm\nselector", ACCENTS["violet"]),
        (0.615, "4", "safety gate\nedit or raw", ACCENTS["rose"]),
    ]
    for x, idx, label, accent in steps:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.505),
                0.083,
                0.135,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                linewidth=1.0,
                edgecolor="#e5e7eb",
                facecolor="#f8fafc",
                zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.011, 0.602),
                0.022,
                0.026,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                linewidth=0,
                facecolor=accent,
                zorder=3,
            )
        )
        _plain_label(ax, x + 0.022, 0.615, idx, size=7.8, weight="bold", color="white")
        _plain_label(ax, x + 0.0415, 0.555, label, size=8.2, color=COLORS["ink"])

    for x0, x1 in [(0.398, 0.415), (0.498, 0.515), (0.598, 0.615)]:
        _thin_arrow(ax, x0 + 0.006, 0.572, x1 - 0.006, 0.572, color="#9ca3af")

    _center_card(
        ax,
        0.805,
        0.485,
        0.145,
        0.245,
        title="PyMatching",
        body="final logical frame\nI / X / Z / Y",
        fill="#f8fafc",
        edge="#cbd5e1",
        accent=ACCENTS["slate"],
        title_size=11.5,
        body_size=9.3,
    )

    _thin_arrow(ax, 0.218, 0.608, 0.282, 0.608, color="#64748b")
    _thin_arrow(ax, 0.738, 0.608, 0.802, 0.608, color="#64748b")

    _center_card(
        ax,
        0.215,
        0.160,
        0.570,
        0.120,
        title="Selected mode is conservative",
        body="adopt only high-confidence edits; otherwise preserve raw detector events",
        fill="#f9fafb",
        edge="#d1d5db",
        accent=None,
        title_size=11.0,
        body_size=9.0,
    )
    _thin_arrow(ax, 0.660, 0.505, 0.625, 0.283, color="#cbd5e1")

    outputs = _save(fig, out_dir, "fig1_predecoder_pipeline_v2")
    return {
        "figure": "fig1",
        "title": "Transition-aware neural pre-decoder pipeline",
        "outputs": outputs,
    }


def _mini_box(ax: Axes, x: float, y: float, w: float, h: float, label: str, fill: str) -> None:
    _box(ax, x, y, w, h, label=label, fill=fill, size=10, radius=0.012)


def build_figure_2(out_dir: Path) -> dict[str, Any]:
    fig, ax = _setup_axes(13.6, 6.4)

    def card(
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        title: str,
        body: str = "",
        face: str = "#ffffff",
        edge: str = COLORS["border"],
        title_size: float = 9.5,
        body_size: float = 7.6,
        lw: float = 1.0,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                zorder=2,
            )
        )
        if body:
            ax.text(
                x + w / 2,
                y + 0.70 * h,
                title,
                ha="center",
                va="center",
                fontsize=title_size,
                fontweight="bold",
                color=COLORS["ink"],
                linespacing=1.08,
                zorder=4,
            )
            ax.text(
                x + w / 2,
                y + 0.26 * h,
                body,
                ha="center",
                va="center",
                fontsize=body_size,
                color=COLORS["muted"],
                linespacing=1.18,
                zorder=4,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_size,
                fontweight="bold",
                color=COLORS["ink"],
                linespacing=1.08,
                zorder=4,
            )

    def arrow(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: str = "#94a3b8",
        lw: float = 1.6,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=lw,
                color=color,
                shrinkA=3,
                shrinkB=3,
                zorder=3,
            )
        )

    _plain_label(
        ax,
        0.035,
        0.955,
        "Neural predecoder architecture",
        size=20,
        weight="bold",
        ha="left",
    )
    _plain_label(
        ax,
        0.035,
        0.900,
        "A shared 3D trunk feeds local edit proposals and candidate-level benefit/harm ranking.",
        size=11.2,
        color=COLORS["muted"],
        ha="left",
    )

    # Top row: keep the feature-extraction path visually separate from the selector.
    card(
        0.050,
        0.610,
        0.145,
        0.165,
        title="Input tensor",
        body="[36, L, L, T]\nsyndrome/noise",
        face="#eff6ff",
        edge="#bfdbfe",
        title_size=10.0,
        body_size=8.2,
    )

    trunk_panel = FancyBboxPatch(
        (0.245, 0.485),
        0.255,
        0.330,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.1,
        edgecolor="#bbf7d0",
        facecolor="#f0fdf4",
        zorder=1,
    )
    ax.add_patch(trunk_panel)
    _plain_label(ax, 0.373, 0.775, "SyndromeEditPreDecoder", size=10.4, weight="bold")
    _plain_label(ax, 0.373, 0.745, "shared 3D residual trunk", size=8.1, color=COLORS["muted"])
    for y0, label in [(0.675, "3D conv stem"), (0.605, "Residual block x3"), (0.535, "Shared feature volume")]:
        card(
            0.285,
            y0,
            0.165,
            0.046,
            title=label,
            face="#ffffff",
            edge="#86efac",
            title_size=7.8,
        )
    arrow(0.367, 0.675, 0.367, 0.655, color="#86efac", lw=1.2)
    arrow(0.367, 0.605, 0.367, 0.585, color="#86efac", lw=1.2)
    arrow(0.198, 0.692, 0.242, 0.692, color="#64748b", lw=1.8)

    heads = [
        (0.565, 0.692, "edit_logits", "local edit proposal", "#f0fdf4", "#86efac"),
        (0.565, 0.600, "needs_edit_logits", "shot edit need", "#fffbeb", "#fde68a"),
        (0.565, 0.508, "pooled_features", "shot embedding", "#faf5ff", "#ddd6fe"),
    ]
    for x0, y0, title, body, face, edge in heads:
        card(
            x0,
            y0,
            0.150,
            0.066,
            title=title,
            body=body,
            face=face,
            edge=edge,
            title_size=7.4,
            body_size=6.2,
        )
        arrow(0.503, 0.650, x0 - 0.008, y0 + 0.033, color="#cbd5e1", lw=1.5)

    card(
        0.795,
        0.650,
        0.145,
        0.115,
        title="Candidate set",
        body="identity + local motifs\nrestricted top-k",
        face="#fffbeb",
        edge="#fde68a",
        title_size=8.4,
        body_size=6.9,
    )
    arrow(0.716, 0.725, 0.792, 0.708, color="#94a3b8", lw=1.7)

    # Bottom row: selector has its own lane, so arrows do not pass over labels.
    selector_panel = FancyBboxPatch(
        (0.245, 0.170),
        0.500,
        0.230,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.1,
        edgecolor="#ddd6fe",
        facecolor="#faf5ff",
        zorder=1,
    )
    ax.add_patch(selector_panel)
    _plain_label(ax, 0.495, 0.358, "CandidateEditSelector", size=10.6, weight="bold")
    _plain_label(ax, 0.495, 0.324, "patch-head ranking module", size=8.3, color=COLORS["muted"])
    for x0, label in [(0.300, "pooled shot"), (0.425, "candidate"), (0.550, "local patch")]:
        _chip(ax, x0, 0.235, label, w=0.095, h=0.044, accent=ACCENTS["violet"], size=7.4)
    card(
        0.670,
        0.222,
        0.055,
        0.070,
        title="scalar\nscore",
        body="benefit/harm",
        face="#ffffff",
        edge="#ddd6fe",
        title_size=6.8,
        body_size=5.5,
    )
    arrow(0.395, 0.257, 0.420, 0.257, color="#a78bfa", lw=1.5)
    arrow(0.520, 0.257, 0.545, 0.257, color="#a78bfa", lw=1.5)
    arrow(0.646, 0.257, 0.667, 0.257, color="#a78bfa", lw=1.5)

    card(
        0.795,
        0.455,
        0.150,
        0.100,
        title="Selected output",
        body="edit or fallback\nthen PyMatching",
        face="#f8fafc",
        edge="#cbd5e1",
        title_size=8.2,
        body_size=6.6,
    )
    arrow(0.867, 0.650, 0.867, 0.557, color="#cbd5e1", lw=1.6)

    card(
        0.295,
        0.020,
        0.410,
        0.085,
        title="Training target",
        body="candidate labels use downstream PyMatching benefit/harm\nnot detector reconstruction alone",
        face="#f9fafb",
        edge="#d1d5db",
        title_size=8.0,
        body_size=6.2,
    )

    arrow(0.640, 0.508, 0.640, 0.405, color="#d8b4fe", lw=1.6)
    arrow(0.728, 0.257, 0.792, 0.485, color="#94a3b8", lw=1.6)

    outputs = _save(fig, out_dir, "fig2_model_architecture_v2")
    return {
        "figure": "fig2",
        "title": "Neural predecoder architecture",
        "outputs": outputs,
    }


def build_figure_3(out_dir: Path) -> dict[str, Any]:
    consolidated = _load_json(CONSOLIDATED_PATH)
    rows = sorted(consolidated["paper_result_table"], key=lambda row: _distance_key(row["distance"]))

    series = [
        ("Raw PyMatching", "raw_pymatching_accuracy", SERIES_COLORS["raw"]),
        ("Selected predecoder", "selected_predecoder_accuracy", SERIES_COLORS["selected"]),
        ("Candidate branch", "candidate_branch_accuracy", SERIES_COLORS["candidate"]),
        ("Target local-edit oracle", "target_local_edit_oracle_accuracy", SERIES_COLORS["oracle"]),
    ]
    distances = [row["distance"] for row in rows]
    x_positions = list(range(len(rows)))
    bar_w = 0.18
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)

    for offset, (label, key, color) in zip(offsets, series):
        values = [row[key] for row in rows]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_w,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            if label in {"Selected predecoder", "Target local-edit oracle"}:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.003,
                    _pct(value),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=COLORS["ink"],
                    rotation=0,
                )

    for idx, row in enumerate(rows):
        delta = row["selected_predecoder_accuracy"] - row["raw_pymatching_accuracy"]
        ax.text(
            idx - 0.5 * bar_w,
            row["selected_predecoder_accuracy"] + 0.014,
            _pp(delta),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=SERIES_COLORS["selected"],
        )

    ax.set_ylim(0.84, 1.01)
    ax.set_ylabel("Logical-frame accuracy", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(distances, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.985,
        "Held-out stage C accuracy by code distance",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.055,
        0.938,
        "Selected mode improves d3/d5 over raw PyMatching, while d7 stays near raw despite large oracle headroom.",
        ha="left",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.035, 0.04, 0.985, 0.88))

    outputs = _save(fig, out_dir, "fig3_main_accuracy_comparison_v2")
    return {
        "figure": "fig3",
        "title": "Held-out stage C accuracy by code distance",
        "outputs": outputs,
    }


def _stacked_bar(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    segments: list[tuple[str, int, str]],
    total: int,
) -> None:
    _plain_label(ax, x, y + h + 0.045, title, size=12, weight="bold", ha="left")
    cursor = x
    for label, count, color in segments:
        width = w * count / total if total else 0.0
        rect = FancyBboxPatch(
            (cursor, y),
            width,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=1.0,
            edgecolor="white",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(rect)
        if width > 0.07:
            _plain_label(
                ax,
                cursor + width / 2,
                y + h / 2,
                f"{count}",
                size=12,
                weight="bold",
                color="white" if color != SERIES_COLORS["neutral"] else COLORS["ink"],
            )
        cursor += width

    legend_x = x
    for label, count, color in segments:
        ax.add_patch(
            FancyBboxPatch(
                (legend_x, y - 0.055),
                0.020,
                0.022,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                linewidth=0,
                facecolor=color,
            )
        )
        _plain_label(
            ax,
            legend_x + 0.026,
            y - 0.044,
            f"{label} {count}/{total}",
            size=9.5,
            color=COLORS["muted"],
            ha="left",
        )
        legend_x += 0.145


def build_figure_4(out_dir: Path) -> dict[str, Any]:
    consolidated = _load_json(CONSOLIDATED_PATH)
    taxonomy = _load_json(D7_TAXONOMY_PATH)
    d7_gap = consolidated["d7_oracle_gap"]
    counts = taxonomy["summary_counts"]

    fig, ax = _setup_axes(13.6, 5.7)
    _plain_label(
        ax,
        0.035,
        0.955,
        "D7 limitation: candidate-oracle headroom, unreliable adoption",
        size=20,
        weight="bold",
        ha="left",
    )
    _plain_label(
        ax,
        0.035,
        0.895,
        "Useful edits exist, but validation-positive candidates often become harmful on held-out seeds.",
        size=11.2,
        color=COLORS["muted"],
        ha="left",
    )

    kpis = [
        ("Candidate-oracle", "58/58 seeds", _pp(d7_gap["mean_candidate_oracle_delta_over_no_edit"]), ACCENTS["green"]),
        ("Candidate branch", "held-out mean", _pp(d7_gap["mean_candidate_delta_over_no_edit"]), ACCENTS["amber"]),
        ("False positives", "validation-positive", "13/22", ACCENTS["rose"]),
        ("Selected mode", "final gain", _pp(d7_gap["mean_selected_delta_over_no_edit"]), ACCENTS["blue"]),
    ]
    for idx, (title, subtitle, value, accent) in enumerate(kpis):
        x0 = 0.060 + idx * 0.230
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.705),
                0.185,
                0.125,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=COLORS["border"],
                facecolor="#ffffff",
                zorder=2,
            )
        )
        ax.add_patch(Rectangle((x0, 0.705), 0.010, 0.125, linewidth=0, facecolor=accent, zorder=3))
        _plain_label(ax, x0 + 0.025, 0.793, title, size=9.7, weight="bold", ha="left")
        _plain_label(ax, x0 + 0.025, 0.764, subtitle, size=7.8, color=COLORS["muted"], ha="left")
        _plain_label(ax, x0 + 0.025, 0.727, value, size=12.2, weight="bold", color=accent, ha="left")

    left_panel = FancyBboxPatch(
        (0.055, 0.295),
        0.420,
        0.305,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.0,
        edgecolor=COLORS["border"],
        facecolor="#ffffff",
        zorder=0,
    )
    right_panel = FancyBboxPatch(
        (0.525, 0.295),
        0.420,
        0.305,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.0,
        edgecolor=COLORS["border"],
        facecolor="#ffffff",
        zorder=0,
    )
    ax.add_patch(left_panel)
    ax.add_patch(right_panel)

    _segmented_bar_all_counts(
        ax,
        0.095,
        0.430,
        0.340,
        0.065,
        title="Held-out candidate outcomes",
        segments=[
            ("pos.", counts["candidate_positive"], SERIES_COLORS["positive"]),
            ("neutral", counts["candidate_neutral"], SERIES_COLORS["neutral"]),
            ("harmful", counts["candidate_harmful"], SERIES_COLORS["harmful"]),
        ],
        total=counts["candidate_positive"] + counts["candidate_neutral"] + counts["candidate_harmful"],
    )
    _plain_label(
        ax,
        0.095,
        0.338,
        "Useful local edits exist; harmful outcomes remain frequent.",
        size=8.4,
        color=COLORS["muted"],
        ha="left",
    )

    _segmented_bar_all_counts(
        ax,
        0.565,
        0.430,
        0.340,
        0.065,
        title="Validation-positive held-out outcomes",
        segments=[
            ("TP", counts["validation_true_positive"], SERIES_COLORS["positive"]),
            ("neutral", counts["validation_positive_neutral"], SERIES_COLORS["neutral"]),
            ("harmful", counts["validation_false_positive_harmful"], SERIES_COLORS["harmful"]),
        ],
        total=counts["validation_positive"],
    )
    _plain_label(
        ax,
        0.565,
        0.338,
        "Validation positives often fail to transfer to held-out seeds.",
        size=8.4,
        color=COLORS["muted"],
        ha="left",
    )

    conclusion = FancyBboxPatch(
        (0.175, 0.120),
        0.650,
        0.110,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.0,
        edgecolor="#fecaca",
        facecolor="#fff1f2",
        zorder=0,
    )
    ax.add_patch(conclusion)
    _plain_label(
        ax,
        0.500,
        0.188,
        "Conclusion: d7 is limited by selector reliability, not by local-edit availability.",
        size=11.0,
        weight="bold",
    )
    _plain_label(
        ax,
        0.500,
        0.150,
        "The safety gate blocks harmful adoption, but selected-mode recovery remains near zero.",
        size=9.1,
        color=COLORS["muted"],
    )

    outputs = _save(fig, out_dir, "fig4_d7_oracle_gap_false_positive_v2")
    return {
        "figure": "fig4",
        "title": "D7 limitation: candidate-oracle headroom, unreliable adoption",
        "outputs": outputs,
    }


def build_figure_5(out_dir: Path) -> dict[str, Any]:
    scatter = _load_json(D7_SCATTER_PATH)
    rows = scatter["rows"]
    xs = [100.0 * row["validation_delta_over_no_edit"] for row in rows]
    ys = [100.0 * row["held_out_candidate_delta_over_no_edit"] for row in rows]

    fig, ax = plt.subplots(figsize=(12.4, 6.5), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.75)

    class_colors = {
        "positive": SERIES_COLORS["positive"],
        "neutral": SERIES_COLORS["neutral"],
        "harmful": SERIES_COLORS["harmful"],
    }
    for cls in ["harmful", "neutral", "positive"]:
        cls_rows = [row for row in rows if row["held_out_candidate_delta_class"] == cls]
        ax.scatter(
            [100.0 * row["validation_delta_over_no_edit"] for row in cls_rows],
            [100.0 * row["held_out_candidate_delta_over_no_edit"] for row in cls_rows],
            s=[
                86 if row["validation_delta_class"] == "positive" else 46
                for row in cls_rows
            ],
            c=class_colors[cls],
            edgecolors=[
                COLORS["ink"] if row["validation_delta_class"] == "positive" else "white"
                for row in cls_rows
            ],
            linewidths=[
                1.2 if row["validation_delta_class"] == "positive" else 0.5
                for row in cls_rows
            ],
            alpha=0.90,
            label=f"held-out {cls}",
        )

    ax.axhline(0.0, color=COLORS["line"], linewidth=1.2)
    ax.axvline(0.0, color=COLORS["line"], linewidth=1.2)
    pad_x = max(0.15, (max(xs) - min(xs)) * 0.12)
    pad_y = max(0.15, (max(ys) - min(ys)) * 0.12)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_xlabel("Validation gain over raw (percentage points)", fontsize=12)
    ax.set_ylabel("Held-out candidate gain (percentage points)", fontsize=12, labelpad=8)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    custom_handles = [
        Line2D([0], [0], marker="o", color="w", label="held-out harmful", markerfacecolor=SERIES_COLORS["harmful"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="held-out neutral", markerfacecolor=SERIES_COLORS["neutral"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="held-out positive", markerfacecolor=SERIES_COLORS["positive"], markersize=8),
        Line2D([0], [0], marker="o", color=COLORS["ink"], label="validation-positive seed", markerfacecolor="white", markersize=8, linewidth=0),
    ]
    ax.legend(handles=custom_handles, loc="lower left", frameon=True, framealpha=0.95, fontsize=9.5)

    stats = scatter
    ax.text(
        0.985,
        0.965,
        "validation-positive seeds\n"
        f"harmful: {stats['validation_positive_harmful_count']} / {stats['validation_positive_count']}\n"
        f"neutral: {stats['validation_positive_neutral_count']} / {stats['validation_positive_count']}\n"
        f"true positive: {stats['validation_positive_true_positive_count']} / {stats['validation_positive_count']}\n"
        f"Pearson r = {stats['pearson_correlation']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": COLORS["grid"],
            "linewidth": 1.0,
        },
    )
    fig.text(
        0.055,
        0.985,
        "D7 selector calibration: validation gain does not predict held-out gain",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.055,
        0.938,
        "Large outlined markers are validation-positive seeds; many land below zero on held-out evaluation.",
        ha="left",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.815, bottom=0.145)

    outputs = _save(fig, out_dir, "fig5_d7_validation_heldout_scatter_v2")
    return {
        "figure": "fig5",
        "title": "D7 selector calibration",
        "outputs": outputs,
    }


def build_figure_6(out_dir: Path) -> dict[str, Any]:
    recovery = _load_json(ORACLE_RECOVERY_PATH)
    rows = sorted(recovery["distance_results"], key=lambda row: _distance_key(row["distance"]))

    fig, ax = plt.subplots(figsize=(12.6, 6.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.75)

    y_positions = list(range(len(rows)))
    bar_h = 0.20
    plotted_values: list[float] = []
    for idx, row in enumerate(rows):
        selected = row["selected_recovery_fraction_distribution"]["mean"]
        candidate = row["candidate_recovery_fraction_distribution"]["mean"]
        plotted_values.extend([selected, candidate])
        ax.barh(
            idx + bar_h / 1.4,
            selected,
            height=bar_h,
            color=SERIES_COLORS["selected"],
            edgecolor="white",
            label="Selected mode" if idx == 0 else None,
        )
        ax.barh(
            idx - bar_h / 1.4,
            candidate,
            height=bar_h,
            color=SERIES_COLORS["candidate"],
            edgecolor="white",
            label="Learned candidate branch" if idx == 0 else None,
        )
        oracle_dist = row.get("candidate_oracle_recovery_fraction_distribution")
        if oracle_dist:
            oracle = oracle_dist["mean"]
            plotted_values.append(oracle)
            ax.barh(
                idx - bar_h * 2.1,
                oracle,
                height=bar_h,
                color=SERIES_COLORS["oracle"],
                edgecolor="white",
                label="Candidate-oracle upper bound",
            )
            ax.text(oracle + 0.015, idx - bar_h * 2.1, _pct(oracle), va="center", fontsize=10, color=COLORS["ink"])

        for value, y in [(selected, idx + bar_h / 1.4), (candidate, idx - bar_h / 1.4)]:
            offset = 0.012 if value >= 0 else -0.012
            ha = "left" if value >= 0 else "right"
            ax.text(value + offset, y, _pct(value), va="center", ha=ha, fontsize=10, color=COLORS["ink"])

    x_min = min(-0.08, min(plotted_values) - 0.04)
    x_max = max(1.0, max(plotted_values) + 0.08)
    ax.set_xlim(x_min, x_max)
    ax.axvline(0.0, color=COLORS["line"], linewidth=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["distance"] for row in rows], fontsize=12, fontweight="bold")
    ax.set_xlabel("Fraction of target local-edit oracle gap recovered", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)

    fig.text(
        0.055,
        0.985,
        "Oracle-gap recovery by distance",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.055,
        0.938,
        "d3/d5 recover a small but positive share of the oracle gap; d7 has candidate-oracle headroom but selected recovery is near zero.",
        ha="left",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.045, 0.055, 0.985, 0.885))

    outputs = _save(fig, out_dir, "fig6_oracle_recovery_distribution_v2")
    return {
        "figure": "fig6",
        "title": "Oracle-gap recovery by distance",
        "outputs": outputs,
    }


def build(out_dir: Path) -> dict[str, Any]:
    figures = [
        build_figure_1(out_dir),
        build_figure_2(out_dir),
        build_figure_3(out_dir),
        build_figure_4(out_dir),
        build_figure_5(out_dir),
        build_figure_6(out_dir),
    ]
    summary = {
        "schema_version": "predecoder_matplotlib_figures.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": out_dir.as_posix(),
        "figures": figures,
        "notes": [
            "Figure 1 and Figure 2 are generated with matplotlib patches.",
            "Figure 3 through Figure 6 are generated directly from fixed evaluation JSON artifacts.",
            "Outputs are isolated under predecoder_v2 and do not overwrite the original figure package.",
        ],
    }
    summary_path = out_dir / "predecoder_v2_figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = summary_path.as_posix()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = build(args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
