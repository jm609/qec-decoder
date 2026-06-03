"""Build high-readability presentation schematic figures.

These figures are intended for PPT slides.  They are deliberately more
schematic than the thesis figures: large labels, limited detail, and 16:9
canvas size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("artifacts/figures/presentation")
W, H = 1600, 900

COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "grid": "#cbd5e1",
    "blue": "#2563eb",
    "blue_soft": "#dbeafe",
    "green": "#16a34a",
    "green_soft": "#dcfce7",
    "amber": "#f59e0b",
    "amber_soft": "#fef3c7",
    "red": "#dc2626",
    "red_soft": "#fee2e2",
    "purple": "#7c3aed",
    "purple_soft": "#ede9fe",
    "panel": "#f8fafc",
    "white": "#ffffff",
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    return img, ImageDraw.Draw(img)


def _text(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    size: int = 32,
    bold: bool = False,
    fill: str = COLORS["ink"],
    anchor: str = "la",
) -> None:
    d.text(xy, text, font=_font(size, bold), fill=fill, anchor=anchor)


def _center_text(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: Iterable[str],
    *,
    size: int = 30,
    bold: bool = False,
    fill: str = COLORS["ink"],
    gap: int = 42,
) -> None:
    lines = list(lines)
    total = (len(lines) - 1) * gap
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2 - total // 2
    for i, line in enumerate(lines):
        _text(d, (cx, cy + i * gap), line, size=size, bold=bold, fill=fill, anchor="mm")


def _rounded(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str = COLORS["grid"],
    width: int = 3,
    radius: int = 18,
) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _line(
    d: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    *,
    fill: str = COLORS["muted"],
    width: int = 5,
    arrow: bool = False,
) -> None:
    d.line([p1, p2], fill=fill, width=width)
    if not arrow:
        return
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 22
    base = (x2 - ux * size, y2 - uy * size)
    points = [
        (x2, y2),
        (base[0] + px * size * 0.55, base[1] + py * size * 0.55),
        (base[0] - px * size * 0.55, base[1] - py * size * 0.55),
    ]
    d.polygon(points, fill=fill)


def _detector_grid(
    d: ImageDraw.ImageDraw,
    *,
    x0: int,
    y0: int,
    cell: int = 76,
    active: set[tuple[int, int]] | None = None,
    highlight: set[tuple[int, int]] | None = None,
    title: str = "",
) -> None:
    active = active or set()
    highlight = highlight or set()
    if title:
        _text(d, (x0 + cell * 1.5, y0 - 42), title, size=28, bold=True, anchor="mm")
    for r in range(3):
        for c in range(3):
            cx = x0 + c * cell
            cy = y0 + r * cell
            d.ellipse(
                [cx - 19, cy - 19, cx + 19, cy + 19],
                fill=COLORS["blue"] if (r, c) in active else "#e2e8f0",
                outline=COLORS["ink"],
                width=3,
            )
            if (r, c) in highlight:
                d.ellipse(
                    [cx - 30, cy - 30, cx + 30, cy + 30],
                    outline=COLORS["red"],
                    width=6,
                )
    for r in range(3):
        for c in range(2):
            _line(d, (x0 + c * cell + 22, y0 + r * cell), (x0 + (c + 1) * cell - 22, y0 + r * cell), fill=COLORS["grid"], width=3)
    for r in range(2):
        for c in range(3):
            _line(d, (x0 + c * cell, y0 + r * cell + 22), (x0 + c * cell, y0 + (r + 1) * cell - 22), fill=COLORS["grid"], width=3)


def build_qec_pipeline() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Quantum error correction workflow", size=44, bold=True)
    _text(d, (70, 132), "A classical decoder interprets repeated syndrome measurements and updates the logical frame.", size=28, fill=COLORS["muted"])
    boxes = [
        (90, 300, 365, 500, COLORS["blue_soft"], ["physical", "qubits"]),
        (470, 300, 745, 500, COLORS["amber_soft"], ["syndrome", "measurement"]),
        (850, 300, 1125, 500, COLORS["purple_soft"], ["classical", "decoder"]),
        (1230, 300, 1505, 500, COLORS["green_soft"], ["logical", "correction"]),
    ]
    for x1, y1, x2, y2, fill, lines in boxes:
        _rounded(d, (x1, y1, x2, y2), fill=fill, outline=COLORS["grid"])
        _center_text(d, (x1, y1, x2, y2), lines, size=35, bold=True, gap=48)
    for x in [365, 745, 1125]:
        _line(d, (x + 20, 400), (x + 85, 400), arrow=True, width=7)
    _rounded(d, (245, 660, 1355, 760), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (245, 660, 1355, 760), ["This thesis modifies the decoder input, not the quantum circuit itself."], size=33, bold=True)
    img.save(OUT_DIR / "slide02_qec_pipeline.png")


def build_surface_code_syndrome() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Rotated surface code and syndrome measurement", size=44, bold=True)
    _text(d, (70, 132), "A d=3 schematic: stabilizer checks produce syndrome bits at each round.", size=28, fill=COLORS["muted"])
    x0, y0, step = 420, 260, 150
    for r in range(3):
        for c in range(3):
            x, y = x0 + c * step, y0 + r * step
            d.ellipse([x - 32, y - 32, x + 32, y + 32], fill="#fbbf24", outline=COLORS["ink"], width=4)
            _text(d, (x, y), "D", size=25, bold=True, anchor="mm")
    checks = [
        (x0 + 75, y0 + 75, COLORS["blue_soft"], COLORS["blue"], "X", "1"),
        (x0 + 225, y0 + 75, COLORS["green_soft"], COLORS["green"], "Z", "0"),
        (x0 + 75, y0 + 225, COLORS["green_soft"], COLORS["green"], "Z", "0"),
        (x0 + 225, y0 + 225, COLORS["blue_soft"], COLORS["blue"], "X", "1"),
    ]
    for x, y, fill, outline, label, bit in checks:
        _rounded(d, (x - 43, y - 43, x + 43, y + 43), fill=fill, outline=outline, width=4, radius=12)
        _text(d, (x, y - 8), label, size=27, bold=True, fill=outline, anchor="mm")
        _text(d, (x, y + 24), f"s={bit}", size=23, bold=True, fill=COLORS["ink"], anchor="mm")
    _rounded(d, (90, 270, 330, 580), fill=COLORS["panel"], outline=COLORS["grid"])
    _text(d, (210, 325), "Legend", size=31, bold=True, anchor="mm")
    d.ellipse([130, 380, 180, 430], fill="#fbbf24", outline=COLORS["ink"], width=3)
    _text(d, (220, 405), "data qubit", size=26, anchor="lm")
    _rounded(d, (130, 465, 180, 515), fill=COLORS["blue_soft"], outline=COLORS["blue"], width=3, radius=8)
    _text(d, (220, 490), "stabilizer check", size=26, anchor="lm")
    _rounded(d, (1010, 315, 1490, 535), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (1010, 315, 1490, 535), ["At each round t:", "syndrome s(t)", "is a vector of check outcomes."], size=31, bold=True, gap=44)
    img.save(OUT_DIR / "slide03_surface_code_syndrome.png")


def build_detection_event_xor() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Detection event from repeated syndrome measurements", size=42, bold=True)
    _text(d, (70, 128), "Schematic: detector fires when a stabilizer outcome changes between rounds.", size=28, fill=COLORS["muted"])

    _detector_grid(d, x0=200, y0=320, active={(0, 1), (2, 0)}, title="syndrome s(t-1)")
    _text(d, (430, 425), "XOR", size=42, bold=True, fill=COLORS["purple"], anchor="mm")
    _detector_grid(d, x0=570, y0=320, active={(0, 1), (1, 1), (2, 0)}, title="syndrome s(t)")
    _line(d, (790, 425), (940, 425), arrow=True, fill=COLORS["muted"], width=6)
    _detector_grid(d, x0=1030, y0=320, active={(1, 1)}, highlight={(1, 1)}, title="detection event D(t)")

    _rounded(d, (220, 675, 1380, 755), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (220, 675, 1380, 755), ["D_i(t) approx s_i(t) XOR s_i(t-1)"], size=34, bold=True)
    img.save(OUT_DIR / "slide04_detection_event_xor.png")


def build_pymatching_graph_construction() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "PyMatching: DetectorErrorModel to matching graph", size=42, bold=True)
    _text(d, (70, 128), "Detectors become graph nodes; graphlike error mechanisms become weighted edges.", size=28, fill=COLORS["muted"])

    boxes = [
        (80, 235, 400, 420, COLORS["blue_soft"], "Stim circuit", ["repeated", "measurements"]),
        (600, 235, 1000, 420, COLORS["amber_soft"], "DetectorErrorModel", ["detectors", "error mechanisms"]),
        (1120, 235, 1520, 420, COLORS["green_soft"], "Matching graph", ["nodes", "weighted edges"]),
    ]
    for box in boxes:
        x1, y1, x2, y2, fill, title, lines = box
        _rounded(d, (x1, y1, x2, y2), fill=fill, outline=COLORS["grid"])
        _text(d, ((x1 + x2) // 2, y1 + 70), title, size=34, bold=True, anchor="mm")
        _center_text(d, (x1, y1 + 90, x2, y2), lines, size=30, bold=True, fill=COLORS["muted"], gap=40)
    _line(d, (480, 328), (600, 328), arrow=True, width=6)
    _line(d, (1000, 328), (1120, 328), arrow=True, width=6)

    nodes = [(1195, 610), (1335, 560), (1450, 650), (1300, 735)]
    for a, b, weight in [(0, 1, "w=1.2"), (1, 2, "w=2.4"), (0, 3, "w=0.8"), (3, 2, "w=1.7")]:
        _line(d, nodes[a], nodes[b], fill=COLORS["muted"], width=5)
        mx = (nodes[a][0] + nodes[b][0]) // 2
        my = (nodes[a][1] + nodes[b][1]) // 2
        _text(d, (mx, my - 10), weight, size=22, fill=COLORS["muted"], anchor="mm")
    for idx, pt in enumerate(nodes):
        d.ellipse([pt[0] - 24, pt[1] - 24, pt[0] + 24, pt[1] + 24], fill=COLORS["white"], outline=COLORS["ink"], width=4)
        _text(d, pt, f"D{idx}", size=20, bold=True, anchor="mm")
    _rounded(d, (105, 610, 940, 750), fill=COLORS["panel"], outline=COLORS["grid"])
    _text(d, (140, 658), "One edge represents a possible error mechanism.", size=30, bold=True)
    _text(d, (140, 706), "One-detector errors are represented as boundary edges.", size=28, fill=COLORS["muted"])
    img.save(OUT_DIR / "slide05_pymatching_graph_construction.png")


def build_mwpm_two_paths() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "PyMatching: MWPM decoding objective", size=42, bold=True)
    _text(d, (70, 128), "Choose the lowest-weight error chain consistent with the observed detection events.", size=28, fill=COLORS["muted"])

    panel = (120, 210, 1480, 690)
    _rounded(d, panel, fill=COLORS["panel"], outline=COLORS["grid"])
    active = [(360, 390), (920, 390)]
    boundary_left = (200, 390)
    boundary_right = (1300, 390)
    for x in range(300, 1100, 140):
        d.ellipse([x - 16, 540 - 16, x + 16, 540 + 16], fill="#e2e8f0", outline=COLORS["grid"], width=2)
    for pt in active:
        d.ellipse([pt[0] - 32, pt[1] - 32, pt[0] + 32, pt[1] + 32], fill=COLORS["blue"], outline=COLORS["ink"], width=4)
    _text(d, active[0], "1", size=28, bold=True, fill="white", anchor="mm")
    _text(d, active[1], "1", size=28, bold=True, fill="white", anchor="mm")

    _line(d, active[0], active[1], fill=COLORS["green"], width=9)
    _text(d, (640, 345), "path A: weight 2.1", size=30, bold=True, fill=COLORS["green"], anchor="mm")
    _line(d, boundary_left, active[0], fill=COLORS["red"], width=7)
    _line(d, active[1], boundary_right, fill=COLORS["red"], width=7)
    _text(d, (750, 475), "path B: weight 3.8 + logical flip", size=30, bold=True, fill=COLORS["red"], anchor="mm")
    d.line([boundary_left[0], 250, boundary_left[0], 600], fill=COLORS["ink"], width=5)
    d.line([boundary_right[0], 250, boundary_right[0], 600], fill=COLORS["ink"], width=5)
    _text(d, (boundary_left[0], 625), "boundary", size=24, fill=COLORS["muted"], anchor="mm")
    _text(d, (boundary_right[0], 625), "logical boundary", size=24, fill=COLORS["muted"], anchor="mm")

    _rounded(d, (310, 735, 1290, 815), fill=COLORS["green_soft"], outline=COLORS["green"])
    _center_text(d, (310, 735, 1290, 815), ["MWPM selects path A because it has lower total weight."], size=31, bold=True)
    img.save(OUT_DIR / "slide06_mwpm_two_paths.png")


def build_problem_statement() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Problem statement", size=44, bold=True)
    _text(d, (70, 132), "Hard shots can make PyMatching choose the wrong logical frame.", size=29, fill=COLORS["muted"])
    _rounded(d, (110, 260, 470, 545), fill=COLORS["red_soft"], outline=COLORS["grid"])
    _center_text(d, (110, 260, 470, 545), ["noisy", "detection events"], size=35, bold=True, fill=COLORS["red"], gap=50)
    _line(d, (470, 402), (620, 402), arrow=True, width=7)
    _rounded(d, (620, 260, 980, 545), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (620, 260, 980, 545), ["ambiguous", "matching paths"], size=35, bold=True, gap=50)
    _line(d, (980, 402), (1130, 402), arrow=True, width=7)
    _rounded(d, (1130, 260, 1490, 545), fill=COLORS["red_soft"], outline=COLORS["red"])
    _center_text(d, (1130, 260, 1490, 545), ["wrong", "logical class"], size=35, bold=True, fill=COLORS["red"], gap=50)
    _rounded(d, (260, 660, 1340, 770), fill=COLORS["green_soft"], outline=COLORS["green"])
    _center_text(d, (260, 660, 1340, 770), ["Question: can a neural pre-decoder edit selected detection events before PyMatching?"], size=33, bold=True, fill=COLORS["green"])
    img.save(OUT_DIR / "slide07_problem_statement.png")


def build_input_representation_36ch() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Input representation: 36-channel space-time volume", size=42, bold=True)
    _text(d, (70, 132), "Each shot is represented as [36, T, H, W]; channel semantics stay fixed across distances.", size=27, fill=COLORS["muted"])
    x, y = 130, 290
    front = [(x, y + 90), (x + 260, y + 90), (x + 260, y + 310), (x, y + 310)]
    back = [(x + 90, y), (x + 350, y), (x + 350, y + 220), (x + 90, y + 220)]
    d.polygon(back, fill=COLORS["blue_soft"], outline=COLORS["ink"])
    d.polygon(front, fill="#eff6ff", outline=COLORS["ink"])
    for p, q in zip(front, back):
        d.line([p, q], fill=COLORS["ink"], width=3)
    _text(d, (300, y + 350), "3D detector volume", size=30, bold=True, anchor="mm")
    _text(d, (60, y + 50), "time T", size=24, bold=True, fill=COLORS["muted"])
    _text(d, (380, y + 250), "H x W", size=24, bold=True, fill=COLORS["muted"])
    groups = [
        ("event", COLORS["blue"]),
        ("geometry", COLORS["green"]),
        ("stage/context", COLORS["purple"]),
        ("noise summary", COLORS["amber"]),
    ]
    sx, sy = 585, 265
    for i, (label, color) in enumerate(groups):
        yy = sy + i * 88
        _rounded(d, (sx, yy, sx + 360, yy + 62), fill=COLORS["panel"], outline=color, width=4)
        _text(d, (sx + 180, yy + 33), label, size=29, bold=True, fill=color, anchor="mm")
    _text(d, (sx + 180, sy - 48), "36 channels", size=33, bold=True, anchor="mm")
    _line(d, (505, 445), (585, 445), arrow=True, width=6)
    _line(d, (945, 445), (1030, 445), arrow=True, width=6)
    _rounded(d, (1030, 230, 1510, 640), fill=COLORS["panel"], outline=COLORS["grid"])
    _text(d, (1270, 280), "Feature groups", size=32, bold=True, anchor="mm")
    rows = [
        ("Observed", "event"),
        ("Geometry", "valid mask, boundary, final round, X/Z check"),
        ("Context", "stage, distance, rounds, event fractions"),
        ("Noise", "p_cz, p_meas, correlated-error fractions"),
    ]
    yy = 338
    for name, desc in rows:
        _text(d, (1065, yy), name, size=24, bold=True, fill=COLORS["ink"])
        _text(d, (1065, yy + 34), desc, size=20, fill=COLORS["muted"])
        yy += 74
    _rounded(d, (260, 735, 1340, 820), fill=COLORS["amber_soft"], outline=COLORS["amber"])
    _center_text(d, (260, 735, 1340, 820), ["d3: [36, 4, 4, 4]     d5: [36, 6, 6, 6]"], size=34, bold=True)
    img.save(OUT_DIR / "slide09_input_representation_36ch.png")


def build_local_motif_candidates() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Local motif candidates", size=44, bold=True)
    _text(d, (70, 132), "The model ranks a small candidate set instead of freely editing many detectors.", size=28, fill=COLORS["muted"])
    _rounded(d, (90, 230, 630, 700), fill=COLORS["panel"], outline=COLORS["grid"])
    _detector_grid(d, x0=260, y0=380, active={(1, 1), (2, 1)}, highlight={(1, 1), (2, 1)}, title="d=3 local patch")
    _text(d, (360, 655), "candidate motif", size=30, bold=True, fill=COLORS["purple"], anchor="mm")
    _rounded(d, (760, 220, 1510, 710), fill=COLORS["panel"], outline=COLORS["grid"])
    _text(d, (1135, 275), "Candidate set for one shot", size=33, bold=True, anchor="mm")
    rows = [
        ("candidate 0", "no edit", COLORS["muted"]),
        ("candidate 1", "flip detector A", COLORS["blue"]),
        ("candidate 2", "flip detector B", COLORS["blue"]),
        ("candidate 3", "flip local motif A-B", COLORS["purple"]),
    ]
    yy = 350
    for left, right, color in rows:
        _rounded(d, (830, yy - 32, 1440, yy + 32), fill="white", outline=COLORS["grid"], radius=10)
        _text(d, (870, yy), left, size=25, bold=True, fill=COLORS["ink"], anchor="lm")
        _text(d, (1120, yy), right, size=25, bold=True, fill=color, anchor="lm")
        yy += 82
    _rounded(d, (260, 770, 1340, 835), fill=COLORS["green_soft"], outline=COLORS["green"])
    _center_text(d, (260, 770, 1340, 835), ["The identity/no-edit candidate is always included as a safety path."], size=30, bold=True, fill=COLORS["green"])
    img.save(OUT_DIR / "slide11_local_motif_candidates.png")


def build_training_flow() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Training flow", size=44, bold=True)
    _text(d, (70, 132), "The neural model is trained around PyMatching outcomes, not as a direct logical-class classifier.", size=28, fill=COLORS["muted"])
    boxes = [
        (80, 260, 335, 480, COLORS["blue_soft"], ["sample", "detector events"]),
        (425, 260, 680, 480, COLORS["panel"], ["raw", "PyMatching"]),
        (770, 260, 1025, 480, COLORS["amber_soft"], ["local-edit", "targets"]),
        (1115, 260, 1510, 480, COLORS["green_soft"], ["train 3D trunk", "edit + needs-edit"]),
    ]
    for x1, y1, x2, y2, fill, lines in boxes:
        _rounded(d, (x1, y1, x2, y2), fill=fill, outline=COLORS["grid"])
        _center_text(d, (x1, y1, x2, y2), lines, size=31, bold=True, gap=44)
    for x in [335, 680, 1025]:
        _line(d, (x + 18, 370), (x + 90, 370), arrow=True, width=6)

    lower = [
        (245, 610, 575, 770, COLORS["purple_soft"], ["candidate set", "with no-edit"]),
        (635, 610, 965, 770, COLORS["amber_soft"], ["benefit/harm", "selector ranking"]),
        (1025, 610, 1355, 770, COLORS["green_soft"], ["validation", "adoption policy"]),
    ]
    for x1, y1, x2, y2, fill, lines in lower:
        _rounded(d, (x1, y1, x2, y2), fill=fill, outline=COLORS["grid"])
        _center_text(d, (x1, y1, x2, y2), lines, size=29, bold=True, gap=40)
    _line(d, (575, 690), (635, 690), arrow=True, width=6)
    _line(d, (965, 690), (1025, 690), arrow=True, width=6)
    _line(d, (1315, 480), (1315, 610), arrow=True, width=6)
    _text(d, (1285, 548), "pooled features", size=23, fill=COLORS["muted"], anchor="rm")
    img.save(OUT_DIR / "slide10_training_flow.png")


def build_neural_predecoder_detailed() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Neural pre-decoder structure", size=44, bold=True)
    _text(d, (70, 132), "The neural part scores local edit candidates; PyMatching remains the final decoder.", size=28, fill=COLORS["muted"])

    _rounded(d, (70, 300, 300, 470), fill=COLORS["blue_soft"], outline=COLORS["blue"], width=4)
    _center_text(d, (70, 300, 300, 470), ["36-channel", "volume"], size=29, bold=True, fill=COLORS["blue"], gap=40)
    _line(d, (300, 385), (390, 385), arrow=True, width=6)

    _rounded(d, (390, 230, 700, 540), fill=COLORS["panel"], outline=COLORS["grid"], width=4)
    _text(d, (545, 280), "SyndromeEditPreDecoder", size=25, bold=True, anchor="mm")
    _center_text(d, (420, 300, 670, 455), ["3D conv stem", "+", "3 residual blocks"], size=27, bold=True, gap=36)
    _text(d, (545, 505), "shared feature volume", size=22, bold=True, fill=COLORS["muted"], anchor="mm")

    # Output nodes
    outputs = [
        (775, 205, 1030, 275, COLORS["red_soft"], COLORS["red"], "edit_logits"),
        (775, 325, 1030, 395, COLORS["amber_soft"], COLORS["amber"], "needs_edit_logits"),
        (775, 445, 1030, 515, COLORS["green_soft"], COLORS["green"], "pooled_features"),
    ]
    for x1, y1, x2, y2, fill, outline, label in outputs:
        _rounded(d, (x1, y1, x2, y2), fill=fill, outline=outline, width=3)
        _text(d, ((x1 + x2) // 2, (y1 + y2) // 2), label, size=24, bold=True, fill=outline, anchor="mm")
    _line(d, (700, 320), (775, 240), arrow=True, width=5, fill=COLORS["red"])
    _line(d, (700, 385), (775, 360), arrow=True, width=5, fill=COLORS["amber"])
    _line(d, (700, 450), (775, 480), arrow=True, width=5, fill=COLORS["green"])

    _rounded(d, (1085, 230, 1345, 395), fill=COLORS["purple_soft"], outline=COLORS["purple"], width=4)
    _center_text(d, (1085, 230, 1345, 395), ["candidate", "feature builder"], size=27, bold=True, fill=COLORS["purple"], gap=38)
    _line(d, (1030, 240), (1085, 285), arrow=True, width=5, fill=COLORS["red"])
    _line(d, (1030, 360), (1085, 340), arrow=True, width=5, fill=COLORS["amber"])

    _rounded(d, (1085, 465, 1345, 535), fill=COLORS["green_soft"], outline=COLORS["green"], width=3)
    _text(d, (1215, 500), "shot_features", size=24, bold=True, fill=COLORS["green"], anchor="mm")
    _line(d, (1030, 480), (1085, 500), arrow=True, width=5, fill=COLORS["green"])

    _rounded(d, (1085, 100, 1345, 170), fill=COLORS["purple_soft"], outline=COLORS["purple"], width=3)
    _text(d, (1215, 135), "candidate_features", size=24, bold=True, fill=COLORS["purple"], anchor="mm")
    _line(d, (1215, 230), (1215, 170), arrow=True, width=5, fill=COLORS["purple"])

    _rounded(d, (1390, 235, 1530, 535), fill=COLORS["amber_soft"], outline=COLORS["grid"], width=4)
    _center_text(d, (1390, 235, 1530, 535), ["Candidate", "Edit", "Selector", "MLP"], size=23, bold=True, gap=33)
    _line(d, (1345, 135), (1390, 310), arrow=True, width=5, fill=COLORS["purple"])
    _line(d, (1345, 500), (1390, 455), arrow=True, width=5, fill=COLORS["green"])

    _rounded(d, (545, 675, 805, 770), fill=COLORS["panel"], outline=COLORS["grid"], width=3)
    _center_text(d, (545, 675, 805, 770), ["candidate scalar scores"], size=25, bold=True)
    _line(d, (1460, 535), (1460, 620), width=5, fill=COLORS["muted"])
    _line(d, (1460, 620), (805, 720), arrow=True, width=5, fill=COLORS["muted"])

    _rounded(d, (875, 650, 1135, 795), fill=COLORS["red_soft"], outline=COLORS["red"], width=4)
    _center_text(d, (875, 650, 1135, 795), ["selected-mode", "safety policy"], size=25, bold=True, fill=COLORS["red"], gap=34)
    _line(d, (805, 720), (875, 720), arrow=True, width=5)

    _rounded(d, (1210, 650, 1510, 795), fill=COLORS["green_soft"], outline=COLORS["green"], width=4)
    _center_text(d, (1210, 650, 1510, 795), ["edit or no-edit", "then PyMatching"], size=26, bold=True, fill=COLORS["green"], gap=36)
    _line(d, (1135, 720), (1210, 720), arrow=True, width=5)

    img.save(OUT_DIR / "slide10_neural_predecoder_detailed.png")


def build_d3_detector_flip_example() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Schematic d=3 example: one detector flip changes PyMatching output", size=40, bold=True)
    _text(d, (70, 128), "Use this as a conceptual example, not as a claim about one specific saved shot.", size=27, fill=COLORS["muted"])

    _rounded(d, (80, 205, 480, 695), fill=COLORS["red_soft"], outline=COLORS["grid"])
    _text(d, (280, 255), "raw detector events", size=30, bold=True, anchor="mm")
    _detector_grid(d, x0=185, y0=365, active={(0, 0), (1, 1), (2, 2)}, highlight={(1, 1)})
    _line(d, (185, 365), (337, 517), fill=COLORS["red"], width=8)
    _line(d, (337, 517), (337, 517), fill=COLORS["red"], width=8)
    _text(d, (280, 635), "PyMatching: X", size=31, bold=True, fill=COLORS["red"], anchor="mm")
    _text(d, (280, 675), "true: I  -> wrong", size=27, bold=True, fill=COLORS["red"], anchor="mm")

    _rounded(d, (600, 305, 1000, 590), fill=COLORS["amber_soft"], outline=COLORS["grid"])
    _text(d, (800, 365), "NN candidate edit", size=32, bold=True, anchor="mm")
    d.ellipse([760, 425, 840, 505], fill=COLORS["blue"], outline=COLORS["ink"], width=4)
    _text(d, (800, 465), "D_k", size=26, bold=True, fill="white", anchor="mm")
    _text(d, (800, 545), "flip: 1 -> 0", size=34, bold=True, fill=COLORS["purple"], anchor="mm")
    _line(d, (480, 450), (600, 450), arrow=True, width=6)
    _line(d, (1000, 450), (1120, 450), arrow=True, width=6)

    _rounded(d, (1120, 205, 1520, 695), fill=COLORS["green_soft"], outline=COLORS["grid"])
    _text(d, (1320, 255), "edited detector events", size=30, bold=True, anchor="mm")
    _detector_grid(d, x0=1225, y0=365, active={(0, 0), (2, 2)})
    _line(d, (1225, 365), (1377, 517), fill=COLORS["green"], width=8)
    _text(d, (1320, 635), "PyMatching: I", size=31, bold=True, fill=COLORS["green"], anchor="mm")
    _text(d, (1320, 675), "true: I  -> correct", size=27, bold=True, fill=COLORS["green"], anchor="mm")

    _rounded(d, (285, 750, 1315, 825), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (285, 750, 1315, 825), ["The pre-decoder changes PyMatching's input, not its algorithm."], size=31, bold=True)
    img.save(OUT_DIR / "slide12_d3_detector_flip_example.png")


def build_transition_aware_selector() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Transition-aware selector objective", size=44, bold=True)
    _text(d, (70, 132), "Candidates are judged by the downstream PyMatching transition, not only by detector reconstruction.", size=27, fill=COLORS["muted"])
    _rounded(d, (105, 250, 470, 520), fill=COLORS["red_soft"], outline=COLORS["grid"])
    _center_text(d, (105, 250, 470, 520), ["raw D", "PyMatching", "wrong"], size=32, bold=True, fill=COLORS["red"], gap=44)
    _line(d, (470, 385), (620, 385), arrow=True, width=7)
    _rounded(d, (620, 250, 980, 520), fill=COLORS["amber_soft"], outline=COLORS["grid"])
    _center_text(d, (620, 250, 980, 520), ["candidate edit", "D' = D xor e"], size=32, bold=True, gap=44)
    _line(d, (980, 385), (1130, 385), arrow=True, width=7)
    _rounded(d, (1130, 250, 1495, 520), fill=COLORS["green_soft"], outline=COLORS["green"])
    _center_text(d, (1130, 250, 1495, 520), ["edited D'", "PyMatching", "correct"], size=32, bold=True, fill=COLORS["green"], gap=44)
    _rounded(d, (210, 640, 1390, 790), fill=COLORS["panel"], outline=COLORS["grid"])
    columns = [
        ("raw wrong -> edited correct", "beneficial", COLORS["green"]),
        ("raw correct -> edited wrong", "harmful", COLORS["red"]),
        ("no useful change", "neutral / miss", COLORS["muted"]),
    ]
    x = 300
    for condition, label, color in columns:
        _text(d, (x, 690), condition, size=24, bold=True, anchor="mm")
        _text(d, (x, 740), label, size=30, bold=True, fill=color, anchor="mm")
        x += 500
    img.save(OUT_DIR / "slide13_transition_aware_selector.png")


def build_selected_mode_safety() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Selected-mode safety policy", size=42, bold=True)
    _text(d, (70, 128), "The candidate branch is adopted only when validation evidence is strong enough.", size=28, fill=COLORS["muted"])

    _rounded(d, (90, 310, 400, 465), fill=COLORS["blue_soft"], outline=COLORS["grid"])
    _center_text(d, (90, 310, 400, 465), ["candidate", "selector"], size=32, bold=True)
    _line(d, (400, 388), (560, 388), arrow=True, width=6)

    _rounded(d, (560, 270, 1040, 505), fill=COLORS["amber_soft"], outline=COLORS["grid"])
    _center_text(d, (560, 270, 1040, 505), ["validation evidence", "passes harm/support", "guards?"], size=32, bold=True, gap=44)
    _line(d, (1040, 350), (1210, 260), arrow=True, width=6, fill=COLORS["green"])
    _line(d, (1040, 430), (1210, 600), arrow=True, width=6, fill=COLORS["red"])

    _rounded(d, (1210, 165, 1510, 335), fill=COLORS["green_soft"], outline=COLORS["green"])
    _center_text(d, (1210, 165, 1510, 335), ["apply", "selected edit"], size=31, bold=True, fill=COLORS["green"])
    _rounded(d, (1210, 535, 1510, 705), fill=COLORS["red_soft"], outline=COLORS["red"])
    _center_text(d, (1210, 535, 1510, 705), ["fallback to", "raw no-edit"], size=31, bold=True, fill=COLORS["red"])
    _text(d, (1110, 294), "yes", size=28, bold=True, fill=COLORS["green"], anchor="mm")
    _text(d, (1110, 522), "no", size=28, bold=True, fill=COLORS["red"], anchor="mm")

    _rounded(d, (300, 755, 1300, 825), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (300, 755, 1300, 825), ["Fallback is part of the proposed method because harmful edits are possible."], size=30, bold=True)
    img.save(OUT_DIR / "slide14_selected_mode_safety.png")


def build_noise_stage_design() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Noise-stage design", size=44, bold=True)
    _text(d, (70, 132), "Stage B/C are Willow-inspired synthetic noise families, not a faithful Willow hardware reproduction.", size=27, fill=COLORS["red"])
    stages = [
        ("Stage A", "stage_a_si1000", COLORS["blue_soft"], COLORS["blue"], ["uniform", "SI1000-Base"]),
        ("Stage B", "stage_b_local", COLORS["amber_soft"], COLORS["amber"], ["Stage A +", "local heterogeneity"]),
        ("Stage C", "stage_c_corr", COLORS["green_soft"], COLORS["green"], ["Stage B +", "correlated stray", "surrogate"]),
    ]
    xs = [95, 585, 1075]
    for i, (title, family, fill, outline, lines) in enumerate(stages):
        x1, y1, x2, y2 = xs[i], 255, xs[i] + 430, 570
        _rounded(d, (x1, y1, x2, y2), fill=fill, outline=outline, width=4)
        _text(d, ((x1 + x2) // 2, y1 + 48), title, size=35, bold=True, fill=outline, anchor="mm")
        _text(d, ((x1 + x2) // 2, y1 + 92), family, size=24, bold=True, fill=COLORS["muted"], anchor="mm")
        _center_text(d, (x1, y1 + 105, x2, y2), lines, size=31, bold=True, fill=COLORS["ink"], gap=42)
        if i < 2:
            _line(d, (x2 + 18, 415), (xs[i + 1] - 18, 415), arrow=True, width=6)
    _rounded(d, (150, 655, 720, 815), fill=COLORS["panel"], outline=COLORS["grid"])
    _center_text(d, (150, 655, 720, 815), ["Used for", "training / validation", "Stage A and Stage B"], size=28, bold=True, gap=38)
    _rounded(d, (880, 655, 1450, 815), fill=COLORS["green_soft"], outline=COLORS["green"])
    _center_text(d, (880, 655, 1450, 815), ["Held-out evaluation", "Stage C is kept for", "final performance claim"], size=28, bold=True, fill=COLORS["green"], gap=38)
    img.save(OUT_DIR / "slide15_noise_stage_design.png")


def build_noise_family_summary() -> None:
    img, d = _canvas()
    _text(d, (70, 60), "Noise-Family Behavior of Selected Pre-Decoding", size=54, bold=True)
    _text(
        d,
        (70, 128),
        "Mean selected-mode accuracy gain; Stage C is held out for the final claim.",
        size=34,
        fill=COLORS["muted"],
    )

    stages = [
        ("Stage A", "stage_a_si1000", "train/validation", 0.021103896, 0.003246753, "7/1/0", "1/6/1"),
        ("Stage B", "stage_b_local", "train/validation", 0.042207792, -0.0, "8/0/0", "1/6/1"),
        ("Stage C", "stage_c_corr", "held-out", 0.006591797, 0.005615234, "8/0/0", "2/6/0"),
    ]

    # Main chart. Keep all result labels outside the bars to prevent overlap
    # when the deltas are small.
    chart = (95, 180, 1505, 645)
    _rounded(d, chart, fill=COLORS["panel"], outline=COLORS["grid"], width=3, radius=16)
    _text(d, (140, 218), "Mean selected delta over raw PyMatching", size=34, bold=True)
    _text(d, (140, 263), "Higher is better", size=27, fill=COLORS["muted"])

    legend_y = 228
    d.rounded_rectangle([1055, legend_y - 18, 1091, legend_y + 18], radius=5, fill=COLORS["blue"])
    _text(d, (1105, legend_y), "d3", size=29, bold=True, anchor="lm")
    d.rounded_rectangle([1165, legend_y - 18, 1201, legend_y + 18], radius=5, fill=COLORS["purple"])
    _text(d, (1215, legend_y), "d5", size=29, bold=True, anchor="lm")
    _rounded(d, (1290, 203, 1438, 252), fill=COLORS["green_soft"], outline=COLORS["green"], width=3, radius=10)
    _text(d, (1364, 228), "held-out", size=24, bold=True, fill=COLORS["green"], anchor="mm")

    base_y = 515
    top_y = 300
    max_delta = 0.045
    usable_h = base_y - top_y

    # Y-axis reference lines.
    for value in [0.04, 0.02, 0.00]:
        y = int(base_y - (value / max_delta) * usable_h)
        d.line([185, y, 1435, y], fill=COLORS["grid"], width=2)
        label = "0" if value == 0 else f"+{value:.2f}"
        _text(d, (155, y), label, size=22, fill=COLORS["muted"], anchor="rm")

    group_centers = [405, 800, 1195]
    bar_w = 92
    gap = 34

    for idx, (stage_label, stage_name, role, d3_delta, d5_delta, _d3_seeds, _d5_seeds) in enumerate(stages):
        cx = group_centers[idx]
        if role == "held-out":
            _rounded(d, (cx - 178, 276, cx + 178, 558), fill=COLORS["green_soft"], outline=COLORS["green"], width=3, radius=14)
            # Redraw grid line across the highlighted held-out group.
            d.line([cx - 150, base_y, cx + 150, base_y], fill=COLORS["grid"], width=2)

        for j, (label, delta, color) in enumerate([("d3", d3_delta, COLORS["blue"]), ("d5", d5_delta, COLORS["purple"])]):
            x0 = cx - bar_w - gap // 2 if j == 0 else cx + gap // 2
            height = max(3, int(max(delta, 0.0) / max_delta * usable_h))
            y0 = base_y - height
            d.rounded_rectangle(
                [x0, y0, x0 + bar_w, base_y],
                radius=8,
                fill=color,
                outline=COLORS["ink"],
                width=2,
            )
            value_text = "0.0000" if abs(delta) < 0.00005 else f"{delta:+.4f}"
            _text(d, (x0 + bar_w // 2, y0 - 28), value_text, size=27, bold=True, fill=color, anchor="mm")
            _text(d, (x0 + bar_w // 2, base_y + 34), label, size=25, bold=True, anchor="mm")

        _text(d, (cx, 590), stage_label, size=33, bold=True, anchor="mm")
        _text(d, (cx, 626), stage_name, size=26, fill=COLORS["muted"], anchor="mm")

    # Seed-count table separated from the plot area.
    table = (180, 675, 1420, 875)
    _rounded(d, table, fill=COLORS["white"], outline=COLORS["grid"], width=3, radius=14)
    _text(d, (215, 707), "Seed counts: positive / neutral / harmful", size=31, bold=True)
    headers = [("Stage", 315, COLORS["ink"]), ("Role", 555, COLORS["muted"]), ("d3", 855, COLORS["blue"]), ("d5", 1135, COLORS["purple"])]
    for text, x, color in headers:
        _text(d, (x, 742), text, size=27, bold=True, fill=color, anchor="mm")

    d.line([205, 760, 1395, 760], fill=COLORS["grid"], width=2)
    row_ys = [790, 823, 856]
    for row_y, (stage_label, _stage_name, role, _d3_delta, _d5_delta, d3_seeds, d5_seeds) in zip(row_ys, stages):
        fill = COLORS["green"] if role == "held-out" else COLORS["ink"]
        _text(d, (315, row_y), stage_label, size=26, bold=True, fill=fill, anchor="mm")
        _text(d, (555, row_y), role, size=24, bold=True, fill=fill if role == "held-out" else COLORS["muted"], anchor="mm")
        _text(d, (855, row_y), d3_seeds, size=27, bold=True, fill=COLORS["blue"], anchor="mm")
        _text(d, (1135, row_y), d5_seeds, size=27, bold=True, fill=COLORS["purple"], anchor="mm")
    img.save(OUT_DIR / "slide15_noise_family_summary.png")
    img.save(OUT_DIR / "slide16_noise_family_summary.png")


def build_main_heldout_results() -> None:
    img, d = _canvas()
    _text(d, (70, 60), "Main Held-Out Results", size=56, bold=True)
    _text(
        d,
        (70, 128),
        "Stage C evaluation: selected-mode pre-decoding improves d3/d5, while d7 remains a calibration bottleneck.",
        size=31,
        fill=COLORS["muted"],
    )

    rows = [
        ("d3", 0.928710938, 0.935302734, 0.992187500, "+0.0066", "10.38%"),
        ("d5", 0.888671875, 0.894287109, 0.978515625, "+0.0056", "6.25%"),
        ("d7", 0.873046875, 0.873198411, 0.984375000, "+0.0002", "0.14%"),
    ]
    series = [
        ("raw", COLORS["grid"], COLORS["ink"]),
        ("selected", COLORS["blue"], COLORS["blue"]),
        ("oracle", COLORS["green"], COLORS["green"]),
    ]

    chart = (85, 185, 1125, 760)
    _rounded(d, chart, fill=COLORS["panel"], outline=COLORS["grid"], width=3, radius=16)
    _text(d, (125, 225), "Logical class-4 accuracy", size=34, bold=True)
    _text(d, (125, 270), "raw vs selected vs oracle headroom", size=27, fill=COLORS["muted"])

    legend_x = 560
    for i, (name, color, text_color) in enumerate(series):
        x = legend_x + i * 175
        d.rounded_rectangle([x, 215, x + 40, 255], radius=6, fill=color, outline=COLORS["ink"], width=2)
        _text(d, (x + 54, 235), name, size=27, bold=True, fill=text_color, anchor="lm")

    plot_left, plot_right = 160, 1060
    plot_top, plot_base = 320, 640
    y_min, y_max = 0.84, 1.00
    for value in [1.00, 0.96, 0.92, 0.88, 0.84]:
        y = int(plot_base - ((value - y_min) / (y_max - y_min)) * (plot_base - plot_top))
        d.line([plot_left, y, plot_right, y], fill=COLORS["grid"], width=2)
        _text(d, (plot_left - 18, y), f"{value:.2f}", size=24, fill=COLORS["muted"], anchor="rm")

    group_centers = [320, 610, 900]
    bar_w = 52
    offsets = [-68, 0, 68]

    for cx, (dist, raw, selected, oracle, delta, recovery) in zip(group_centers, rows):
        values = [raw, selected, oracle]
        for (name, color, _text_color), dx, value in zip(series, offsets, values):
            x0 = cx + dx - bar_w // 2
            y = int(plot_base - ((value - y_min) / (y_max - y_min)) * (plot_base - plot_top))
            d.rounded_rectangle(
                [x0, y, x0 + bar_w, plot_base],
                radius=7,
                fill=color,
                outline=COLORS["ink"],
                width=2,
            )
            label = f"{value:.3f}" if name != "selected" else f"{value:.3f}"
            _text(d, (x0 + bar_w // 2, y - 24), label, size=23, bold=True, fill=COLORS["ink"], anchor="mm")

        _text(d, (cx, plot_base + 45), dist, size=34, bold=True, anchor="mm")
        _rounded(d, (cx - 120, plot_base + 76, cx + 120, plot_base + 135), fill=COLORS["blue_soft"], outline=COLORS["blue"], width=3, radius=12)
        _text(d, (cx, plot_base + 96), f"delta {delta}", size=23, bold=True, fill=COLORS["blue"], anchor="mm")
        _text(d, (cx, plot_base + 122), f"oracle recovery {recovery}", size=20, fill=COLORS["muted"], anchor="mm")

    side = (1160, 205, 1505, 760)
    _rounded(d, side, fill=COLORS["white"], outline=COLORS["grid"], width=3, radius=16)
    _text(d, (1195, 250), "Interpretation", size=34, bold=True)

    notes = [
        (COLORS["blue_soft"], COLORS["blue"], "d3", "held-out gain"),
        (COLORS["purple_soft"], COLORS["purple"], "d5", "small positive gain"),
        (COLORS["red_soft"], COLORS["red"], "d7", "near raw"),
        (COLORS["green_soft"], COLORS["green"], "oracle", "headroom remains"),
    ]
    y = 305
    for fill, outline, title, body in notes:
        _rounded(d, (1195, y, 1470, y + 82), fill=fill, outline=outline, width=3, radius=12)
        _text(d, (1225, y + 24), title, size=27, bold=True, fill=outline)
        _text(d, (1225, y + 53), body, size=20, bold=True, fill=COLORS["ink"])
        y += 105

    _rounded(d, (1195, 705, 1470, 742), fill=COLORS["green_soft"], outline=COLORS["green"], width=2, radius=10)
    _text(d, (1333, 724), "Stage C held-out", size=24, bold=True, fill=COLORS["green"], anchor="mm")

    img.save(OUT_DIR / "slide18_main_heldout_results.png")


def build_d7_limitation_analysis() -> None:
    img, d = _canvas()
    _text(d, (70, 60), "D7 Limitation Analysis", size=56, bold=True)
    _text(
        d,
        (70, 128),
        "Oracle candidates exist, but selected-mode adoption recovers almost none of the available headroom.",
        size=32,
        fill=COLORS["muted"],
    )

    # Panel 1: held-out accuracy bars.
    panel1 = (75, 190, 520, 640)
    _rounded(d, panel1, fill=COLORS["panel"], outline=COLORS["grid"], width=3, radius=16)
    _text(d, (110, 232), "Held-out accuracy", size=33, bold=True)
    _text(d, (110, 270), "Stage C, d7", size=26, fill=COLORS["muted"])

    acc = [
        ("raw", 0.873046875, COLORS["grid"]),
        ("selected", 0.873198411, COLORS["blue"]),
        ("oracle", 0.984375000, COLORS["green"]),
    ]
    y_min, y_max = 0.84, 1.00
    base_y, top_y = 550, 310
    for value in [1.00, 0.92, 0.84]:
        y = int(base_y - ((value - y_min) / (y_max - y_min)) * (base_y - top_y))
        d.line([125, y, 480, y], fill=COLORS["grid"], width=2)
        _text(d, (115, y), f"{value:.2f}", size=22, fill=COLORS["muted"], anchor="rm")

    xs = [190, 305, 420]
    for x, (name, value, color) in zip(xs, acc):
        y = int(base_y - ((value - y_min) / (y_max - y_min)) * (base_y - top_y))
        d.rounded_rectangle([x - 33, y, x + 33, base_y], radius=7, fill=color, outline=COLORS["ink"], width=2)
        _text(d, (x, y - 23), f"{value:.3f}", size=24, bold=True, anchor="mm")
        _text(d, (x, base_y + 36), name, size=23, bold=True, fill=COLORS["blue"] if name == "selected" else COLORS["green"] if name == "oracle" else COLORS["ink"], anchor="mm")

    # Panel 2: oracle-gap recovery.
    panel2 = (555, 190, 1065, 640)
    _rounded(d, panel2, fill=COLORS["panel"], outline=COLORS["grid"], width=3, radius=16)
    _text(d, (590, 232), "Oracle-gap recovery", size=33, bold=True)
    _text(d, (590, 270), "available local-edit headroom used", size=24, fill=COLORS["muted"])

    axis_x0, axis_x1 = 620, 1015
    y_axis = 400
    d.line([axis_x0, y_axis, axis_x1, y_axis], fill=COLORS["grid"], width=4)
    for frac, label in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        x = int(axis_x0 + frac * (axis_x1 - axis_x0))
        d.line([x, y_axis - 14, x, y_axis + 14], fill=COLORS["muted"], width=3)
        _text(d, (x, y_axis + 42), label, size=21, fill=COLORS["muted"], anchor="mm")

    # Selected recovery is almost at zero; candidate-oracle shows candidate coverage.
    selected_recovery = 0.0014
    candidate_oracle = 0.8684
    sx = int(axis_x0 + selected_recovery * (axis_x1 - axis_x0))
    ox = int(axis_x0 + candidate_oracle * (axis_x1 - axis_x0))
    d.line([axis_x0, 335, ox, 335], fill=COLORS["green"], width=22)
    d.ellipse([ox - 18, 335 - 18, ox + 18, 335 + 18], fill=COLORS["green"], outline=COLORS["ink"], width=2)
    _text(d, (835, 300), "candidate-oracle: 86.84%", size=24, bold=True, fill=COLORS["green"], anchor="mm")
    d.ellipse([sx - 15, 465 - 15, sx + 15, 465 + 15], fill=COLORS["blue"], outline=COLORS["ink"], width=2)
    _text(d, (sx + 128, 465), "selected 0.14%", size=27, bold=True, fill=COLORS["blue"], anchor="lm")

    _rounded(d, (600, 515, 1020, 595), fill=COLORS["green_soft"], outline=COLORS["green"], width=3, radius=12)
    _text(d, (810, 542), "oracle headroom: 58/58 seeds", size=24, bold=True, fill=COLORS["green"], anchor="mm")
    _text(d, (810, 572), "candidate generation is not the bottleneck", size=20, fill=COLORS["ink"], anchor="mm")

    # Panel 3: validation-positive false positives.
    panel3 = (1100, 190, 1525, 640)
    _rounded(d, panel3, fill=COLORS["panel"], outline=COLORS["grid"], width=3, radius=16)
    _text(d, (1135, 232), "Held-out behavior", size=33, bold=True)
    _text(d, (1135, 270), "validation-positive candidate branches", size=23, fill=COLORS["muted"])

    total = 22
    segments = [
        ("positive", 5, COLORS["green"]),
        ("neutral", 4, COLORS["grid"]),
        ("harmful", 13, COLORS["red"]),
    ]
    bar_x0, bar_y0, bar_w, bar_h = 1145, 345, 330, 55
    x = bar_x0
    for label, count, color in segments:
        seg_w = int(bar_w * count / total)
        d.rectangle([x, bar_y0, x + seg_w, bar_y0 + bar_h], fill=color, outline=COLORS["ink"])
        _text(d, (x + seg_w // 2, bar_y0 + bar_h // 2), str(count), size=25, bold=True, fill=COLORS["white"] if color in {COLORS["green"], COLORS["red"]} else COLORS["ink"], anchor="mm")
        x += seg_w
    _text(d, (bar_x0 + bar_w // 2, bar_y0 + 91), "5 positive   4 neutral   13 harmful", size=23, bold=True, anchor="mm")

    _rounded(d, (1145, 485, 1475, 575), fill=COLORS["red_soft"], outline=COLORS["red"], width=3, radius=12)
    _text(d, (1310, 518), "false-positive rate", size=24, bold=True, fill=COLORS["red"], anchor="mm")
    _text(d, (1310, 552), "13 / 22 = 59.09%", size=30, bold=True, fill=COLORS["red"], anchor="mm")

    # Bottom conclusion.
    _rounded(d, (150, 700, 1450, 835), fill=COLORS["amber_soft"], outline=COLORS["amber"], width=4, radius=18)
    _text(d, (800, 735), "Conclusion", size=32, bold=True, fill=COLORS["amber"], anchor="mm")
    _text(d, (800, 776), "d7 bottleneck: selector ranking / calibration / adoption", size=32, bold=True, fill=COLORS["ink"], anchor="mm")
    _text(d, (800, 814), "not candidate coverage; another threshold sweep alone is not enough", size=26, fill=COLORS["muted"], anchor="mm")

    img.save(OUT_DIR / "slide19_d7_limitation_analysis.png")


def build_conclusion_summary() -> None:
    img, d = _canvas()
    _text(d, (70, 76), "Conclusion", size=44, bold=True)
    _text(d, (70, 132), "Bounded claim: useful d3/d5 pre-decoding, controlled d7 limitation.", size=29, fill=COLORS["muted"])
    rows = [
        (COLORS["blue_soft"], COLORS["blue"], "Method", "PyMatching-compatible neural pre-decoder"),
        (COLORS["green_soft"], COLORS["green"], "Positive result", "selected-mode gain on held-out d3/d5"),
        (COLORS["amber_soft"], COLORS["amber"], "Noise analysis", "Stage C claim after A/B validation"),
        (COLORS["red_soft"], COLORS["red"], "Limitation", "d7 selector calibration / ranking bottleneck"),
    ]
    y = 235
    for fill, outline, title, body in rows:
        _rounded(d, (165, y, 1435, y + 105), fill=fill, outline=outline, width=4)
        _text(d, (245, y + 55), title, size=31, bold=True, fill=outline, anchor="lm")
        _text(d, (615, y + 55), body, size=31, bold=True, fill=COLORS["ink"], anchor="lm")
        y += 132
    img.save(OUT_DIR / "slide19_conclusion_summary.png")
    img.save(OUT_DIR / "slide20_conclusion_summary.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_qec_pipeline()
    build_surface_code_syndrome()
    build_detection_event_xor()
    build_pymatching_graph_construction()
    build_mwpm_two_paths()
    build_problem_statement()
    build_input_representation_36ch()
    build_local_motif_candidates()
    build_training_flow()
    build_neural_predecoder_detailed()
    build_d3_detector_flip_example()
    build_transition_aware_selector()
    build_selected_mode_safety()
    build_noise_stage_design()
    build_noise_family_summary()
    build_main_heldout_results()
    build_d7_limitation_analysis()
    build_conclusion_summary()
    print(f"wrote presentation figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
