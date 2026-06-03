"""Rasterize generated predecoder SVG figures to PNG without external tools.

The renderer intentionally supports the small SVG subset emitted by the local
figure builders: rectangles, lines, circles, polygons, text, and a limited
rotated text group used for axis labels. It avoids Overleaf depending on the
`svg` package or Inkscape during final compilation.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont


DEFAULT_FIGURE_DIR = Path("artifacts/figures/predecoder")
FIGURE_BASENAMES = [
    "fig1_predecoder_pipeline",
    "fig2_model_architecture",
    "fig3_main_accuracy_comparison",
    "fig4_d7_oracle_gap_false_positive",
    "fig5_d7_validation_heldout_scatter",
    "fig6_oracle_recovery_distribution",
]


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(str(value).replace("px", ""))
    except ValueError:
        return default


def _color(value: str | None, opacity: float = 1.0) -> tuple[int, int, int, int] | None:
    if value is None or value == "" or value == "none" or value == "transparent":
        return None
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError:
        return None
    if len(rgb) == 4:
        r, g, b, a = rgb
        return (r, g, b, int(a * opacity))
    r, g, b = rgb
    return (r, g, b, int(255 * opacity))


def _font(size: int, weight: int = 400) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if weight >= 700 else "DejaVuSans.ttf",
        "Arial Bold.ttf" if weight >= 700 else "Arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _points(value: str | None, scale: float) -> list[tuple[float, float]]:
    if not value:
        return []
    pts: list[tuple[float, float]] = []
    for item in value.strip().split():
        if "," not in item:
            continue
        x, y = item.split(",", 1)
        pts.append((_float(x) * scale, _float(y) * scale))
    return pts


def _parse_transform(value: str | None) -> dict[str, float]:
    result = {"tx": 0.0, "ty": 0.0, "rotate": 0.0}
    if not value:
        return result
    translate = re.search(r"translate\(([-0-9.]+)[ ,]+([-0-9.]+)\)", value)
    if translate:
        result["tx"] = float(translate.group(1))
        result["ty"] = float(translate.group(2))
    rotate = re.search(r"rotate\(([-0-9.]+)\)", value)
    if rotate:
        result["rotate"] = float(rotate.group(1))
    return result


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: tuple[int, int, int, int],
    scale: float,
) -> None:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 9.0 * scale
    base_x = x2 - ux * size
    base_y = y2 - uy * size
    pts = [
        (x2, y2),
        (base_x + px * size * 0.45, base_y + py * size * 0.45),
        (base_x - px * size * 0.45, base_y - py * size * 0.45),
    ]
    draw.polygon(pts, fill=color)


def _draw_text(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    attrs: dict[str, str],
    text: str,
    *,
    scale: float,
    tx: float = 0.0,
    ty: float = 0.0,
    rotate: float = 0.0,
) -> None:
    if not text:
        return
    size = int(round(_float(attrs.get("font-size"), 13) * scale))
    weight = int(_float(attrs.get("font-weight"), 400))
    font = _font(size, weight)
    fill = _color(attrs.get("fill", "#111827"), _float(attrs.get("opacity"), 1.0)) or (17, 24, 39, 255)
    x = (_float(attrs.get("x")) + tx) * scale
    y = (_float(attrs.get("y")) + ty) * scale
    anchor = attrs.get("text-anchor", "start")
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if anchor == "middle":
        x -= w / 2
    elif anchor == "end":
        x -= w
    y -= h * 0.82

    if rotate == 0.0:
        draw.text((x, y), text, font=font, fill=fill)
        return

    pad = int(8 * scale)
    text_img = Image.new("RGBA", (max(1, int(w + pad * 2)), max(1, int(h + pad * 2))), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((pad, pad), text, font=font, fill=fill)
    rotated = text_img.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (int(x), int(y)))


def _render_node(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    node: ET.Element,
    *,
    scale: float,
    tx: float = 0.0,
    ty: float = 0.0,
    rotate: float = 0.0,
) -> None:
    tag = _strip_ns(node.tag)
    attrs = {str(k): str(v) for k, v in node.attrib.items()}
    transform = _parse_transform(attrs.get("transform"))
    tx += transform["tx"]
    ty += transform["ty"]
    rotate += transform["rotate"]

    if tag == "g" or tag == "svg":
        for child in node:
            _render_node(image, draw, child, scale=scale, tx=tx, ty=ty, rotate=rotate)
        return
    if tag in {"defs", "marker", "path", "title"}:
        return

    opacity = _float(attrs.get("opacity"), 1.0)
    fill = _color(attrs.get("fill"), opacity)
    stroke = _color(attrs.get("stroke"), opacity)
    sw = max(1, int(round(_float(attrs.get("stroke-width"), 1.0) * scale)))

    if tag == "rect":
        x = (_float(attrs.get("x")) + tx) * scale
        y = (_float(attrs.get("y")) + ty) * scale
        w = _float(attrs.get("width")) * scale
        h = _float(attrs.get("height")) * scale
        radius = _float(attrs.get("rx"), 0.0) * scale
        box = [x, y, x + w, y + h]
        if radius > 0:
            draw.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke, width=sw)
        else:
            draw.rectangle(box, fill=fill, outline=stroke, width=sw)
        return

    if tag == "line":
        x1 = (_float(attrs.get("x1")) + tx) * scale
        y1 = (_float(attrs.get("y1")) + ty) * scale
        x2 = (_float(attrs.get("x2")) + tx) * scale
        y2 = (_float(attrs.get("y2")) + ty) * scale
        color = stroke or (51, 65, 85, 255)
        draw.line([x1, y1, x2, y2], fill=color, width=sw)
        if "marker-end" in attrs:
            _draw_arrowhead(draw, x1, y1, x2, y2, color=color, scale=scale)
        return

    if tag == "circle":
        cx = (_float(attrs.get("cx")) + tx) * scale
        cy = (_float(attrs.get("cy")) + ty) * scale
        r = _float(attrs.get("r")) * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=stroke, width=sw)
        return

    if tag == "polygon":
        pts = [((x / scale + tx) * scale, (y / scale + ty) * scale) for x, y in _points(attrs.get("points"), scale)]
        if pts:
            draw.polygon(pts, fill=fill, outline=stroke)
        return

    if tag == "text":
        text = "".join(node.itertext())
        _draw_text(image, draw, attrs, text, scale=scale, tx=tx, ty=ty, rotate=rotate)
        return


def rasterize_svg(svg_path: Path, png_path: Path, *, scale: float = 2.0) -> None:
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    width = int(_float(root.attrib.get("width")))
    height = int(_float(root.attrib.get("height")))
    image = Image.new("RGBA", (int(width * scale), int(height * scale)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for child in root:
        _render_node(image, draw, child, scale=scale)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(png_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    outputs: list[str] = []
    for basename in FIGURE_BASENAMES:
        svg_path = args.figure_dir / f"{basename}.svg"
        png_path = args.figure_dir / f"{basename}.png"
        if not svg_path.exists():
            raise FileNotFoundError(svg_path)
        rasterize_svg(svg_path, png_path, scale=args.scale)
        outputs.append(png_path.as_posix())
    print(json_like({"num_png": len(outputs), "outputs": outputs}))


def json_like(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
