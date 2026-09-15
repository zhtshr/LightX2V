#!/usr/bin/env python3
"""Render throughput-latency schematic as PNG."""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[2] / "lightx2v/disagg/figures/throughput-latency-tradeoff.png"
OUT_ALONGSIDE_MD = Path(__file__).resolve().parents[2] / "lightx2v/disagg/throughput-latency-tradeoff.png"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

PLOT_W, PLOT_H = 480, 320
MARGIN = 18
PAD_AFTER_CROP = 12

METHODS = [
    {"label": "独立单卡", "x": 0.82, "y": 0.22, "color": "#666666"},
    {"label": "朴素并行", "x": 0.38, "y": 0.88, "color": "#3b82f6"},
    {"label": "重叠+混合并行", "x": 0.76, "y": 0.86, "color": "#3fa266"},
]


def vertical_text_image(text: str, font: ImageFont.FreeTypeFont, fill: str) -> Image.Image:
    probe = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(probe)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    layer = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=fill)
    return layer.rotate(90, expand=True)


def trim_whitespace(img: Image.Image, padding: int = PAD_AFTER_CROP) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    return img.crop((left, top, right, bottom))


def main() -> None:
    title_f = ImageFont.truetype(FONT_BOLD, 36)
    axis_f = ImageFont.truetype(FONT_BOLD, 30)
    tick_f = ImageFont.truetype(FONT, 26)
    legend_f = ImageFont.truetype(FONT, 28)
    hint_f = ImageFont.truetype(FONT, 20)

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_bbox = probe.textbbox((0, 0), "吞吐 vs 延迟（同 GPU 数，示意）", font=title_f)
    title_h = title_bbox[3] - title_bbox[1]

    y_label = vertical_text_image("延迟性能", axis_f, "#1a1a1a")
    y_tick_w = probe.textbbox((0, 0), "高", font=tick_f)[2]

    legend_gap = 14
    dot_r = 8
    legend_items = []
    for m in METHODS:
        tw = probe.textbbox((0, 0), m["label"], font=legend_f)[2]
        legend_items.append((m, tw))
    legend_w = sum(dot_r * 2 + legend_gap + tw for _, tw in legend_items) + legend_gap * (len(legend_items) - 1)
    legend_h = max(dot_r * 2, probe.textbbox((0, 0), "重叠+混合并行", font=legend_f)[3])

    plot_block_w = y_label.width + 12 + y_tick_w + 8 + PLOT_W
    content_w = max(plot_block_w, legend_w, title_bbox[2] - title_bbox[0])
    W = content_w + 2 * MARGIN

    title_top = MARGIN
    plot_y0 = title_top + title_h + 16
    plot_x0 = MARGIN + y_label.width + 12 + y_tick_w + 8
    plot_x1 = plot_x0 + PLOT_W
    plot_y1 = plot_y0 + PLOT_H

    x_axis_h = 28 + 34  # ticks + 「吞吐」
    legend_top = plot_y1 + x_axis_h + 20
    H = legend_top + legend_h + MARGIN

    def to_x(n: float) -> float:
        return plot_x0 + n * PLOT_W

    def to_y(n: float) -> float:
        return plot_y0 + (1 - n) * PLOT_H

    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    draw.text((W // 2, title_top), "吞吐 vs 延迟（同 GPU 数，示意）", fill="#1a1a1a", font=title_f, anchor="mt")

    x0, y0, x1, y1 = plot_x0, plot_y0, plot_x1, plot_y1
    draw.rectangle([x0, y0, x1, y1], fill="#f5f5f5", outline="#d0d0d0", width=2)
    draw.text((x1 - 6, y0 + 8), "理想 →", fill="#888888", font=hint_f, anchor="rt")

    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    draw.line([(mid_x, y0), (mid_x, y1)], fill="#e0e0e0", width=1)
    draw.line([(x0, mid_y), (x1, mid_y)], fill="#e0e0e0", width=1)
    draw.line([(x0, y1), (x1, y1)], fill="#1a1a1a", width=3)
    draw.line([(x0, y0), (x0, y1)], fill="#1a1a1a", width=3)

    tick_y = y1 + 24
    draw.text((x0, tick_y), "低", fill="#666666", font=tick_f, anchor="mt")
    draw.text((x1, tick_y), "高", fill="#666666", font=tick_f, anchor="mt")
    draw.text(((x0 + x1) / 2, y1 + 56), "吞吐", fill="#1a1a1a", font=axis_f, anchor="mt")

    draw.text((x0 - 10, y1), "低", fill="#666666", font=tick_f, anchor="rm")
    draw.text((x0 - 10, y0), "高", fill="#666666", font=tick_f, anchor="rm")

    ly = (y0 + y1 - y_label.height) // 2
    lx = plot_x0 - y_label.width - y_tick_w - 12
    img.paste(y_label, (lx, ly), y_label)

    r = 12
    for m in METHODS:
        cx, cy = to_x(m["x"]), to_y(m["y"])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=m["color"], outline="#ffffff", width=3)

    legend_x = (W - legend_w) // 2
    legend_cy = legend_top + legend_h // 2
    for m, tw in legend_items:
        draw.ellipse(
            [legend_x, legend_cy - dot_r, legend_x + dot_r * 2, legend_cy + dot_r],
            fill=m["color"],
        )
        draw.text(
            (legend_x + dot_r * 2 + legend_gap, legend_cy),
            m["label"],
            fill="#333333",
            font=legend_f,
            anchor="lm",
        )
        legend_x += dot_r * 2 + legend_gap + tw + legend_gap

    img = trim_whitespace(img)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    img.save(OUT_ALONGSIDE_MD, "PNG", optimize=True)
    print(f"{OUT} ({img.width}x{img.height})")
    print(f"{OUT_ALONGSIDE_MD} (copy for md preview)")


if __name__ == "__main__":
    main()
