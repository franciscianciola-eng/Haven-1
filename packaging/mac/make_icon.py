"""Draws Haven's app icon (Haven.icns): Haven in its garden, next to a berry bush.

Needs Pillow: python -m pip install pillow, then python packaging/mac/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
HERE = Path(__file__).parent


def icon() -> Image.Image:
    scale = 4  # drawn big, then shrunk, for smooth edges
    s = SIZE * scale
    art = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    margin, radius = int(s * 0.1), int(s * 0.18)  # the rounded square macOS icons sit in
    box = (margin, margin, s - margin, s - margin)

    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (box[0], box[1] + s // 90, box[2], box[3] + s // 90), radius, fill=(0, 0, 0, 90)
    )
    art.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(s // 80)))

    garden = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    g = ImageDraw.Draw(garden)
    top, bottom = (109, 176, 99), (54, 125, 64)
    for y in range(box[1], box[3]):
        t = (y - box[1]) / (box[3] - box[1])
        g.line([(box[0], y), (box[2], y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius, fill=255)
    art.paste(garden, (0, 0), mask)

    d = ImageDraw.Draw(art)
    cell = (box[2] - box[0]) / 8
    x0, y0 = box[0], box[1]

    def at(cx, cy):
        return x0 + cx * cell, y0 + cy * cell

    # A berry bush, lower right, with three berries.
    bx, by = at(5.7, 5.5)
    d.rounded_rectangle((bx - cell, by - cell * 0.9, bx + cell, by + cell * 0.9), int(cell * 0.35), fill=(38, 100, 52))
    for i, (dx, dy) in enumerate(((-0.45, -0.2), (0.25, -0.35), (-0.05, 0.35))):
        r = cell * 0.26
        cx, cy = bx + dx * cell, by + dy * cell
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(224, 40, 58))
        hl = r * 0.35
        d.ellipse(
            (cx - r * 0.45 - hl / 2, cy - r * 0.45 - hl / 2, cx - r * 0.45 + hl / 2, cy - r * 0.45 + hl / 2),
            fill=(255, 190, 196),
        )

    # Its nest, upper left.
    nx, ny = at(1.6, 1.7)
    d.rounded_rectangle(
        (nx - cell * 0.8, ny - cell * 0.55, nx + cell * 0.8, ny + cell * 0.55), int(cell * 0.3), fill=(219, 189, 76)
    )

    # Haven: a pale round body with a dark outline, looking toward the berries.
    hx, hy = at(3.4, 3.7)
    r = cell * 1.55
    outline = int(cell * 0.2)
    d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=(27, 26, 24))
    d.ellipse((hx - r + outline, hy - r + outline, hx + r - outline, hy + r - outline), fill=(244, 239, 230))
    ex, ey, er = hx + r * 0.42, hy + r * 0.28, r * 0.2
    d.ellipse((ex - er, ey - er, ex + er, ey + er), fill=(27, 26, 24))
    d.ellipse(
        (ex - er * 0.2 - er * 0.3, ey - er * 0.45 - er * 0.3, ex - er * 0.2 + er * 0.3, ey - er * 0.45 + er * 0.3),
        fill=(255, 255, 255),
    )
    return art.resize((SIZE, SIZE), Image.LANCZOS)


if __name__ == "__main__":
    image = icon()
    image.save(HERE / "Haven.icns")
    print("wrote", HERE / "Haven.icns")
