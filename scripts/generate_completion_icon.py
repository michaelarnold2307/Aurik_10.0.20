"""
Generates a premium 128×128 px photorealistic restoration-completion icon.
Output: Aurik10/resources/completion_success.png

Design: Elegant golden trophy with subtle confetti particles and a small
vinyl disc badge — symbolises improved audio quality without being garish.
Matches the premium carrier-icon aesthetic (3D gradients, specular highlights).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent.parent / "Aurik10" / "resources"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 128
HALF = SIZE // 2


def _font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _radial_gradient(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r: int,
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
):
    """Draw a soft radial gradient from inner to outer color."""
    for i in range(r, 0, -1):
        t = 1.0 - (i / r)
        c = tuple(int(outer[j] + (inner[j] - outer[j]) * t) for j in range(4))
        draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=c)


def _draw_trophy(draw: ImageDraw.ImageDraw, img: Image.Image):
    """Draw an elegant golden trophy — compact, 3D, photorealistic feel."""

    # ── Base / Pedestal ──────────────────────────────────────────────────
    # Bottom plate (dark wood look)
    draw.rounded_rectangle([38, 105, 90, 114], radius=3, fill=(60, 42, 25, 255))
    # Lighter top bevel
    draw.rounded_rectangle([40, 104, 88, 108], radius=2, fill=(90, 65, 40, 255))
    # Metallic gold rim on pedestal
    draw.rounded_rectangle([42, 102, 86, 105], radius=2, fill=(210, 175, 80, 255))

    # ── Stem ─────────────────────────────────────────────────────────────
    # Main stem (gold gradient via layered rects)
    for i in range(18):
        t = i / 17.0
        gold = int(195 + 35 * t)
        g = int(155 + 30 * t)
        b = int(55 + 25 * t)
        a = 255
        x_off = int(2 * math.sin(t * math.pi))  # Slight waist curve
        draw.rectangle([59 - x_off, 84 + i, 69 + x_off, 85 + i], fill=(gold, g, b, a))

    # Stem specular highlight (vertical bright line)
    draw.line([(62, 85), (62, 101)], fill=(255, 240, 180, 100), width=1)

    # ── Cup body ─────────────────────────────────────────────────────────
    # Golden cup — use a polygon for the tapered shape
    # Cup outline (outer)
    cup_pts = [
        (40, 42),  # top-left rim
        (88, 42),  # top-right rim
        (82, 78),  # bottom-right
        (72, 86),  # meet stem right
        (56, 86),  # meet stem left
        (46, 78),  # bottom-left
    ]
    draw.polygon(cup_pts, fill=(215, 180, 70, 255))

    # Inner cup (darker inset)
    inner_pts = [
        (44, 46),
        (84, 46),
        (79, 76),
        (71, 83),
        (57, 83),
        (49, 76),
    ]
    draw.polygon(inner_pts, fill=(195, 160, 55, 255))

    # Cup specular highlight (left side bright reflection)
    high_pts = [
        (44, 46),
        (55, 46),
        (52, 76),
        (48, 76),
    ]
    draw.polygon(high_pts, fill=(255, 235, 140, 90))

    # Cup rim highlight (top bright edge)
    draw.line([(42, 43), (86, 43)], fill=(255, 245, 180, 180), width=2)

    # ── Handles ──────────────────────────────────────────────────────────
    # Left handle
    draw.arc([26, 46, 44, 72], start=180, end=360, fill=(200, 165, 60, 255), width=4)
    draw.arc([27, 47, 43, 71], start=180, end=360, fill=(255, 235, 140, 80), width=1)  # Highlight

    # Right handle
    draw.arc([84, 46, 102, 72], start=0, end=180, fill=(200, 165, 60, 255), width=4)
    draw.arc([85, 47, 101, 71], start=0, end=180, fill=(255, 235, 140, 80), width=1)  # Highlight

    # ── Small vinyl disc badge (bottom-right) ────────────────────────────
    vcx, vcy, vr = 100, 96, 16
    # Shadow
    draw.ellipse([vcx - vr - 1, vcy - vr + 1, vcx + vr + 1, vcy + vr + 3], fill=(0, 0, 0, 50))
    # Disc body (dark)
    draw.ellipse([vcx - vr, vcy - vr, vcx + vr, vcy + vr], fill=(25, 25, 30, 255))
    # Grooves
    for gr in range(5, vr - 1, 2):
        a = 35 + int(15 * math.sin(gr * 0.5))
        draw.ellipse([vcx - gr, vcy - gr, vcx + gr, vcy + gr], outline=(70, 70, 80, a), width=1)
    # Specular
    for i in range(4):
        a = max(0, 40 - i * 12)
        draw.ellipse(
            [vcx - vr + 3 + i, vcy - vr + 3 + i, vcx - vr + 10 - i, vcy - vr + 10 - i], fill=(255, 255, 255, a)
        )
    # Label
    draw.ellipse([vcx - 5, vcy - 5, vcx + 5, vcy + 5], fill=(180, 50, 50, 255))
    # Spindle
    draw.ellipse([vcx - 1, vcy - 1, vcx + 1, vcy + 1], fill=(50, 50, 50, 255))

    # ── Musical note sparkle on cup ──────────────────────────────────────
    font = _font(16)
    draw.text((57, 54), "♪", fill=(255, 245, 200, 160), font=font)


def _draw_confetti(draw: ImageDraw.ImageDraw):
    """Draw subtle, elegant confetti particles — not carnival, more like
    gentle celebration sparkles in muted gold/green/sage tones."""
    rng = random.Random(42)  # Deterministic for reproducibility

    # Confetti palette — muted, elegant tones (matching Aurik sage-green success)
    palette = [
        (130, 184, 154, 160),  # Sage green (SUCCESS_TEXT)
        (210, 185, 100, 140),  # Muted gold
        (180, 160, 120, 120),  # Warm tan
        (150, 170, 190, 110),  # Soft steel-blue
        (200, 160, 80, 130),  # Bronze
        (170, 140, 110, 100),  # Warm grey
    ]

    for _ in range(18):
        x = rng.randint(4, SIZE - 4)
        y = rng.randint(4, 40)  # Concentrate at top
        col = rng.choice(palette)
        shape = rng.choice(["rect", "line", "dot"])

        if shape == "rect":
            w = rng.randint(3, 7)
            h = rng.randint(2, 4)
            rng.randint(-30, 30)
            # Simple rotated rectangle approximation
            draw.rectangle([x, y, x + w, y + h], fill=col)
        elif shape == "line":
            length = rng.randint(4, 8)
            angle_rad = math.radians(rng.randint(-60, 60))
            x2 = x + int(length * math.cos(angle_rad))
            y2 = y + int(length * math.sin(angle_rad))
            draw.line([(x, y), (x2, y2)], fill=col, width=2)
        else:  # dot / sparkle
            r = rng.randint(1, 3)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    # A few scattered lower particles (falling effect)
    for _ in range(8):
        x = rng.randint(4, SIZE - 4)
        y = rng.randint(30, 95)
        col = rng.choice(palette)
        col = (*col[:3], col[3] // 2)  # More transparent
        r = rng.randint(1, 2)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col)


def _draw_sparkles(draw: ImageDraw.ImageDraw):
    """Draw small star-like sparkles around the trophy for a quality feel."""
    sparkle_positions = [
        (18, 28),
        (108, 32),
        (22, 70),
        (106, 68),
        (14, 50),
        (114, 52),
        (30, 14),
        (98, 16),
    ]
    for sx, sy in sparkle_positions:
        # 4-point star sparkle
        r = 3
        col = (255, 240, 180, 120)
        draw.line([(sx - r, sy), (sx + r, sy)], fill=col, width=1)
        draw.line([(sx, sy - r), (sx, sy + r)], fill=col, width=1)
        # Center bright dot
        draw.point((sx, sy), fill=(255, 255, 230, 180))


def _shadow(img: Image.Image) -> Image.Image:
    """Apply a soft drop shadow beneath the icon."""
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 45), (2, 3), mask=img.split()[3])
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas = Image.alpha_composite(canvas, img)
    return canvas


def generate():
    """Generate the completion icon."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Layer 1: Confetti (behind trophy)
    _draw_confetti(draw)

    # Layer 2: Trophy + disc badge
    _draw_trophy(draw, img)

    # Layer 3: Sparkles (on top)
    _draw_sparkles(draw)

    # Apply drop shadow
    img = _shadow(img)

    path = OUT / "completion_success.png"
    img.save(path, "PNG", optimize=True)
    print(f"  ✓ {path.name}  ({SIZE}×{SIZE})")
    return path


if __name__ == "__main__":
    print("Generating completion icon …")
    p = generate()
    print(f"Done → {p}")
