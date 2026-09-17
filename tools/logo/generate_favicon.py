"""Generate the browser favicon, a lowercase ``n`` in Michroma, white on black, written to
``docs/assets/favicon.ico``.

A small companion to ``generate_wordmark.py``: the same vendored Michroma, under the Open Font
License (OFL), rendered as a single lowercase ``n``, white on a solid-black square, and packed as a
multi-resolution ``.ico`` so the browser-tab mark matches the wordmark. Regenerate with::

    uv run --with pillow python tools/logo/generate_favicon.py

The script rasterises the glyph once on a high-resolution master canvas, centered on its ink
bounds, and downsamples it with LANCZOS to each icon size, so its proportions are the same at
every resolution. Edit the CONFIG block to change the letter, colours, or padding.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT_PATH = HERE / "Michroma-Regular.ttf"
OUT = HERE.parents[1] / "docs" / "assets" / "favicon.ico"

# ── CONFIG ───────────────────────────────────────────────────────────────────
GLYPH = "n"  # the mark: Michroma lowercase n, the word's first letter
FG = (255, 255, 255, 255)  # white glyph
BG = (0, 0, 0, 255)  # solid black square
MASTER = 256  # high-res master; every icon size downsamples from it
PAD_FRAC = 0.12  # ink padding each side as a fraction of the canvas; the glyph fills ~76%
ICO_SIZES = [16, 32, 48, 64, 128]  # packed into the one .ico; the browser picks the closest fit
# ─────────────────────────────────────────────────────────────────────────────


def render_master() -> Image.Image:
    """Render the glyph centered on its ink bounds on a ``MASTER``-square black canvas."""
    img = Image.new("RGBA", (MASTER, MASTER), BG)
    draw = ImageDraw.Draw(img)
    # Size the font so the glyph's larger ink dimension fills the padded box, then
    # center on the measured ink bbox; Michroma's side bearings ≠ the advance width.
    avail = MASTER * (1.0 - 2.0 * PAD_FRAC)
    probe = ImageFont.truetype(str(FONT_PATH), MASTER)
    x0, y0, x1, y1 = draw.textbbox((0, 0), GLYPH, font=probe)
    font = ImageFont.truetype(str(FONT_PATH), max(1, round(MASTER * avail / max(x1 - x0, y1 - y0))))
    x0, y0, x1, y1 = draw.textbbox((0, 0), GLYPH, font=font)
    x = (MASTER - (x1 - x0)) / 2.0 - x0
    y = (MASTER - (y1 - y0)) / 2.0 - y0
    draw.text((x, y), GLYPH, font=font, fill=FG)
    return img


def main() -> None:
    # The .ico packs every listed size from one source via `sizes=`; Pillow downsamples the
    # high-res master to each. The master is 256², so every frame stays crisp.
    master = render_master()
    master.save(OUT, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {OUT} ({', '.join(f'{s}x{s}' for s in ICO_SIZES)})")


if __name__ == "__main__":
    main()
