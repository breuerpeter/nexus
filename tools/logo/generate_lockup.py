"""Compose the wordmark + tagline into one *lockup* SVG → ``docs/assets/lockup{,-light}.svg``,
then drop that lockup onto the Astro-snow photo → ``docs/assets/lockup-banner.svg``.

Run *after* ``generate_wordmark.py`` and ``generate_subtitle.py``: this derives from their
canonical outputs, ``logo*.svg`` + ``subtitle*.svg``, rather than re-laying the
font, so the lockup can never drift from the standalone marks::

    uv run --with fonttools python tools/logo/generate_wordmark.py
    uv run --with fonttools python tools/logo/generate_subtitle.py
    uv run python tools/logo/generate_lockup.py        # no font needed here

Both marks render at the *same* width, the wordmark at its native scale, the
tagline scaled by wordmark_width / tagline_width, stacked with a fixed gap. The
README and the social card both embed this single file, so this file defines the
wordmark→tagline spacing once and it stays the same in both places.

The *banner*, ``lockup-banner.svg``, is the home-page hero: a wide strip of the
Astro-snow photo, base64-embedded so it renders inside an ``<img>``, unlike an
external ``href`` which browsers block in sandboxed SVG, a legibility scrim, and
the dark lockup left as *vector* so the wordmark stays crisp at any size. Its wide
aspect + modest lockup suit a full-bleed banner, ``object-fit: cover``,
so the wordmark isn't oversized when the strip spans the viewport. Single
self-contained file, with no font and no imaging libs.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE.parents[1] / "docs" / "assets"

# ── CONFIG ───────────────────────────────────────────────────────────────────
GAP = 333  # vertical space below the wordmark box before the tagline, in wordmark units

# (out, wordmark source, tagline source): dark / light, mirroring the marks.
VARIANTS = (
    ("lockup.svg", "logo.svg", "subtitle.svg"),
    ("lockup-light.svg", "logo-light.svg", "subtitle-light.svg"),
)

# The home-page hero banner: dark lockup over a wide strip of the Astro-snow photo. Wide aspect +
# a modest lockup so the wordmark reads at a sensible size when the banner spans the viewport.
BANNER = {
    "out": "lockup-banner.svg",
    "photo": "images/astro_snow.jpg",  # base64-embedded so it renders inside an <img>
    "scrim": "social-card-scrim.svg",  # legibility gradient; its <defs> reused at the banner size
    "lockup": "lockup.svg",  # the dark lockup, kept vector
    "width": 1920,
    "height": 560,
    "lockup_width": 600,  # rendered lockup width inside the banner; smaller = less dominant wordmark
    "photo_focus": 0.3,  # fraction of the vertical crop taken from the top; 0.5 = centered, <0.5 shows more top
}
# ─────────────────────────────────────────────────────────────────────────────

_SVG = re.compile(r'<svg[^>]*\bviewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>(.*)</svg>\s*$', re.S)
_DEFS = re.compile(r"<defs>.*?</defs>", re.S)


def _parse(path):
    """Return (width, height, inner-markup) for a single-root SVG file."""
    m = _SVG.match(path.read_text())
    if m is None:
        raise ValueError(f"{path} is not a plain <svg viewBox=…>…</svg> file")
    return float(m.group(1)), float(m.group(2)), m.group(3)


def _jpeg_size(path):
    """Return (width, height) of a baseline or progressive Joint Photographic Experts Group (JPEG) image by
    reading its Start Of Frame (SOF) marker, without the Python Imaging Library (PIL). The SOF0-15 range
    also holds the Define Huffman Table (DHT), JPG extension and Define Arithmetic Coding (DAC) markers,
    which the scan skips.
    """
    data = path.read_bytes()
    i = 2  # skip the Start Of Image (SOI) marker, FF D8
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # SOF0-15, not DHT, JPG or DAC
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        i += 2 + ((data[i + 2] << 8) | data[i + 3])  # skip this segment
    raise ValueError(f"{path}: no JPEG SOF marker found")


def _build_banner():
    """Compose a wide strip of the Astro-snow photo + scrim + dark lockup into one self-contained SVG."""
    w, h, lw = BANNER["width"], BANNER["height"], BANNER["lockup_width"]
    photo = ASSETS / BANNER["photo"]
    b64 = base64.b64encode(photo.read_bytes()).decode("ascii")
    # Place the photo "cover"-style, filling the banner and keeping aspect, but with an explicit vertical
    # offset so the crop isn't forced to center: photo_focus picks how much of the overflow comes off the
    # top compared to the bottom. preserveAspectRatio can only do top/center/bottom, hence the manual maths.
    pw, ph = _jpeg_size(photo)
    pscale = max(w / pw, h / ph)
    sw, sh = pw * pscale, ph * pscale
    px = (w - sw) / 2  # center horizontally
    py = -(sh - h) * BANNER["photo_focus"]  # focus<0.5 crops less off the top, more off the bottom
    # Reuse the scrim's gradient <defs>, but draw the rect at the banner size; its native rect is
    # card-sized. The gradient is in objectBoundingBox units, so it rescales to any aspect.
    scrim_defs = _DEFS.search((ASSETS / BANNER["scrim"]).read_text()).group(0)
    lk_w, lk_h, lk_inner = _parse(ASSETS / BANNER["lockup"])
    scale = lw / lk_w
    x = (w - lw) / 2
    y = (h - lk_h * scale) / 2  # vertically center the lockup in the banner
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="nexus-banner" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" version="1.1">'
        # Photo at cover scale, positioned by px/py; the <svg> viewport clips the overflow.
        f'<image href="data:image/jpeg;base64,{b64}" x="{px:.1f}" y="{py:.1f}" '
        f'width="{sw:.1f}" height="{sh:.1f}" preserveAspectRatio="none"/>'
        f"{scrim_defs}"  # legibility-scrim gradient, reused from social-card-scrim.svg
        f'<rect width="{w}" height="{h}" fill="url(#scrim)"/>'
        f'<g transform="translate({x:.1f} {y:.1f}) scale({scale:.6f})">{lk_inner}</g>'
        f"</svg>\n"
    )
    (ASSETS / BANNER["out"]).write_text(svg)
    print(f"wrote docs/assets/{BANNER['out']}  ({w}x{h}, lockup {lw}px, focus {BANNER['photo_focus']})")


def main():
    for out, logo_f, sub_f in VARIANTS:
        wm_w, wm_h, wm_inner = _parse(ASSETS / logo_f)
        tg_w, tg_h, tg_inner = _parse(ASSETS / sub_f)
        scale = wm_w / tg_w  # shrink the tagline to the wordmark's rendered width
        tagline_y = wm_h + GAP
        total_h = round(wm_h + GAP + tg_h * scale)
        width = round(wm_w)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="nexus-lockup" '
            f'viewBox="0 0 {width} {total_h}" width="{width}" height="{total_h}" version="1.1">'
            f"{wm_inner}"  # wordmark at native scale
            f'<g transform="translate(0 {tagline_y}) scale({scale:.6f})">{tg_inner}</g>'
            f"</svg>\n"
        )
        (ASSETS / out).write_text(svg)
        print(f"wrote docs/assets/{out}  ({width}x{total_h}, gap={GAP})")

    _build_banner()  # derives from the dark lockup just written


if __name__ == "__main__":
    main()
