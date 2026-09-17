"""Shared Michroma layout primitives for the brand-asset generators.

``generate_wordmark.py`` and ``generate_subtitle.py`` both lay out the vendored Michroma
font, under the Open Font License (OFL), into SVG outline paths at 1:1, so the font paths,
metrics, font loading, and glyph layout live here once: a single source of truth, no drift
between the two. The other two generators use different mechanisms and do **not** depend on
this module: ``generate_lockup.py`` composes the already-rendered SVGs, and
``generate_favicon.py`` rasterises with Pillow.
"""

from __future__ import annotations

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

FONT_DIR = Path(__file__).parent
FONT_PATH = FONT_DIR / "Michroma-Regular.ttf"
REPO_ROOT = FONT_DIR.parents[1]
OUT_DIR = REPO_ROOT / "docs" / "assets"

# Michroma metrics in the wordmark's coordinate space: Units Per Em (UPM) 2048, cap height 1536.
BASELINE = 1536.0  # SVG y of the baseline
VIEW_H = 1558  # canvas height; includes the round-letter descender overshoot


def load_font():
    """Load the vendored Michroma TrueType Font (TTF) → ``(glyph set, chosen cmap, hmtx table)``."""
    font = TTFont(FONT_PATH)
    return font.getGlyphSet(), font.getBestCmap(), font["hmtx"]


def layout_text(glyphs, cmap, hmtx, text, pen_x, *, tracking=0):
    """Lay out ``text`` from the font at 1:1 → ``(path-command strings, ink box)``.

    The layout flips each glyph into SVG's y-down space at ``BASELINE`` and advances it by its
    metric width plus ``tracking``, extra letter-spacing in font units at UPM 2048. Empty
    glyphs, for example the space, advance the pen but contribute no path.

    Args:
        glyphs: the font's glyph set, from :func:`load_font`.
        cmap: the font's character map from ``getBestCmap``: codepoint → glyph name.
        hmtx: the font's horizontal-metrics table: glyph name → (advance, lsb).
        text: the string to lay out.
        pen_x: the starting pen x, the left edge of the first glyph, in font units.
        tracking: extra letter-spacing added to each glyph's advance, in font units.

    Returns:
        ``(paths, box)``: the per-glyph SVG path-command strings, empties skipped, and the
        run's inked box as ``(left, top, right, bottom)`` in SVG coordinates. A run with no
        ink at all, for example a single space, boxes at the pen on the baseline.
    """
    paths = []
    left = right = pen_x
    top = bottom = BASELINE
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, pen_x, BASELINE)))
        cmds = pen.getCommands()
        if cmds:  # skip empty glyphs, for example space: no path, but still advance the pen
            paths.append(cmds)
        bounds = BoundsPen(glyphs)
        glyphs[name].draw(bounds)
        if bounds.bounds is not None:
            x0, y0, x1, y1 = bounds.bounds  # font space, y up
            left = min(left, pen_x + x0)
            right = max(right, pen_x + x1)
            top = min(top, BASELINE - y1)  # the flip swaps the two y edges
            bottom = max(bottom, BASELINE - y0)
        pen_x += hmtx[name][0] + tracking
    return paths, (left, top, right, bottom)
