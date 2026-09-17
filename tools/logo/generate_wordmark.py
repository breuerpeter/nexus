"""Generate the *nexus* wordmark logo → ``docs/assets/logo.svg``.

The script renders the logo from the Michroma font, vendored beside this script under the
Open Font License (OFL), so anyone can regenerate or restyle it without a design tool::

    uv run --with fonttools python tools/logo/generate_wordmark.py

Edit the CONFIG block below to change the word or its colours. The word is Michroma at 1:1,
the same size and stroke weight as the tagline, and the canvas is its ink box, so the letters
fill whatever box a viewer draws the mark in. See ``README.md`` for the knobs.
"""

from __future__ import annotations

import math

from _michroma import OUT_DIR, REPO_ROOT, layout_text, load_font

# ── CONFIG ───────────────────────────────────────────────────────────────────
TEXT = "nexus"  # the wordmark, lowercase, as the project spells its name

# The script emits the wordmark in two themes so it stays legible on any background: the dark
# one is the docs-site header and GitHub dark mode, the light one is GitHub light mode.
# Each entry: (filename, fill).
VARIANTS = (
    ("logo.svg", "#ffffff"),  # dark bg
    ("logo-light.svg", "#0a1020"),  # light bg
)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    glyphs, cmap, hmtx = load_font()
    paths, (left, top, right, bottom) = layout_text(glyphs, cmap, hmtx, TEXT, 0)
    # The canvas is the ink box: the word sits flush in it, so a viewer that scales the mark to a
    # fixed height draws the letters at that height rather than the font's full line box.
    width, height = math.ceil(right - left), math.ceil(bottom - top)
    markup = "".join(f'<path d="{d}"/>' for d in paths)

    for filename, fill in VARIANTS:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="nexus-wordmark" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" fill="{fill}" version="1.1">'
            # The fill sits on the group, not on the root alone: the lockup lifts this markup out of
            # its `<svg>` wrapper, so a fill left on the root would not travel with it.
            f'<g fill="{fill}" transform="translate({-left} {-top})">{markup}</g>'
            f"</svg>\n"
        )
        out = OUT_DIR / filename
        out.write_text(svg)
        print(f"wrote {out.relative_to(REPO_ROOT)}  ({width}x{height}, {fill})")


if __name__ == "__main__":
    main()
