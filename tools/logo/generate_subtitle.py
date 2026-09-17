"""Generate the *nexus* tagline → ``docs/assets/subtitle{,-light}.svg``.

A small companion to ``generate_wordmark.py``: the project description, set in the same
vendored Michroma, under the Open Font License (OFL), and emitted as outline paths via the shared
``_michroma`` layout
so GitHub / the docs site render it without the font installed. It sits directly under the
wordmark in the README, so it ships in the same light + dark pair::

    uv run --with fonttools python tools/logo/generate_subtitle.py

Edit the CONFIG block to change the wording, tracking, or colours.
"""

from __future__ import annotations

from _michroma import OUT_DIR, REPO_ROOT, VIEW_H, layout_text, load_font

# ── CONFIG ───────────────────────────────────────────────────────────────────
TEXT = "GPU-accelerated drone simulation"
TRACKING = 90  # extra letter-spacing in font units at Units Per Em (UPM) 2048: geometric-tagline air
PAD = 4  # ink padding on each side, in font units

# (filename, fill): mirrors the wordmark's dark / light pair.
VARIANTS = (
    ("subtitle.svg", "#cbd5e1"),  # dark bg: slate-300
    ("subtitle-light.svg", "#475569"),  # light bg: slate-600
)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    glyphs, cmap, hmtx = load_font()

    paths, (_, _, ink_right, _) = layout_text(glyphs, cmap, hmtx, TEXT, PAD, tracking=TRACKING)
    width = round(ink_right + PAD)
    markup = "".join(f'<path d="{d}"/>' for d in paths)

    for filename, fill in VARIANTS:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="nexus-tagline" '
            f'viewBox="0 0 {width} {VIEW_H}" width="{width}" height="{VIEW_H}" '
            f'fill="{fill}" version="1.1">'
            f'<g fill="{fill}">{markup}</g>'
            f"</svg>\n"
        )
        out = OUT_DIR / filename
        out.write_text(svg)
        print(f"wrote {out.relative_to(REPO_ROOT)}  ({width}x{VIEW_H}, {fill})")


if __name__ == "__main__":
    main()
