# Logo generator

Generates the **nexus** word mark from the **Michroma** font, so anyone can regenerate or
restyle it without a design tool, in two themes:

- `docs/assets/logo.svg`, for **dark backgrounds**: the `mkdocs.yml` header, `theme.logo`, and
  the dark lockup. White `nexus`.
- `docs/assets/logo-light.svg`, for **light backgrounds**: the light lockup. Ink `nexus` in
  `#0a1020`.

The GitHub README embeds `docs/assets/lockup-banner.svg`, see [Lockup & hero banner](#lockup--hero-banner).

```bash
uv run --with fonttools python tools/logo/generate_wordmark.py
```

## What it produces

The word set in lowercase Michroma on a transparent SVG canvas, white in the dark variant and
ink in the light one. The canvas is the word's ink box, so a viewer that scales the mark to a
fixed height draws the letters at that height, with no empty band over them.

The word is Michroma at **1:1**, per the `_michroma.py` metrics with cap height 1536, so it
matches the tagline in size and **stroke weight**, with no scaling.

## Files

- `generate_wordmark.py`: the renderer. Lays out the word from the font and writes the SVG.
- `_michroma.py`: shared font primitives, `load_font` + the 1:1 `layout_text` + the metrics,
  used by `generate_wordmark.py` and `generate_subtitle.py`, so the Michroma layout lives in
  one place. The lockup and favicon generators use different mechanisms and don't depend on it.
- `Michroma-Regular.ttf` + `OFL.txt`: the vendored font and its SIL Open Font License.

## Lockup & hero banner

`generate_lockup.py` stacks the word mark over the tagline into
`docs/assets/lockup{,-light}.svg`, the mark that the README header and social card
embed. It then drops the dark lockup onto the Astro-snow photo →
`docs/assets/lockup-banner.svg`, the home-page hero. The banner mirrors
`nexus-card.yml`, photo + legibility scrim + centred lockup, but is a
single self-contained SVG. The `.jpg` is base64-embedded so it renders inside an
`<img>`, because browsers block external refs in sandboxed SVG, while the lockup stays
vector so the word mark is crisp at any size. No font or imaging libraries needed:

```bash
uv run python tools/logo/generate_lockup.py   # run after the wordmark + subtitle
```

## Favicon

`generate_favicon.py` renders a single Michroma lowercase **n**, white on a black square,
into a multi-resolution `docs/assets/favicon.ico` at 16 to 128 pixels, so the browser-tab mark
matches the word mark. It renders the glyph to pixels once on a high-res master and scales it down
to each icon size, so the proportions are the same at every resolution:

```bash
uv run --with pillow python tools/logo/generate_favicon.py
```

Edit its `CONFIG` block to change the letter, colors, padding, or the packed sizes.

## Changing it

Edit the `CONFIG` block at the top of `generate_wordmark.py` and re-run:

- **`TEXT`**: the word, default `"nexus"`.
- **`VARIANTS`**: the themes to emit, each `(filename, fill)`.

Every run lays the word mark out from the font, so a change to `TEXT` needs no other edit.
Re-check the letter spacing after one: the mark takes Michroma's own spacing.

## Preview

To preview as the site shows it, run the docs server with
`uv run --group docs mkdocs serve`, or open the SVG in a browser.
