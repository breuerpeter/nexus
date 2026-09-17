# Documentation

[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) builds the docs, and a push
to `main` deploys them to GitHub Pages automatically. The docs tools live in the `docs`
dependency group, which isn't installed by default.

## Structure

The docs follow [Diátaxis](https://diataxis.fr/), which splits documentation by the
reader's need, and the top-level tabs map onto it. **Home** orients a new reader and points
into the rest. **Guide** is the action-oriented material: learning-oriented tutorials and
goal-oriented how-tos. **Design** is the explanation: how the framework works and *why*.
**Reference** is the information-oriented lookup. When adding a page, put it in the tab
whose reader-need it serves rather than by topic, and keep the modes from bleeding into each
other. A how-to shouldn't turn into an essay, and reference shouldn't narrate.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

## Local preview

```bash
uv run --group docs mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000). Changes to Markdown files
show up live.

## Build

CI builds with `--strict`, which fails on broken links or missing nav entries, so run it
the same way before pushing:

```bash
uv run --group docs mkdocs build --strict
```

The build writes its output to `site/`.

## Prose lint

[Vale](https://vale.sh) checks the Markdown, and the comments and docstrings in Python, against `.vale.ini` at the repo root. That config loads Google's developer documentation style plus single rules from Red Hat, `write-good`, and `proselint`, each pinned by its release URL. To update a style, change the tag in its URL.

Install the pinned version from the [releases page](https://github.com/vale-cli/vale/releases) or with `brew install vale`. Fetch the styles once, then run the check on the tracked files:

```bash
vale sync
vale $(git ls-files '*.md' '*.py')
```

CI runs the same check on every pull request. One job judges only the lines the pull request adds and fails on any alert among them, so lines nobody touched don't block a merge. A nightly job checks the whole repo and fails on any alert, so a red night means something leaked past the gate. You can also start that job by hand from the Actions tab.

When the spelling rule flags a domain word on a line you add, add it to `.vale/styles/config/vocabularies/Names/accept.txt`, one pattern per line, case-insensitive. Put a file extension, an identifier, or a command in code font instead, which Vale skips. A British spelling isn't a domain word. The style is American.

## Diagrams

The diagram pipeline generates the system diagrams from source:

```bash
uv run --group docs python tools/diagrams/scripts/generate.py
```
