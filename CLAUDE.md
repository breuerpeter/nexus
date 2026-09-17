# nexus

nexus is a GPU-accelerated drone simulation framework: a `uv` **workspace** with the
framework as a single `nexus` package and the docs site at the repo root. The Isaac Lab RL
training app lives under `nexus-rl/` as a **separate `uv` project** with its own virtual
environment, which depends on `nexus` via an editable path source. That keeps the heavy,
prerelease-pinned Isaac Lab stack out of the core environment. Run it with
`uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py`.

- **PX4 is the one first-class controller**, living in `nexus/_src/`. Every other controller,
  `pid`, `policy`, `sampling_mpc`, and `acados_nmpc`, is a self-contained, **zero-arg** example
  under `nexus/examples/controllers/<name>/`. Shared example machinery lives in
  `nexus/examples/_lib/`. It ships in the wheel, and `nexus-rl` imports from it. Core must
  never import `nexus.examples`, an import-linter contract in `.importlinter`. Run any
  example with `uv run -m nexus.examples <name>` or RTX-rendered via
  `uv run nexus script nexus.examples <name>`. Read `nexus/examples/CLAUDE.md`
  before adding or changing an example.
- **`evo` is GPLv3** → it lives only in the `ci` dependency-group and never ships as a dependency.
  Examples dump plain `.npz` and JSON files, and only `scripts/ci/evaluate_examples.py` imports
  `evo` to score them.
- **Use `uv` for all Python.** `uv sync` installs the framework, and
  `uv run nexus run` runs the command-line tool, PX4 only. Its defaults are the registry
  default vehicle `astro_max_base` and `--control px4-sitl`.
- **Docs** live in the `docs` dependency-group, not installed by default:
  `uv run --group docs mkdocs serve|build`. The site source is `docs/`, and
  `mkdocs.yml` is at the repo root. Read `docs/CLAUDE.md` before editing anything under `docs/`.
- **The API reference comes from docstrings.** Write **Google-style** docstrings, with
  `Args:`, `Returns:`, and `Raises:` sections, on the public `nexus.*` surface. The reference
  filters out underscore-prefixed members and the `nexus._src` layout, so document the curated
  public API, not internals.
- **Vehicle and scene USDs are content-addressed, hosted assets.** The registry is
  `nexus/_src/config/registry.yaml`. The runtime resolver, `nexus._src.assets.resolver`,
  fetches `{url, sha256}` to a verified cache at `assets/cache/`, which `$NEXUS_ASSET_CACHE`
  overrides. Read `scripts/assets/CLAUDE.md` before adding or updating an asset.
- `docs/guide/running.md` documents **the decoupled Software In The Loop (SITL) flight workflow**,
  nexus sim plus PX4 SITL plus QGroundControl, and its port map and gotchas.
- **Lint and format:** `uvx pre-commit run -a`, which runs ruff. Config in
  `.pre-commit-config.yaml` and `[tool.ruff]` in `pyproject.toml`. The `extend-exclude` entry
  keeps `docs/` out of lint.
- **Tests:** `uv run --extra policy --with pytest pytest tests -q`.
- **Prose:** `vale sync` once, then `vale $(git ls-files '*.md' '*.py')` checks the tracked Markdown, and the comments and docstrings in Python, against `.vale.ini`. The ini pins each style by release URL, and the sync fetches them into `.vale/styles/`, a directory git ignores. CI runs the same check on the lines a pull request adds and fails on any alert among them. A nightly `vale-all` run, which you can also start from the Actions tab, checks the whole repo and fails on any alert, so a red night means something leaked past the gate. A domain word or acronym that a rule flags on a line you add, and that has no plain rewrite, goes into `.vale/styles/config/vocabularies/Names/accept.txt`, one pattern per line. An entry there exempts the word from every rule that flags it. A Python file that starts with a shebang or a comment defines its acronyms in a `# Acronyms:` comment after that header, because Vale's text rules skip a module docstring behind one.

## Commits

Conventional commits: `type(scope): description`, with the scope optional.

With release-please enabled, commit types drive the version bump:
- `feat` → minor: 0.1.0 → 0.2.0
- `fix` → patch: 0.1.0 → 0.1.1
- `feat!` or a `BREAKING CHANGE:` footer → **minor while the version is below 1.0.0**, so 0.1.0 → 0.2.0, and major after that. The release-please config sets `bump-minor-pre-major`, so a breaking change can't cut 1.0.0 by accident. Reaching 1.0.0 is a deliberate act.
- `chore`, `docs`, `ci`, `refactor`, `test`, `build`, `style` → changelog only, no bump
