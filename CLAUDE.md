# nexus

This repo holds a GPU-accelerated drone simulation framework: a `uv` **project** with the
framework as a single `nexus_sim` package and the docs site at the repo root. The Isaac Lab RL
training app lives under `nexus-rl/` as a **separate `uv` project** with its own virtual
environment, which depends on `nexus-sim` via an editable path source. That keeps the heavy,
prerelease-pinned Isaac Lab stack out of the core environment. Run it with
`uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py`.

- **The name takes three forms.** `nexus-sim` is the PyPI distribution. `nexus_sim` is the import
  package and every name Python resolves. `nexus` is the project and every other name, such as the
  command, the `nexus:` schema attributes and the logger. Prose writes `nexus` in lowercase and
  starts no sentence with it. `docs/design/conventions.md` states the rule, and the `Project` Vale
  style in `.vale/styles/Project/` checks the prose.
- **PX4 is the one first-class controller**, living in `nexus_sim/_src/`. Every other controller,
  `pid`, `policy`, `sampling_mpc`, and `acados_nmpc`, is a self-contained, **zero-arg** example
  under `nexus_sim/examples/controllers/<name>/`. Shared example machinery lives in
  `nexus_sim/examples/_lib/`. It ships in the wheel, and `nexus-rl` imports from it. Core must
  never import `nexus_sim.examples`, an import-linter contract in `.importlinter`. Run any
  example with `uv run -m nexus_sim.examples <name>`. Read `nexus_sim/examples/CLAUDE.md`
  before adding or changing an example.
- **No generic part names PX4.** PX4's folders are `nexus_sim/_src/peers/px4_sitl/` and
  `nexus_sim/_src/vehicle/controllers/px4/`. Nothing else under `nexus_sim/_src/` imports them or
  `nexus_sim.px4`, an import-linter contract in `.importlinter`. Its `ignore_imports` lists the
  imports that remain: those of the builder and those of the argument parser. A script that
  commands PX4 names it.
  It imports the offboard client from `nexus_sim.px4` and reads the link's address from `sim.ports`.
- **RTX sensors render in the Kit render peer**, a container the run starts on the host when the
  vehicle's Universal Scene Description (USD) file declares a sensor whose class requires it: a camera
  or lidar schema on a `Camera` or `OmniLidar` prim. Kit is a required peer, so no vehicle names it,
  and `docs/design/conventions.md` states the rule. The loop
  never runs inside Kit, and no module imports Kit, an import-linter contract. Read
  `nexus_sim/_src/peers/kit/CLAUDE.md` before changing the peer, its program, or the render link.
- **Each peer lives in one folder**, `nexus_sim/_src/peers/<name>/`, with its runner, its pin, its
  image or program and its link's definition. The docstring of `nexus_sim/_src/peers/__init__.py`
  states the layout. Read it before adding a peer or a file to one.
- **Component schemas live in a USD schema plugin** in `nexus_sim/_src/usd/`, which `import nexus_sim`
  registers. Read `nexus_sim/_src/usd/CLAUDE.md` before adding or changing a schema.
- **`evo` is GPLv3** → it lives only in the `ci` dependency-group and never ships as a dependency.
  Examples dump plain `.npz` and JSON files, and only `scripts/ci/evaluate_examples.py` imports
  `evo` to score them.
- **Use `uv` for all Python.** `uv sync` installs the framework, and
  `uv run nexus run --vehicle astro_max_base --scene empty` runs the command-line tool, PX4 only.
  Every run names its vehicle and scene, since a catalog holds no defaults, and the vehicle's USD
  declares its controller.
- **Docs** live in the `docs` dependency-group, not installed by default:
  `uv run --group docs mkdocs serve|build`. The site source is `docs/`, and
  `mkdocs.yml` is at the repo root. Read `docs/CLAUDE.md` before editing anything under `docs/`.
- **The API reference comes from docstrings.** Write **Google-style** docstrings, with
  `Args:`, `Returns:`, and `Raises:` sections, on the public `nexus_sim.*` surface. The reference
  filters out underscore-prefixed members and the `nexus_sim._src` layout, so document the curated
  public API, not internals.
- **Vehicle and scene USDs are content-addressed, hosted assets.** The catalog is
  `nexus_sim/_src/config/catalog.yaml`. The runtime resolver, `nexus_sim._src.assets.resolver`,
  fetches `{url, sha256}` to a verified cache at `assets/cache/`, which `$NEXUS_ASSET_CACHE`
  overrides. Read `scripts/assets/CLAUDE.md` before adding or updating an asset.
- **A recorded row's entity path names who wrote it**: the process, `sim` or a peer's folder name,
  then the component's role folder, then the instance, as in `sim/vehicle/sensors/imu`.
  `docs/reference/api/logging.md` states the rule. A component never spells its path: the
  orchestrator hands it a `ScopedLogger` at `set_logger`, and the component logs only its own
  row's name, such as `horizon`. The Recorder's channel keys are the same paths below the root.
- **Every script lives under `scripts/`**, and `docs/guide/contributing/scripts.md` or another page
  names each one a reader runs, `tests/docs/test_scripts.py`. `scripts/ground/` is the one
  exception: #43 deletes it.
- `docs/guide/running.md` documents **the decoupled Software In The Loop (SITL) flight workflow**,
  nexus sim plus PX4 SITL plus QGroundControl, and its port map and gotchas.
- **Lint and format:** `uvx pre-commit run -a`, which runs ruff. Config in
  `.pre-commit-config.yaml` and `[tool.ruff]` in `pyproject.toml`. The `extend-exclude` entry
  keeps `docs/` out of lint. `uv run lint-imports` checks the import contracts in `.importlinter`.
- **Tests:** `uv run --extra policy pytest -q`. A test that needs a CUDA device carries the `gpu` marker and no skip of its own, and one that flies real PX4 the `px4_sitl` marker. `docs/guide/contributing/development.md` says which CI leg runs each. The test tree mirrors the source
  tree: the tests of `nexus_sim/_src/<path>/` sit in `tests/<path>/`, and none sits at the root of
  `tests/`. A test that mirrors no source folder sits in a folder named for its kind, `ci`, `docs`,
  `examples`, `packaging` or `suite`. `tests/suite/test_layout.py` fails on any other folder.
- **Prose:** `vale sync` once, then `vale $(git ls-files '*.md' '*.py' ':!CHANGELOG.md')` checks the tracked Markdown, and the comments and docstrings in Python, against `.vale.ini`. The check leaves out `CHANGELOG.md`, which release-please writes, and so do both CI jobs. The ini pins each style by release URL, and the sync fetches them into `.vale/styles/`, a directory git ignores. CI runs the same check on the lines a pull request adds and fails on any alert among them. A nightly `vale-all` run, which you can also start from the Actions tab, checks the whole repo and fails on any alert, so a red night means something leaked past the gate. A domain word or acronym that a rule flags on a line you add, and that has no plain rewrite, goes into `.vale/styles/config/vocabularies/Names/accept.txt`, one pattern per line. An entry there exempts the word from every rule that flags it. A Python file that starts with a shebang or a comment defines its acronyms in a `# Acronyms:` comment after that header, because Vale's text rules skip a module docstring behind one.

## Commits

Conventional commits: `type(scope): description`, with the scope optional.

With release-please enabled, commit types drive the version bump:
- `feat` → minor: 0.1.0 → 0.2.0
- `fix` → patch: 0.1.0 → 0.1.1
- `feat!` or a `BREAKING CHANGE:` footer → **minor while the version is below 1.0.0**, so 0.1.0 → 0.2.0, and major after that. The release-please config sets `bump-minor-pre-major`, so a breaking change can't cut 1.0.0 by accident. Reaching 1.0.0 is a deliberate act.
- `chore`, `docs`, `ci`, `refactor`, `test`, `build`, `style` → changelog only, no bump
