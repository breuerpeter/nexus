# nexus

Built on Newton, Warp and OpenUSD, nexus lets you develop drone subsystems and test them as part of
the full flight stack. This repo holds it: a `uv` **project** with the framework as a single
`nexus_uav` package and the docs site at the repo root. The Isaac Lab RL
training app lives under `nexus-rl/` as a **separate `uv` project** with its own virtual
environment, which depends on `nexus-uav` via an editable path source. That keeps the heavy,
prerelease-pinned Isaac Lab stack out of the core environment. Run it with
`uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py`.

- **The name takes three forms.** `nexus-uav` is the PyPI distribution. `nexus_uav` is the import
  package and every name Python resolves. `nexus` is the project and every other name, such as the
  command, the `nexus:` schema attributes and the logger. Prose writes `nexus` in lowercase and
  starts no sentence with it. `docs/design/conventions.md` states the rule, and the `Project` Vale
  style in `.vale/styles/Project/` checks the prose.
- **PX4 is the one first-class controller**, living in `nexus_uav/_src/`. Every other controller,
  `pid`, `policy`, `sampling_mpc`, and `acados_nmpc`, is a self-contained, **zero-arg** example
  under `nexus_uav/examples/controllers/<name>/`. Shared example machinery lives in
  `nexus_uav/examples/_lib/`. It ships in the wheel, and `nexus-rl` imports from it. `nexus-rl`
  imports nothing from `nexus_uav._src`, a ruff banned-import rule in `pyproject.toml`. Core must
  never import `nexus_uav.examples`, an import-linter contract in `.importlinter`. Run any
  example with `uv run -m nexus_uav.examples <name>`. Read `nexus_uav/examples/CLAUDE.md`
  before adding or changing an example.
- **No generic part names PX4.** PX4's folders are `nexus_uav/_src/peers/px4_sitl/` and
  `nexus_uav/_src/vehicle/controllers/px4/`. Nothing else under `nexus_uav/_src/` imports them or
  `nexus_uav.px4`, an import-linter contract in `.importlinter`. Its `ignore_imports` lists the one
  import that remains, in the argument parser. A script that commands PX4 names it.
  It imports the offboard client from `nexus_uav.px4` and reads the link's address from `sim.ports`.
- **RTX sensors render in the loop's process** through ovrtx and ovstage, the extra `nexus-uav[rtx]`. A
  camera or lidar schema on a `Camera` or `OmniLidar` prim declares one. Its class requires the render
  stage, `nexus_uav/_src/vehicle/sensors/rtx/stage.py`, by the name `ovrtx`. The build makes it once per run,
  as a required peer, and the loop hands it to each sensor at `connect`. No module imports ovrtx at its own
  import, so the core runs without the extra. The Kit render peer stays, required by no class, until #362
  removes it. Read `nexus_uav/_src/peers/kit/CLAUDE.md` before changing it.
- **Each peer lives in one folder**, `nexus_uav/_src/peers/<name>/`, with its runner, its pin, its
  image or program and its link's definition. The docstring of `nexus_uav/_src/peers/__init__.py`
  states the layout. Read it before adding a peer or a file to one.
- **Component schemas live in a USD schema plugin** in `nexus_uav/_src/usd/`, which `import nexus_uav`
  registers. Read `nexus_uav/_src/usd/CLAUDE.md` before adding or changing a schema.
- **One contract serves every role.** A component subclasses `Component`, states its stages and
  overrides the lifecycle members it has: `connect`, `close`, `set_logger`, `attached`, `requires`,
  `needs_instance`, `ports`, `artifacts` and `name`. The loop, the builder and `Sim` read those from
  every component and find none by name. A test stand-in subclasses `Component` too. A controller
  writes its own `controls` signal, as wide as its actuators. The registry, `Registry`, maps each
  schema to its class, a component's or a declared peer's, and each name a class requires a peer by.
  A project registers through the `nexus.registry` entry-point group, and a test hands a run its
  stand-ins and its peers' fakes through `Sim(registry=...)`. The builder names no component and no
  peer. A script reads any recorded history by indexing the run with the history's path,
  `sim["vehicle/sensors/imu/imu"]`, a signal's live value with `sim.read(path)`, and `Sim` gives no
  role an accessor.
  `docs/guide/your-own-component.md` walks a project through it.
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
  `Args:`, `Returns:`, and `Raises:` sections, on the public `nexus_uav.*` surface. The reference
  filters out underscore-prefixed members and the `nexus_uav._src` layout, so document the curated
  public API, not internals.
- **Vehicle and scene USDs are content-addressed, hosted assets.** The catalog is
  `nexus_uav/_src/config/catalog.yaml`. A vehicle entry is flat, a `usd`, or composed, a `base` entry
  plus `layers`, each a hosted USD file the resolver stacks over the base, the first listed
  strongest. `astro_max_fpv` is `astro_max_base` plus a camera layer. The runtime resolver,
  `nexus_uav._src.assets.resolver`,
  fetches `{url, sha256}` to a verified cache at `assets/cache/`, which `$NEXUS_ASSET_CACHE`
  overrides. Read `scripts/assets/CLAUDE.md` before adding or updating an asset.
- **A recorded row's entity path names who wrote it**: the process, `sim` or a peer's folder name,
  then the component's role folder, then the instance, as in `sim/vehicle/sensors/imu`.
  `docs/reference/api/logging.md` states the rule. A component never spells its path: the
  orchestrator hands it a `ScopedLogger` at `set_logger`, and the component logs only its own
  row's name, such as `horizon`. The Recorder's histories take the same keys below the root.
- **The Recorder keeps every row of a run, and the Logger writes it in blocks.** Each history has
  a staging buffer on the device, which the loop drains onto host blocks when full, and the Logger
  writes each drained block and the last at teardown, `docs/reference/api/logging.md`. A component
  logs live, through `set_logger`, only a value with no fixed width, such as a camera's frame. No
  component states a per-tick log step, and `Sim` takes no `observe`. The Logger logs the scene's
  meshes once through Newton's viewer base class, never `ViewerRerun`.
- **Every script lives under `scripts/`**, and `docs/guide/contributing/scripts.md` or another page
  names each one a reader runs, `tests/docs/test_scripts.py`. `scripts/ground/` is the one
  exception: #43 deletes it.
- `docs/guide/running.md` documents **the decoupled Software In The Loop (SITL) flight workflow**,
  nexus sim plus PX4 SITL plus QGroundControl, and its port map and gotchas.
- **Lint and format:** `uvx pre-commit run -a`, which runs ruff. Config in
  `.pre-commit-config.yaml` and `[tool.ruff]` in `pyproject.toml`. The `extend-exclude` entry
  keeps `docs/` out of lint. `uv run lint-imports` checks the import contracts in `.importlinter`.
- **Tests:** `uv run --extra policy pytest -q`, which runs one worker per logical CPU, fewer when the free memory holds fewer at 3 GiB each. A test that binds or dials a PX4 port carries `pytest.mark.xdist_group("px4_ports")`. A test that needs a CUDA device carries the `gpu` marker and no skip of its own, and one that flies real PX4 the `px4_sitl` marker. `docs/guide/contributing/development.md` says which CI leg runs each. The test tree mirrors the source
  tree: the tests of `nexus_uav/_src/<path>/` sit in `tests/<path>/`, and none sits at the root of
  `tests/`. A test that mirrors no source folder sits in a folder named for its kind, `ci`, `docs`,
  `examples`, `packaging` or `suite`. `tests/suite/test_layout.py` fails on any other folder.
- **The full suite shares its machine.** While another session here runs the suite or flies PX4, a
  `px4_ports` test fails with `the HIL port 4560 is in use by process N` or `OSError: [Errno 98] Address
  already in use`, and `ps -p N` shows whose run holds the port. Each worker opens its own CUDA context, so
  while another process holds GPU memory, which `nvidia-smi` lists, a CUDA test can fail with `Warp CUDA
  error 2: out of memory` or `Failed to acquire primary context for device cuda:0`. Under that load a timed
  test can miss its deadline too, such as the 20 s exit of `tests/core/test_stop_keeps_recording.py`. Re-run
  the failed tests with `-n 0`: a test that passes alone is no fault of the change.
- **A run that never ends has a worker stuck after a GPU fault.** The output repeats `Warp CUDA error 700: an
  illegal memory access`, and `journalctl -k | grep Xid` shows `NVRM: Xid 31` for the worker's pid: a kernel
  wrote to address 0. The worker spins past `kill` and can hold port 4560, and the suite sets no per-test
  timeout, so stop it with `kill -9` and re-run. Seen twice on 2026-10-15 in the `px4_ports` group's worker,
  whose tests passed alone. The cause isn't known.
- **`tests/packaging` errors under `uv` 0.9.22**: `uv build` exits 2 extracting its own source distribution,
  on main too. Pass `--ignore=tests/packaging` to a local run. The cpu-pytest leg builds the package in CI.
- **Prose:** `vale sync` once, then `python3 scripts/ci/vale_added.py` checks the lines the branch adds to the tracked Markdown, and to the comments and docstrings in Python, against `.vale.ini`. It fails on any alert among them, as the vale-changed check does on a pull request. The `vale-added` pre-commit hook runs it on every commit and under `uvx pre-commit run -a`. `vale $(git ls-files '*.md' '*.py' ':!CHANGELOG.md')` checks the whole tree, as the nightly does. Both leave out `CHANGELOG.md`, which release-please writes. The ini pins each style by release URL, and the sync fetches them into `.vale/styles/`, a directory git ignores. A nightly `vale-all` run, which you can also start from the Actions tab, checks the whole repo and fails on any alert, so a red night means something leaked past the gate. A domain word or acronym that a rule flags on a line you add, and that has no plain rewrite, goes into `.vale/styles/config/vocabularies/Names/accept.txt`, one pattern per line. An entry there exempts the word from every rule that flags it. A Python file that starts with a shebang or a comment defines its acronyms in a `# Acronyms:` comment after that header, because Vale's text rules skip a module docstring behind one.

## Commits

Conventional commits: `type(scope): description`, with the scope optional.

With release-please enabled, commit types drive the version bump:
- `feat` → minor: 0.1.0 → 0.2.0
- `fix` → patch: 0.1.0 → 0.1.1
- `feat!` or a `BREAKING CHANGE:` footer → **minor while the version is below 1.0.0**, so 0.1.0 → 0.2.0, and major after that. The release-please config sets `bump-minor-pre-major`, so a breaking change can't cut 1.0.0 by accident. Reaching 1.0.0 is a deliberate act.
- `chore`, `docs`, `ci`, `refactor`, `test`, `build`, `style` → changelog only, no bump

## References
- `ARCHITECTURE.md`: the target architecture of the codebase as a whole. Read it before you plan a change.
