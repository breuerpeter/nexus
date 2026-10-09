# Development

The repo is a [`uv`](https://docs.astral.sh/uv/) project with a single `nexus_sim`
framework package. Install it and run the command-line tool:

```bash
uv sync                 # installs the framework
uv run nexus run --vehicle astro_max_base --scene empty   # fly a vehicle against PX4 SITL
```

## Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to run code quality checks
before each commit. The hooks enforce:

- **ruff**: lints and formats Python
- **uv-lock**: keeps `uv.lock` and `nexus-rl/uv.lock` in sync with the root `pyproject.toml`.
  A commit that touches only `nexus-rl/` skips it, so the pull request's `lint` check catches a
  stale lock there
- **typos**: catches common misspellings
- **conventional-pre-commit**: enforces [conventional commit](https://www.conventionalcommits.org/) messages

### Setup

Install the hooks, a one-time step:

```bash
uvx pre-commit install
uvx pre-commit install --hook-type commit-msg
```

### Usage

Hooks run automatically on `git commit`. To run them manually against all files:

```bash
uvx pre-commit run --all-files
```

If a hook modifies a file, for example when ruff auto-formats it, pre-commit aborts the
commit. Stage the changes and commit again.

## Import boundaries

[import-linter](https://import-linter.readthedocs.io/) enforces the module boundaries, with
the contracts in `.importlinter`, and CI runs it too:

```bash
uv run lint-imports
```

## Tests

```bash
uv run --extra policy pytest -q
```

Two markers say what a test needs beyond a CPU. A `gpu` test needs a CUDA device, and it skips on a machine with none. A `px4_sitl` test flies a real PX4 Software In The Loop (SITL) build from the PX4 tree on the machine. CI runs each set on its own leg: cpu-pytest runs `-m "not gpu and not px4_sitl"`, and gpu-pytest runs `-m "gpu or px4_sitl" --require-cuda`, which fails a `gpu` test that skips. Mark a test that needs CUDA `gpu`, or only its CUDA parameter with `pytest.param("cuda:0", marks=pytest.mark.gpu)`, and give it no skip of its own: a test that skips for CUDA with no `gpu` marker fails. pytest runs with `--strict-markers`, so a misspelled marker fails the run.

The tests run in parallel, one worker per CPU, and cpu-pytest prints its 25 slowest tests. gpu-pytest runs its tests in one process, `-n 0`. A PX4 instance's Hardware In The Loop (HIL) port is 4560 plus its number, and its offboard port is 14540 plus its number. A test that binds or dials one of them carries `pytest.mark.xdist_group("px4_ports")`. The tests of that group run on one worker, one at a time, since each port is the machine's and two workers would collide on it. Pass `-n 0` to run every test in one process, as when you debug with `pdb`.

### Peer fakes

Each peer the framework ships comes with a fake, a stand-in in this process that speaks the peer's link and starts no process. `Px4Fake`, in `nexus_sim/_src/peers/px4_sitl/fake.py`, answers each `HIL_SENSOR` with one fixed `HIL_ACTUATOR_CONTROLS` over the Hardware In The Loop (HIL) lockstep. `KitFake`, in `nexus_sim/_src/peers/kit/fake.py`, answers each `frame` request one frame behind, with a blank frame at the size each sensor declares.

A test sends a peer to its fake through the builder's peer mapping, by the peer's name:

```python
from nexus_sim._src.build.launch import build_from_launch
from nexus_sim._src.peers.kit.fake import KitFake
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake

loop = build_from_launch(launch, peers={"px4_sitl": Px4Fake, "kit": KitFake})
```

The controller, the renderer and the loop then run as they do against the real process. So a fake proves the loop's side of the link, its encoding, its stage order and its error paths, and nothing about the process behind it. A run against `Px4Fake` has no offboard link, so `sim.ports["offboard"]` raises and names the fake.

## Merging to main

A ruleset on `main` refuses a direct push. A change lands only by pull request, and only by squash merge, so each commit on `main` is one conventional commit that release-please reads. The branch must be up to date with `main`, and six checks must pass: `lint`, `cpu-pytest`, `vale-changed`, `examples`, `gpu-pytest` and `rl`. The last three are the GPU legs. A pull request without the `gpu` label runs no GPU leg, so those three report skipped, and a skipped check counts as passed. GitHub deletes a pull request's branch once it merges. The repo administrator can merge a pull request past a red check, but can't push to `main` either.

## What CI flies

A pull request with the `gpu` label flies every example at once on one GPU box. It gates each example on its correctness rows in `scripts/ci/examples_baselines.json`, so it pays for one box and still proves the change. Main flies each example on a box of its own. It gates the same rows plus each `rtf` row, and it uploads the recordings the docs embed. The weekly schedule flies the benchmark matrix, one cell per box. The Real-Time Factor (RTF) gates run on main, because an RTF is only comparable when a flight has the box to itself. Flights that share a box share its CPUs, and each one's RTF drops. So a speed regression shows on the merge commit, not on the pull request.
