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

## What CI flies

A pull request with the `gpu` label flies every example at once on one GPU box. It gates each example on its correctness rows in `scripts/ci/examples_baselines.json`, so it pays for one box and still proves the change. Main flies each example on a box of its own. It gates the same rows plus each `rtf` row, and it uploads the recordings the docs embed. The weekly schedule flies the benchmark matrix, one cell per box. The Real-Time Factor (RTF) gates run on main, because an RTF is only comparable when a flight has the box to itself. Flights that share a box share its CPUs, and each one's RTF drops. So a speed regression shows on the merge commit, not on the pull request.
