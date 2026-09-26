# Development

nexus is a [`uv`](https://docs.astral.sh/uv/) workspace with a single `nexus`
framework package. Install it and run the command-line tool:

```bash
uv sync                 # installs the framework
uv run nexus run   # fly the default vehicle against PX4 SITL
```

## Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to run code quality checks
before each commit. The hooks enforce:

- **ruff**: lints and formats Python
- **uv-lock**: keeps `uv.lock` in sync with `pyproject.toml`
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
uvx --from import-linter lint-imports
```

## Tests

```bash
uv run --extra policy --with pytest pytest tests -q
```
