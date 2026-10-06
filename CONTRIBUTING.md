# Contributing to nexus

Thanks for your interest in nexus. The full contributing guide lives in
[`docs/guide/contributing/`](docs/guide/contributing/index.md). This page holds the steps you need
first.

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then install the framework and the pre-commit hooks:

```bash
uv sync
uvx pre-commit install
uvx pre-commit install --hook-type commit-msg
```

[Development](docs/guide/contributing/development.md) covers the hooks and the tests.

## Pull requests

- Write each commit message as a [conventional commit](https://www.conventionalcommits.org/):
  `type(scope): description`.
- Run `uvx pre-commit run -a` before you commit.
- Vale checks the prose lines you add: the Markdown, and the comments and docstrings in Python.
- GPU CI runs once a maintainer adds the `gpu` label to the pull request.

[Pull requests](docs/guide/contributing/pull-requests.md) gives the full rules, and
[Reporting issues](docs/guide/contributing/reporting-issues.md) covers bug reports.
