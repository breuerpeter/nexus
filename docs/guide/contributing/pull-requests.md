# Pull requests

- Use [conventional commits](https://www.conventionalcommits.org/): `type(scope): description`
- Run `uvx pre-commit run -a` before committing. It runs ruff lint and format, and typos
- CI and [pre-commit.ci](https://pre-commit.ci) check PRs
- Vale checks the prose lines you add, which are the Markdown and the comments and docstrings in Python. See [Prose lint](documentation.md#prose-lint).
- GPU CI runs once a maintainer adds the `gpu` label to the pull request. On a pull request from a fork, each run also waits for approval by a maintainer.
