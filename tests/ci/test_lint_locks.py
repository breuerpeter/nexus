"""What `lint` does with a stale lock: its pre-commit run on all files fails and shows the lock it rewrote."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The files `lint`'s hooks read for the two projects. Neither holds `.acados/`, as on a CI runner.
_LINTED_FILES = (
    ".pre-commit-config.yaml",
    "_typos.toml",
    "pyproject.toml",
    "uv.lock",
    "nexus-rl/pyproject.toml",
    "nexus-rl/uv.lock",
)

# A package both locks already hold, so a re-lock adds no new download.
_LOCKED_PACKAGE = '"packaging",'


@pytest.fixture
def tree(tmp_path):
    """A git checkout of the two projects with both locks fresh."""
    for name in _LINTED_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    subprocess.run(["git", "init", "-q"], check=True, cwd=tmp_path)
    return tmp_path


def _insert_after(file: Path, line: str, new: str) -> None:
    text = file.read_text()
    assert text.count(f"{line}\n") == 1, f"{file.name} holds no single {line!r} line"
    file.write_text(text.replace(f"{line}\n", f"{line}\n    {new}\n"))


def _lint(tree: Path) -> str:
    """Run the hooks as `lint` runs them, and return what the run prints."""
    subprocess.run(["git", "add", "-A"], check=True, cwd=tree)
    out = subprocess.run(
        ["uvx", "pre-commit", "run", "--all-files", "--show-diff-on-failure"],
        check=False,
        cwd=tree,
        capture_output=True,
        text=True,
    )
    return out.stdout + out.stderr


def test_a_root_group_change_that_stales_the_rl_lock_reds_lint(tree):
    """A pull request that changes the root project's dependencies, extras or groups and leaves
    `nexus-rl/uv.lock` stale goes red on `lint`.
    """
    _insert_after(tree / "pyproject.toml", "[dependency-groups]", f"probe = [{_LOCKED_PACKAGE}]")
    assert "diff --git a/nexus-rl/uv.lock" in _lint(tree)


def test_an_rl_dependency_change_that_stales_the_rl_lock_reds_lint(tree):
    """A pull request that changes `nexus-rl/pyproject.toml` and leaves `nexus-rl/uv.lock` stale goes
    red on `lint`.
    """
    _insert_after(tree / "nexus-rl/pyproject.toml", "dependencies = [", _LOCKED_PACKAGE)
    assert "diff --git a/nexus-rl/uv.lock" in _lint(tree)


def test_a_stale_root_lock_still_reds_lint(tree):
    """A pull request that leaves the root `uv.lock` stale still goes red on `lint`."""
    _insert_after(tree / "pyproject.toml", "dependencies = [", _LOCKED_PACKAGE)
    assert "diff --git a/uv.lock" in _lint(tree)
