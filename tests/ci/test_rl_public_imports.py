"""The RL project imports only the public surface, and `lint` fails on an import of `nexus_sim._src` in it."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The files `lint`'s Python hooks read for the RL project's files.
_LINTED_FILES = (".pre-commit-config.yaml", "_typos.toml", "pyproject.toml", "nexus-rl/pyproject.toml")


def _private_imports(path: Path) -> list[int]:
    """The lines of `path` that import `nexus_sim._src` or a module under it, at any depth."""

    def private(module: str) -> bool:
        return module == "nexus_sim._src" or module.startswith("nexus_sim._src.")

    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import) and any(private(alias.name) for alias in node.names):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if private(node.module) or (node.module == "nexus_sim" and any(a.name == "_src" for a in node.names)):
                lines.append(node.lineno)
    return lines


def test_the_rl_project_imports_no_name_from_nexus_sim_src():
    """The RL project imports no name from `nexus_sim._src`, in its package or its scripts.

    Given the `nexus-rl` tree, when CI's import check runs, then it passes, with an import inside a
    function checked too.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "nexus-rl/*.py"], check=True, cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    found = [f"{name}:{line}" for name in tracked for line in _private_imports(ROOT / name)]

    assert found == []


@pytest.fixture
def tree(tmp_path):
    """A git checkout of the files `lint`'s Python hooks read."""
    for name in _LINTED_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    subprocess.run(["git", "init", "-q"], check=True, cwd=tmp_path)
    return tmp_path


def _lint(tree: Path, name: str, module: str, member: str) -> str:
    """Add a file at `name` that imports `member` from `module`, run `lint`'s hooks on it, and return what
    the run prints.
    """
    (tree / name).parent.mkdir(parents=True, exist_ok=True)
    (tree / name).write_text(f"from {module} import {member}\n\nprint({member})\n")
    subprocess.run(["git", "add", "-A"], check=True, cwd=tree)
    out = subprocess.run(
        ["uvx", "pre-commit", "run", "--files", name], check=False, cwd=tree, capture_output=True, text=True
    )
    return out.stdout + out.stderr


def test_a_new_import_of_nexus_sim_src_in_the_rl_package_fails_ci(tree):
    """A new import of `nexus_sim._src` in the RL package fails CI.

    Given a module under `nexus-rl/src/nexus_rl/` that imports `nexus_sim._src.config`, when CI's import
    check runs, then it fails and names the module and the import.
    """
    out = _lint(tree, "nexus-rl/src/nexus_rl/probe.py", "nexus_sim._src.config", "LaunchConfig")

    assert ("`nexus_sim._src` is banned" in out, "nexus-rl/src/nexus_rl/probe.py:1:1" in out) == (True, True)


def test_a_new_import_of_nexus_sim_src_in_an_rl_script_fails_ci_the_same_way(tree):
    """A new import of `nexus_sim._src` in an RL script outside the package fails CI the same way.

    Given a script under `nexus-rl/scripts/` that imports `nexus_sim._src.logging`, when CI's import check
    runs, then it fails and names the script and the import.
    """
    out = _lint(tree, "nexus-rl/scripts/probe.py", "nexus_sim._src.logging", "Logger")

    assert ("`nexus_sim._src` is banned" in out, "nexus-rl/scripts/probe.py:1:1" in out) == (True, True)
