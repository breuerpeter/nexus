"""The test tree mirrors the source tree: a module's tests sit at the path of the module.

A folder of tests under ``tests/`` sits at the path of a folder under ``nexus/_src/``. A test that
mirrors no source folder sits in a folder named for its kind, one of ``UNMIRRORED``. The tests check
the rule over the tracked tree, so they need no GPU and no container.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The test folders that mirror no folder under ``nexus/_src/``, each named for the kind of its tests.
UNMIRRORED = ("ci", "docs", "examples", "packaging", "suite")


def _tracked_test_modules() -> list[Path]:
    """Every tracked ``test_*.py`` under ``tests/``, as repo-relative paths."""
    out = subprocess.run(
        ["git", "ls-files", "tests"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [Path(f) for f in out if Path(f).match("test_*.py")]


def _unmirrored(folders: list[str]) -> list[str]:
    """The test folders, as repo-relative paths, that match no source folder and no name in ``UNMIRRORED``."""
    paths = [(f, Path(f).relative_to("tests")) for f in folders]
    return [f for f, rel in paths if rel.parts[0] not in UNMIRRORED and not (REPO / "nexus" / "_src" / rel).is_dir()]


def test_every_test_folder_mirrors_a_source_folder() -> None:
    """Every folder under `tests/` that holds a test module sits at the path of a folder under
    `nexus/_src/`, or is one of a stated set that mirrors no source folder: `ci`, `docs`, `examples`,
    `packaging`, `suite`.
    """
    folders = sorted({f.parent.as_posix() for f in _tracked_test_modules() if f.parent != Path("tests")})
    assert _unmirrored(folders) == []


def test_a_test_folder_with_no_source_folder_fails_the_rule_by_name() -> None:
    """A test folder with no source folder fails the rule, and the failure names the folder."""
    assert _unmirrored(["tests/vehicle/sensors", "tests/runtimes"]) == ["tests/runtimes"]


def test_no_test_module_sits_at_the_root() -> None:
    """No test module sits at the root of `tests/`."""
    assert [f.as_posix() for f in _tracked_test_modules() if f.parent == Path("tests")] == []
