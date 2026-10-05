"""The test and lint tools come from the lock: no command fetches one at its newest release.

The tests read the tracked tree and ``uv.lock``, so they need no GPU and no container.
"""

import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_the_lock_pins_the_test_and_lint_tools() -> None:
    """The lock pins pytest, pytest-cov and import-linter."""
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    names = [p["name"] for p in lock["package"]]
    assert {tool: names.count(tool) for tool in ("pytest", "pytest-cov", "import-linter")} == {
        "pytest": 1,
        "pytest-cov": 1,
        "import-linter": 1,
    }


def test_no_tracked_file_names_a_tool_outside_the_lock() -> None:
    """No tracked file names a test or lint tool outside the lock."""
    out = subprocess.run(
        [
            "git", "grep", "-I", "-n", "-F",
            "-e", "--with pytest", "-e", "--with pytest-cov", "-e", "uvx --from import-linter",
            "--", ".github/workflows", "docs", "scripts", "*CLAUDE.md",
        ],
        check=False,  # git grep exits 1 on no match
        cwd=REPO,
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.stdout.splitlines() == []
