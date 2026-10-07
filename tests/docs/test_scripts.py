"""Each tracked script has a docs page, and every script lives under ``scripts/``.

The tests read the tracked tree, so they need no GPU and no container.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The camera advertiser, the one ground-station script left, gets no page: #43 deletes it.
UNDOCUMENTED = "scripts/ground/"


def _tracked(*pathspecs: str) -> list[str]:
    """The tracked files that match the pathspecs, as repo-relative paths."""
    out = subprocess.run(
        ["git", "ls-files", "--", *pathspecs], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def _is_entry_point(path: str) -> bool:
    """Whether a reader runs the file: a shell script, or a Python file with a ``__main__`` guard."""
    if path.endswith(".sh"):
        return True
    return path.endswith(".py") and 'if __name__ == "__main__":' in (REPO / path).read_text()


def test_every_tracked_script_has_a_docs_page() -> None:
    """Every tracked script a reader runs has a page under `docs/` that says what the script does
    and how to run it, except the one in `scripts/ground/`, which #43 deletes.
    """
    pages = "\n".join(
        (REPO / p).read_text() for p in _tracked("docs/*.md") if Path(p).name != "CLAUDE.md"
    )
    scripts = [p for p in _tracked("scripts") if _is_entry_point(p) and not p.startswith(UNDOCUMENTED)]
    assert [p for p in scripts if p not in pages] == []


def test_no_tracked_file_sits_under_tools() -> None:
    """No tracked file sits under `tools/`: its scripts and the logo generators live under `scripts/`."""
    assert _tracked("tools") == []
