"""Kit boots in one place: the programs in the Kit image, ``nexus/_src/rendering/kit-peer/``.

The render peer boots Kit to render a run's RTX sensors, and the script launcher boots it before it
runs a Kit-only asset script through ``nexus script``. The launcher prints a failed script's
traceback before Kit's teardown, which can end the process outright and eat both the traceback and
the exit status, so a script that boots its own Kit app loses that. The test checks the rule over
the tracked tree.

The code spells the needle in pieces below, and names rather than quotes it in the prose here, so
this file isn't itself a hit: the guard would fail on its own text, and so would the greps a reader
runs by hand.

Needs no Kit, no GPU and no container, so it runs in the ordinary test job.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = {".py", ".md", ".sh", ".yml", ".yaml", ".toml", ".txt"}

BOOT_CALL = "SimulationApp" + "("  # the *call*, so a prose mention of the class isn't a hit

# The programs the Kit image runs boot Kit. Nothing else can.
BOOT_ALLOWED = ("nexus/_src/rendering/kit-peer/",)


def _tracked_text_files() -> list[Path]:
    """Every tracked file with a scanned text suffix, as repo-relative paths."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    return [Path(f) for f in out if Path(f).suffix in SCANNED_SUFFIXES]


def _hits(needle: str) -> list[str]:
    """Repo-relative ``path:line`` for every tracked file containing ``needle``."""
    found = []
    for rel in _tracked_text_files():
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if needle in line:
                found.append(f"{rel}:{n}")
    return found


def test_only_the_kit_image_programs_boot_kit() -> None:
    """The Kit boot call can appear only in the programs the Kit image runs."""
    offenders = [h for h in _hits(BOOT_CALL) if not h.startswith(BOOT_ALLOWED)]
    assert not offenders, (
        f"these boot Kit themselves: {offenders}. Drop the boot and run the file with "
        "`uv run nexus script <path> [args…]`, which owns the boot, the traceback and the exit code."
    )
