"""Kit boots in two places: the programs in the Kit image, ``nexus/_src/rendering/kit-peer/``, and the
three Kit-only asset scripts under ``scripts/assets/``.

The render peer boots Kit to render a run's RTX sensors, and each Kit-only asset script boots it
before its own work and runs from the host with plain Python. A script's traceback must print before
Kit's teardown, which can end the process outright and eat both the traceback and the exit status,
so the boot and that order live only in the files that own them. The test checks the rule over the
tracked tree.

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

# The programs the Kit image runs and the three Kit-only asset scripts boot Kit. Nothing else can.
BOOT_ALLOWED = (
    "nexus/_src/rendering/kit-peer/",
    "scripts/assets/obj_to_usd.py",
    "scripts/assets/site_scan_splat.py",
    "scripts/assets/author_cesium_scene.py",
)


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


def _offenders(hits: list[str]) -> list[str]:
    """The hits outside the files that can boot Kit."""
    return [h for h in hits if not h.startswith(BOOT_ALLOWED)]


def test_only_the_kit_image_programs_and_the_asset_scripts_boot_kit() -> None:
    """Kit boots only in the peer program and the three asset scripts."""
    offenders = _offenders(_hits(BOOT_CALL))
    assert not offenders, (
        f"these boot Kit themselves: {offenders}. Kit boots only in the peer program and the three "
        "Kit-only asset scripts, which own the boot, the traceback and the exit code."
    )


def test_a_fourth_file_with_the_boot_call_fails_the_rule() -> None:
    """Kit boots only in the peer program and the three asset scripts: a fourth file with the boot call fails it."""
    assert _offenders(["scripts/assets/other.py:3", "scripts/assets/obj_to_usd.py:40"]) == ["scripts/assets/other.py:3"]
