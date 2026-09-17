"""The Kit container has *one* door: `nexus script`, reached through the entrypoint.

Two rules, enumerated over the tracked tree. Both exist because a script that boots its own Kit app
behind an entrypoint override silently loses what the container entrypoint does for every other
invocation: the dependency sync from ``pyproject.toml``, into Kit's ephemeral site-packages, so it
must run at every container start; the ``NEXUS_PINS_DIR`` physics pins from ``uv.lock``; and
``PXR_PLUGINPATH_NAME`` for the Cesium Universal Scene Description (USD) schemas.

The code spells the two needles in pieces below, and names rather than quotes them in the prose here,
so this file isn't itself a hit: the guard would fail on its own text, and so would the greps a reader runs
by hand.

Needs no Kit, no GPU and no container, so it runs in the ordinary test job.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = {".py", ".md", ".sh", ".yml", ".yaml", ".toml", ".txt"}

BOOT_CALL = "SimulationApp" + "("  # the *call*, so a prose mention of the class isn't a hit
ENTRYPOINT_FLAG = "--" + "entrypoint"

# The command-line tool is the boot, and the isaacsim runtime is what it boots into. Nothing else can.
BOOT_ALLOWED = ("nexus/_src/",)


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


def test_only_the_cli_boots_kit() -> None:
    """The Kit boot call can appear only under nexus/_src: everything else goes through
    ``nexus script``.
    """
    offenders = [h for h in _hits(BOOT_CALL) if not h.startswith(BOOT_ALLOWED)]
    assert not offenders, (
        f"these boot Kit themselves, which skips the container entrypoint: {offenders}. "
        "Drop the boot and run the file with `uv run nexus script <path> [args…]`, "
        "which owns the boot, the pins, the traceback and the exit code."
    )


def test_no_documented_entrypoint_override() -> None:
    """The entrypoint-override flag must appear in no tracked file: it drops the dep sync, the
    physics pins and PXR_PLUGINPATH_NAME.
    """
    offenders = _hits(ENTRYPOINT_FLAG)
    assert not offenders, (
        f"these override the container entrypoint: {offenders}. "
        "The container has one door: `docker compose run --rm isaacsim <cli args>`, and "
        "`uv run nexus script <path> [args…]` is how a script goes through it."
    )
