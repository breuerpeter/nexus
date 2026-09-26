"""The Kit-only asset scripts run from the host with plain Python and start the Kit container themselves.

Each test runs a script in a child process from a folder outside the checkout, with ``DOCKER_HOST``
pointing at a socket that doesn't exist, so no container starts and the tests need no GPU.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_with_docker_unreachable_a_script_fails_with_a_message_that_names_docker(tmp_path):
    """With Docker unreachable, a script fails with a message that names Docker and no traceback.

    Given `DOCKER_HOST` pointing nowhere, when `uv run python scripts/assets/obj_to_usd.py <dir> --out
    x.usdz` runs, then it exits non-zero and its last line names Docker.
    """
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "site.obj").write_text("v 0 0 0\n")
    env = {
        **os.environ,
        "DOCKER_HOST": f"unix://{tmp_path / 'no-daemon.sock'}",
        "HOME": str(tmp_path / "home"),
    }

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "assets" / "obj_to_usd.py"),
            str(scan),
            "--out",
            str(tmp_path / "x.usdz"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    assert (r.returncode != 0, "Traceback" in r.stderr, "Docker" in last) == (True, False, True), r.stderr
