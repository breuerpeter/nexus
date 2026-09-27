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


def test_a_failing_script_exits_non_zero_when_kit_teardown_would_end_the_process_with_zero(tmp_path):
    """A failing script shows its traceback and its exit code reaches the shell.

    Kit's teardown ends the process with status 0 whatever the script's code was, so `finish`
    has to make the code reach the shell before it. The stand-in app's `close` ends the process
    with 0 the way Kit's does.
    """
    code = f"""
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("kit_container", {str(ROOT / "scripts" / "assets" / "kit_container.py")!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class App:
    def close(self):
        os._exit(0)

m.finish(App(), lambda argv: 1 / 0, [])
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert (r.returncode, "ZeroDivisionError" in r.stderr) == (1, True), (r.returncode, r.stderr)
