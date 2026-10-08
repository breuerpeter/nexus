"""The suite passes in any order: no test leaves the Warp default device changed for a later one.

Each test runs pytest in a subprocess, since the order bug is a segfault that would end this run.
The tests are `gpu`: without a CUDA device ``cpu`` is the only device, and the default can't change.
A run on the GPU machine fails the same tests as the same run with no CUDA device: a test of an issue
that the code doesn't meet yet fails both, and the device changes no outcome.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parents[2]


def _pytest(*args: str, cuda: bool = True) -> subprocess.CompletedProcess:
    """Run pytest from the repo root in a fresh process, whose Warp default device is ``cuda:0``, or ``cpu``
    with ``cuda`` false, which hides every CUDA device from it.
    """
    env = None if cuda else {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-vv", "-rfE", "-p", "no:cacheprovider", *args],
        check=False,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


def _failures(run: subprocess.CompletedProcess) -> list[str]:
    """The entries of the run's short test summary, one per failed test or file, each with its whole message."""
    summary = run.stdout.partition("short test summary info")[2]
    return re.findall(r"^(?:FAILED|ERROR) .*?(?=\n(?:FAILED|ERROR) |\n=|\Z)", summary, re.M | re.S)


def _outcome(run: subprocess.CompletedProcess) -> tuple[int, list[str]]:
    """The run's exit code and the test or file each entry of its short test summary names, sorted."""
    return run.returncode, sorted(entry.split()[1] for entry in _failures(run))


def test_api_passes_before_sensors():
    """`pytest tests/api tests/vehicle/sensors/test_sensors.py` passes on a GPU machine as it does with no CUDA device."""
    args = ("tests/api", "tests/vehicle/sensors/test_sensors.py")
    run = _pytest(*args)
    assert _outcome(run) == _outcome(_pytest(*args, cuda=False)), run.stdout[-3000:] + run.stderr[-3000:]


def test_api_passes_alone():
    """`pytest tests/api` alone still passes on a GPU machine as it does with no CUDA device."""
    run = _pytest("tests/api")
    assert _outcome(run) == _outcome(_pytest("tests/api", cuda=False)), run.stdout[-3000:] + run.stderr[-3000:]


def test_a_test_that_leaves_the_device_changed_fails_naming_the_device(tmp_path):
    """A test that leaves the Warp default device changed fails, and the failure names the device it left."""
    scratch = tmp_path / "test_leak.py"
    scratch.write_text('import warp as wp\n\n\ndef test_sets_the_device():\n    wp.set_device("cpu")\n')
    run = _pytest("-p", "tests.conftest", str(scratch))
    messages = [f.partition(" - ")[2] for f in _failures(run) if "::test_sets_the_device" in f]
    assert any("cpu" in m for m in messages), run.stdout[-3000:]


def test_a_file_that_sets_the_device_at_import_fails_the_run_naming_the_file(tmp_path):
    """A test file that sets the Warp default device at import fails the run, and the failure names the file."""
    scratch = tmp_path / "test_import_leak.py"
    scratch.write_text('import warp as wp\n\nwp.set_device("cpu")\n\n\ndef test_passes():\n    pass\n')
    run = _pytest("-p", "tests.conftest", str(scratch))
    assert any("test_import_leak.py" in f for f in _failures(run)), run.stdout[-3000:]
