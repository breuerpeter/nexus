"""A `gpu` marker labels the tests that need a CUDA device, and the suite's hook decides when they run.

On a box with no CUDA device a `gpu` test skips, and with `--require-cuda`, the GPU leg's setting, a
`gpu` test that doesn't run fails. Each test runs pytest in a subprocess on a scratch test file, with
the repo's pytest settings and ``tests/conftest.py`` loaded as a plugin.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _pytest(test: Path, *args: str, cuda: bool = True) -> subprocess.CompletedProcess:
    """Run pytest on ``test`` in a fresh process under the repo's settings; ``cuda=False`` hides every CUDA device."""
    env = dict(os.environ)
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_ROOT / "pyproject.toml"),
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-rA",
            str(test),
            *args,
        ],
        check=False,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _summary(run: subprocess.CompletedProcess, outcome: str) -> list[str]:
    """The entries of the run's short test summary with ``outcome``, such as ``FAILED`` or ``SKIPPED``."""
    summary = run.stdout.partition("short test summary info")[2]
    return re.findall(rf"^{outcome} .*$", summary, re.M)


def _scratch(tmp_path: Path, body: str) -> Path:
    """Write a test file holding ``body`` under ``tmp_path``."""
    path = tmp_path / "test_scratch.py"
    path.write_text("import pytest\nimport warp as wp\n\n\n" + body)
    return path


def test_a_misspelled_marker_fails_the_run(tmp_path):
    """A misspelled marker fails the run: given a test marked `gpuu`, when pytest collects it, then
    collection fails and names the marker.
    """
    test = _scratch(tmp_path, "@pytest.mark.gpuu\ndef test_runs():\n    pass\n")
    run = _pytest(test)
    assert any("gpuu" in e for e in _summary(run, "ERROR")), run.stdout[-3000:] + run.stderr[-3000:]


def test_on_a_box_with_no_cuda_device_a_gpu_test_skips(tmp_path):
    """On a box with no CUDA device, a `gpu` test skips: given CUDA hidden from the process and a `gpu`
    test that uses `cuda:0` with no skip of its own, when pytest runs it, then it reports skipped with the
    reason "no CUDA device".
    """
    test = _scratch(tmp_path, '@pytest.mark.gpu\ndef test_uses_cuda():\n    wp.zeros(1, device="cuda:0")\n')
    run = _pytest(test, cuda=False)
    assert any("no CUDA device" in e for e in _summary(run, "SKIPPED")), run.stdout[-3000:] + run.stderr[-3000:]


def test_on_the_gpu_leg_a_gpu_test_that_finds_no_cuda_device_fails(tmp_path):
    """On the GPU leg, a `gpu` test that finds no CUDA device fails: given the GPU leg's settings and CUDA
    hidden from the process, when pytest runs a `gpu` test, then it fails and names the missing device.
    """
    test = _scratch(tmp_path, '@pytest.mark.gpu\ndef test_uses_cuda():\n    wp.zeros(1, device="cuda:0")\n')
    run = _pytest(test, "--require-cuda", cuda=False)
    failed = [e for e in _summary(run, "FAILED") if "::test_uses_cuda" in e]
    assert any("no CUDA device" in e for e in failed), run.stdout[-3000:] + run.stderr[-3000:]


@pytest.mark.gpu
def test_on_the_gpu_leg_a_gpu_test_that_skips_for_its_own_reason_fails(tmp_path):
    """On the GPU leg, a `gpu` test that skips for its own reason fails: given the GPU leg's settings and a
    `gpu` test that skips, as the golden trajectory test does on a GPU model with no recorded digest, when
    pytest runs it, then it fails with the skip's reason.
    """
    reason = "the trajectory was never recorded on this GPU"
    test = _scratch(tmp_path, f'@pytest.mark.gpu\ndef test_skips():\n    pytest.skip("{reason}")\n')
    run = _pytest(test, "--require-cuda")
    failed = [e for e in _summary(run, "FAILED") if "::test_skips" in e]
    assert any(reason in e for e in failed), run.stdout[-3000:] + run.stderr[-3000:]


def test_a_test_that_skips_for_a_missing_cuda_device_without_the_gpu_marker_fails(tmp_path):
    """A test that skips for a missing CUDA device without the `gpu` marker fails: given CUDA hidden from
    the process and a test with no `gpu` marker that skips for no CUDA device, when pytest runs it, then it
    fails and names the `gpu` marker.
    """
    body = 'def test_needs_cuda():\n    if not wp.is_cuda_available():\n        pytest.skip("no CUDA device")\n'
    run = _pytest(_scratch(tmp_path, body), cuda=False)
    failed = [e for e in _summary(run, "FAILED") if "::test_needs_cuda" in e]
    assert any("gpu" in e.partition(" - ")[2] for e in failed), run.stdout[-3000:] + run.stderr[-3000:]
