"""The `acados_nmpc` example on a machine that lacks acados: it names each missing part and its command."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest


def _run_example(tmp_path) -> subprocess.CompletedProcess:
    """Run the example as a user does, with no acados install where it looks and no CUDA device."""
    env = {**os.environ, "ACADOS_SOURCE_DIR": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "nexus.examples", "acados_nmpc"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_with_no_acados_install_the_example_says_what_to_run_and_builds_no_sim(tmp_path):
    """With no acados install, the example says what to run and builds no sim: given
    `ACADOS_SOURCE_DIR` set to an empty directory and no CUDA device, when
    `python -m nexus.examples acados_nmpc` runs, then it exits non-zero with a message that names the
    provision command, and no `ModuleNotFoundError` traceback.
    """
    run = _run_example(tmp_path)

    said = run.stdout + run.stderr
    assert (
        run.returncode != 0,
        "-m nexus.examples acados_nmpc --provision" in said,
        "ModuleNotFoundError" in said,
    ) == (True, True, False)


@pytest.mark.skipif(importlib.util.find_spec("casadi") is not None, reason="this environment has the acados extra")
def test_with_the_acados_extra_not_installed_the_example_names_the_extra(tmp_path):
    """With the `acados` extra not installed, the example names the extra: given an environment
    without `casadi`, when `python -m nexus.examples acados_nmpc` runs, then it exits non-zero with a
    message that names `nexus-sim[acados]`.
    """
    run = _run_example(tmp_path)

    assert (run.returncode != 0, "nexus-sim[acados]" in run.stdout + run.stderr) == (True, True)
