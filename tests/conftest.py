"""Shared test helpers."""

import re
import subprocess
import sys

import pytest
import warp as wp

# Warp's default device when the session starts. A test that leaves it changed makes a later test's
# kernels run on another device than its arrays, so test order would matter; the guards below fail it.
_DEVICE = str(wp.get_device())


def _left_changed() -> str | None:
    """The default device if it differs from the session's, which this puts back; else ``None``."""
    device = str(wp.get_device())
    if device == _DEVICE:
        return None
    wp.set_device(_DEVICE)
    return device


@pytest.fixture(autouse=True)
def _warp_device_guard():
    """Fail a test that leaves the Warp default device changed."""
    yield
    if left := _left_changed():
        pytest.fail(f"the test left the Warp default device on {left}, not {_DEVICE}: scope it with wp.ScopedDevice")


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """Fail a test file whose import changes the Warp default device."""
    report = yield
    if isinstance(collector, pytest.Module):
        left = _left_changed()  # put the device back even when the import failed
        if left and report.passed:  # a failed import keeps its own traceback
            report.outcome = "failed"
            report.longrepr = (
                f"{collector.path.name} sets the Warp default device to {left} at import: scope it in its tests"
            )
    return report


@pytest.fixture
def warp_cpu():
    """Run the test with ``cpu`` as the Warp default device, and put the default back after it."""
    with wp.ScopedDevice("cpu"):
        yield


@pytest.fixture(scope="session")
def rrd_entities():
    """Read the entity paths a written ``.rrd`` carries, through the bundled command-line tool.

    rerun 0.34 dropped the local dataframe API, ``rerun.recording.load_recording``; reading an rrd
    now needs the catalog server + the optional datafusion dep. The version-matched command-line
    tool still prints per-chunk entity paths, so tests parse those.
    """

    def _entities(path: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-m", "rerun", "rrd", "print", path], capture_output=True, text=True, check=True
        ).stdout
        return sorted(set(re.findall(r" - (/\S*) - data columns:", out)))

    return _entities
