"""What PX4 Software In The Loop (SITL) logs at boot when it dials in late to the sim's
Hardware In The Loop (HIL) server, as it can on a slow CI runner.

The module fixture flies one run through ``na.Sim``, which builds and launches PX4 from ``$PX4_DIR``, so
the test needs docker and a PX4 checkout carrying the airframe the sim flies; it skips without one.
"""

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does: whichever import runs
# first sets it, and the controller's HIL_GPS needs the common dialect's fields.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus as na
from nexus._src.vehicle.controllers.px4.sitl import px4_dir

_AIRFRAMES = px4_dir() / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes"
if not list(_AIRFRAMES.glob("*_none_astro_max")):
    pytest.skip(
        f"needs docker and a PX4 checkout at $PX4_DIR ({px4_dir()}) carrying the none_astro_max airframe",
        allow_module_level=True,
    )

_DIAL_IN_S = 30.0  # how long after the sim starts waiting PX4's connection reaches it


@pytest.fixture(scope="module")
def px4_log_of_a_late_dial_in():
    """The PX4 console log of a run whose PX4 dials in 30 s after the sim starts waiting for it, flown
    until PX4 arms.

    The sim's HIL server takes no connection until 30 s after it first looks for one, the same state a
    slow PX4 boot leaves the sim in.
    """
    recv = mavutil.mavtcpin.recv
    first_look = None

    def recv_once_px4_dialed_in(self, n=None):
        nonlocal first_look
        if self.port is None:
            first_look = first_look or time.monotonic()
            if time.monotonic() - first_look < _DIAL_IN_S:
                return ""
        return recv(self, n)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mavutil.mavtcpin, "recv", recv_once_px4_dialed_in)
        with na.Sim(control="px4-sitl") as sim:
            sim.start(timeout=120.0)
            op = sim.operator
            op.arm()
            sim.wait_until(op.is_armed, sim_timeout=120.0)
    return Path(sim.artifacts()["px4_log"]).read_text(errors="ignore")


def test_a_px4_that_dials_in_late_logs_no_ekf2_missing_data(px4_log_of_a_late_dial_in):
    """A PX4 SITL that dials in late logs no ``ekf2 missing data`` line.

    PX4 on the pinned tree dials in 30 s after the sim starts waiting for it, and once PX4 is ready for
    takeoff, its log has no ``Preflight Fail: ekf2 missing data`` line.
    """
    assert "Preflight Fail: ekf2 missing data" not in px4_log_of_a_late_dial_in
