"""Where PX4 Software In The Loop (SITL) starts its clock when it dials in late to the sim's
Hardware In The Loop (HIL) server, as it can on a slow CI runner.

The module fixture flies one run through ``na.Sim``, which builds and launches PX4 from ``$PX4_DIR``, so
the test needs docker and a PX4 checkout carrying the airframe the sim flies; it skips without one.
"""

import os
import time

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

_DIAL_IN_S = 17.0  # how long after the sim starts waiting PX4's connection reaches it, as on a slow CI runner


@pytest.fixture(scope="module")
def first_sensor_stamp_of_a_late_dial_in():
    """The ``time_usec`` of the first ``HIL_SENSOR`` a run sends PX4, when PX4 dials in 17 s after the
    sim starts waiting for it.

    The sim's HIL server takes no connection until 17 s after it first looks for one, the same state a
    slow PX4 boot leaves the sim in. The fixture reads the stamp off the bytes the sim writes to the link.
    """
    recv, write = mavutil.mavtcpin.recv, mavutil.mavtcpin.write
    parser = mavutil.mavlink.MAVLink(None)
    first_look = None
    stamps = []

    def recv_once_px4_dialed_in(self, n=None):
        nonlocal first_look
        if self.port is None:
            first_look = first_look or time.monotonic()
            if time.monotonic() - first_look < _DIAL_IN_S:
                return ""
        return recv(self, n)

    def write_and_read_stamps(self, buf):
        if self.port is not None:  # before PX4 dials in, a write reaches nobody
            stamps.extend(m.time_usec for m in parser.parse_buffer(bytes(buf)) or [] if m.get_type() == "HIL_SENSOR")
        return write(self, buf)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mavutil.mavtcpin, "recv", recv_once_px4_dialed_in)
        mp.setattr(mavutil.mavtcpin, "write", write_and_read_stamps)
        with na.Sim(control="px4-sitl") as sim:
            sim.start(timeout=120.0)
    return stamps[0]


def test_px4s_clock_starts_near_zero_even_when_px4_dials_in_late(first_sensor_stamp_of_a_late_dial_in):
    """PX4's clock starts near zero even when PX4 dials in late.

    PX4 on the pinned tree dials in 17 s after the sim starts waiting for it, and the first ``HIL_SENSOR``
    the sim sends it carries a ``time_usec`` under 10 ms.
    """
    assert first_sensor_stamp_of_a_late_dial_in < 10_000
