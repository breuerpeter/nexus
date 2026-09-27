"""The PX4 controller's end of the Hardware In The Loop (HIL) link: it listens on the port the run
gives it, addresses the system id the run gives it, carries the ULog artifact, and owns no peer.
"""

import pytest

from nexus._src.vehicle.controllers.px4 import controller as ctrl


class _FakeMav:
    mav = type("proto", (), {"srcSystem": 0, "srcComponent": 0})()
    target_system = 0
    target_component = 0

    def close(self):
        pass


@pytest.fixture
def bind(monkeypatch):
    """Replace the MAVLink bind with a recorder, so nothing listens on a real port."""
    binds = []

    def connection(conn, **kw):
        binds.append(conn)
        return _FakeMav()

    monkeypatch.setattr(ctrl.mavutil, "mavlink_connection", connection)
    return binds


def test_connect_listens_on_the_runs_hil_port(bind):
    c = ctrl.Px4MavlinkController(port=4563)
    c.connect()
    assert bind == ["tcpin:0.0.0.0:4563"]


def test_connect_addresses_the_runs_px4_system_id(bind):
    """PX4's system id is its instance + 1; the run hands it over, since the controller knows no instance."""
    c = ctrl.Px4MavlinkController(target_system=4)
    c.connect()
    assert c.mav.target_system == 4


def test_artifacts_carry_the_ulog_and_nothing_of_the_peer(tmp_path):
    """The console log is the peer's artifact: the run merges it, so a controller with no peer, an
    external autopilot, contributes what's its own.
    """
    c = ctrl.Px4MavlinkController(ulog_dir=str(tmp_path))
    assert c.artifacts() == {"ulog": None}


def test_close_closes_the_link_and_tolerates_a_closed_one():
    """A daemon that has gone away is the peer's problem; this end closes its socket and the
    orchestrator still flushes the recording.
    """
    closed = []

    class Mav:
        def close(self):
            closed.append("mav")

    c = ctrl.Px4MavlinkController()
    c.mav = Mav()

    c.close()
    c.close()  # a second close finds the same socket and must not raise
    assert closed == ["mav", "mav"]
