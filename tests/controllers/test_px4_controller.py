"""The PX4 controller's ownership of the Software In The Loop (SITL) peer: that it launches *after*
the Hardware In The Loop (HIL) bind, stops before the rest of the teardown, carries both artifacts,
and passes the registry airframe through with PX4's ``none_`` prefix.
"""

import pytest

from nexus._src.vehicle.controllers.px4 import controller as ctrl


@pytest.fixture
def peer(monkeypatch):
    """Replace ``Px4Sitl`` with a recorder, so the controller's lifecycle is observable and nothing
    reaches the docker daemon.
    """
    events = []

    class FakeSitl:
        def __init__(self, *, container, log_path, airframe="none_astro_max"):
            self.container = container
            self.log_path = log_path
            self.airframe = airframe
            events.append(("init", airframe))

        def launch(self):
            events.append(("launch", self.airframe))

        def stop(self):
            events.append(("stop", self.airframe))

    monkeypatch.setattr(ctrl, "Px4Sitl", FakeSitl)
    monkeypatch.setattr(ctrl, "build_px4_sitl", lambda: events.append(("build", None)))
    return events


def test_the_registry_airframe_reaches_the_launcher_with_the_none_prefix(peer):
    """The registry holds the make target; `none_` is PX4's "no simulator" prefix for external/HIL, and
    the controller prepends it here, so the registry row stays free of PX4 spelling.
    """
    ctrl.Px4MavlinkController(airframe="astro_max_blue")
    assert peer == [("init", "none_astro_max_blue")]


def test_prepare_builds_px4(peer):
    c = ctrl.Px4MavlinkController()
    c.prepare()
    assert ("build", None) in peer


def test_connect_launches_px4_after_the_bind(peer, monkeypatch):
    """The ordering holds by construction, not by a margin: ``tcpin`` binds and listens in the
    constructor, with a lazy accept, so :4560 is up before the controller asks PX4 to dial in.
    """
    order = []
    monkeypatch.setattr(ctrl.mavutil, "mavlink_connection", lambda *a, **kw: order.append("bind") or _FakeMav())

    c = ctrl.Px4MavlinkController()
    c.connect()
    assert order == ["bind"]
    assert [e for e in peer if e[0] == "launch"], "connect() must launch the peer"
    assert peer.index(("launch", "none_astro_max")) == len(peer) - 1  # the last thing connect does


def test_close_stops_px4_first(peer):
    c = ctrl.Px4MavlinkController()
    c.close()
    assert peer[-1] == ("stop", "none_astro_max")


def test_artifacts_carry_both_the_ulog_and_the_px4_console(peer, tmp_path):
    c = ctrl.Px4MavlinkController(ulog_dir=str(tmp_path))
    arts = c.artifacts()
    assert set(arts) == {"ulog", "px4_log"}
    assert arts["px4_log"].endswith(".log")


class _FakeMav:
    mav = type("proto", (), {"srcSystem": 0, "srcComponent": 0})()
    target_system = 0
    target_component = 0

    def close(self):
        pass


def test_a_failing_container_teardown_does_not_cost_the_rest_of_close(peer, monkeypatch):
    """A daemon that has gone away must not end the teardown early: the MAVLink socket still has to
    close, and the orchestrator still has to flush the recording, which is exactly what a failing run
    needs kept.
    """
    closed = []

    class Boom:
        log_path = "/tmp/px4.log"
        airframe = "none_astro_max"

        def stop(self):
            raise RuntimeError("docker daemon went away")

    class Mav:
        def close(self):
            closed.append("mav")

    c = ctrl.Px4MavlinkController()
    c._sitl = Boom()
    c.mav = Mav()

    c.close()  # must not raise
    assert closed == ["mav"], "the MAVLink socket was skipped by the container failure"
