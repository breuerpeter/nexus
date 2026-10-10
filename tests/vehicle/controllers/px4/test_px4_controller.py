"""The PX4 controller's end of the Hardware In The Loop (HIL) link: it listens on the port the run
gives it, addresses the system id the run gives it, carries the ULog artifact, owns no peer, and marks each
sensor's fields as updated only on a new sample of it.
"""

import types

import numpy as np
import pytest
import warp as wp

from nexus_sim._src.core.clock import DeviceClock
from nexus_sim._src.core.interfaces import SensorRun
from nexus_sim._src.core.schema import SimTime
from nexus_sim._src.core.signals import wire
from nexus_sim._src.core.stages import Bound
from nexus_sim._src.scene import Site
from nexus_sim._src.vehicle.controllers.px4 import controller as ctrl
from nexus_sim._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor


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


class _Link:
    """The far end of the HIL link, the network boundary: it keeps the arguments of each `HIL_SENSOR`, `HIL_GPS`
    and `HIL_STATE_QUATERNION` the controller sends, by kind, and answers each exchange with one
    `HIL_ACTUATOR_CONTROLS`.
    """

    def __init__(self):
        self.mav = self  # the protocol object the controller sends through
        self.target_system = self.target_component = 0
        self.sent: dict[str, list[tuple]] = {"HIL_SENSOR": [], "HIL_GPS": [], "HIL_STATE_QUATERNION": []}

    def hil_sensor_send(self, *args):
        self.sent["HIL_SENSOR"].append(args)

    def hil_gps_send(self, *args, **kw):
        self.sent["HIL_GPS"].append(args)

    def hil_state_quaternion_send(self, *args):
        self.sent["HIL_STATE_QUATERNION"].append(args)

    def recv_match(self, **kw):
        return types.SimpleNamespace(controls=[0.5] * 16)

    def close(self):
        pass


@pytest.fixture
def link(monkeypatch):
    """Replace the MAVLink bind with a far end that keeps what the controller sends."""
    far = _Link()
    monkeypatch.setattr(ctrl.mavutil, "mavlink_connection", lambda conn, **kw: far)
    return far


# The run's values each sensor here takes: a 4 ms tick and a site at 47.6, -122.3, 5 m.
_RUN = SensorRun(seed=1, dt=0.004, site=Site(lat=47.6, lon=-122.3, alt=5.0, mag_ned=(0.21, 0.05, 0.43)))


class _AtRest:
    """One body at rest, level, 1 m up: the state each sensor here samples."""

    def __init__(self):
        self.body_q = wp.array(np.array([[0, 0, 1, 0, 0, 0, 1]], dtype=np.float32), dtype=wp.transform)
        self.body_qd = wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.spatial_vector)


def _exchange(*sensors, ticks: int) -> None:
    """Wire a PX4 controller, connected, to `sensors` as a run wires them, with the tick's sim time on the device
    in a clock's signal, and run `ticks` ticks of 4 ms from 0 s: on each, every sensor samples a body at rest,
    then the controller exchanges. The tick's time grows by 4 ms a tick in float64, as the device clock adds it.
    """
    clock, controller = DeviceClock(0.004), ctrl.Px4MavlinkController()
    pairs = [(clock, "clock"), *((sensor, "sensor") for sensor in sensors), (controller, "controller")]
    wire([Bound(stage, component, role) for component, role in pairs for stage in component.stages()])
    controller.connect()
    state, now = _AtRest(), 0.0
    for tick in range(ticks):
        clock.time.write([now])
        for sensor in sensors:
            sensor.sample_wp(state, SimTime(now, tick))
        controller.exchange(SimTime(now, tick), timeout=0.1)
        now += 0.004


def test_hil_sensor_marks_a_sensors_fields_updated_only_on_an_exchange_with_a_new_sample_of_it(link, warp_cpu):
    """`HIL_SENSOR` marks a sensor's fields updated only on an exchange that brings a new sample of it.

    Given the controller wired to an Inertial Measurement Unit (IMU) that samples every tick and a barometer at
    125 Hz, and no magnetometer, when four 4 ms ticks each sample the sensors and exchange, then the four
    `HIL_SENSOR` mark the IMU's fields updated on each, and the barometer's pressure fields on the first and
    the third: `0x1E3F, 0x3F, 0x1E3F, 0x3F`.
    """
    _exchange(ImuSensor(_RUN), BaroSensor(_RUN, rate=125.0), ticks=4)

    assert [args[-2] for args in link.sent["HIL_SENSOR"]] == [0x1E3F, 0x3F, 0x1E3F, 0x3F]


def test_hil_gps_and_the_true_state_go_out_with_each_new_gps_sample_and_on_no_other_exchange(link, warp_cpu):
    """`HIL_GPS` and `HIL_STATE_QUATERNION` go out with each new sample of the GPS receiver, and on no other exchange.

    Given the controller wired to a Global Positioning System (GPS) receiver at 125 Hz, when four 4 ms ticks each
    sample it and exchange, then one `HIL_GPS` and one `HIL_STATE_QUATERNION` go out on the first and on the third,
    stamped 0 and 8 ms.
    """
    _exchange(GpsSensor(_RUN, rate=125.0), ticks=4)

    stamps = [[args[0] for args in link.sent[kind]] for kind in ("HIL_GPS", "HIL_STATE_QUATERNION")]
    assert stamps == [[0, 8000], [0, 8000]]


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


def test_the_px4_controller_declares_its_controls_signal_16_wide():
    """PX4's controller declares its controls signal with the 16 channels `HIL_ACTUATOR_CONTROLS` carries.

    Given the controller, when its stages are read, then the one signal they write is `controls` of shape
    `(1, 16)`.
    """
    written = [signal for stage in ctrl.Px4MavlinkController().stages() for signal in stage.writes]

    assert [(signal.name, signal.shape) for signal in written] == [("controls", (1, 16))]
