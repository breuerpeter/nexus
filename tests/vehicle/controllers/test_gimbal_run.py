"""A vehicle's gimbal controller holds its camera's attitude in a run, through its three joint motors.

Each test builds the fixture quad of ``tests/vehicle/quad.py`` with its camera gimbal: a camera body behind a pan,
a roll and a tilt joint, its center of mass 2 cm off each joint's axis, a torque motor on each joint that reads
the gimbal controller's controls, and a first controller, a stand-in at fixed rotor commands, that flies the
airframe. A flight starts in free flight 2 m up, so the camera starts level and at rest. The world's x axis
points North and its z axis up, and a body's frame has x forward and z down. Skipped without newton or pxr.

The gimbal's controller is a MAVLink gimbal device too, on port 13280 of the User Datagram Protocol (UDP) plus the run's instance, so every run here
binds that port, and a stand-in gimbal manager, a socket of the test's own, talks to it.
"""

import contextlib
import itertools
import logging
import math
import socket
from typing import ClassVar

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import numpy as np
import warp as wp
from pymavlink.dialects.v20 import common as mavlink

from nexus_uav._src.api.sim import Sim
from nexus_uav._src.core.errors import RunError
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.vehicle.controllers.gimbal import GimbalController
from nexus_uav._src.vehicle.controllers.gimbal_device import PORT
from tests.usd import sensor_vehicle as sv
from tests.vehicle import quad

# Every run binds the device's port, which is the machine's, so the tests run on one worker, one at a time.
pytestmark = [pytest.mark.usefixtures("warp_cpu"), pytest.mark.xdist_group("px4_ports")]

FULL = (1.0, 1.0, 1.0, 1.0)  # the airframe climbs and turns nothing
TIP = (1.0, 0.975, 1.0, 0.975)  # more thrust in front: the airframe pitches up
TURN = (1.0, 1.0, 0.935, 0.935)  # more thrust on one diagonal: the airframe's heading turns
FLIGHT = 125  # ticks, 0.5 s
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def _commanding(values) -> type:
    """A stand-in first controller class that sends `values` to the rotors every tick and counts its ticks in the
    class's `ticks`. Each call makes a class of its own, for one run.
    """

    class Commanding(quad.Commands):
        ticks = 0

        def __init__(self, **kwargs):
            super().__init__(values)  # the schema's keyword arguments, which it doesn't read

        def exchange(self, t, timeout=None):
            type(self).ticks += 1
            return super().exchange(t, timeout)

    return Commanding


def _reachable() -> type:
    """The gimbal controller's class, which keeps each instance a run builds in the class's `made`, so a test
    writes its setpoint as a host stage would. Each call makes a class of its own, for one run.
    """

    class Reachable(GimbalController):
        made: ClassVar[list] = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            type(self).made.append(self)

    return Reachable


def _forward(orientation) -> np.ndarray:
    """The world direction of a body's x axis, its view, from its recorded orientation, `(x, y, z, w)`."""
    x, y, z, w = (float(v) for v in orientation)
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])


def _heading(orientation) -> float:
    """A body's heading [degrees], the direction of its x axis over the ground."""
    forward = _forward(orientation)
    return math.degrees(math.atan2(forward[1], forward[0]))


def _apart(a: float, b: float) -> float:
    """How far two headings are apart [degrees], the short way round."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _down(orientation) -> float:
    """How far a body's x axis points below the horizon [degrees]."""
    return math.degrees(math.asin(-_forward(orientation)[2]))


def _fly(tmp_path, commands, ticks: int = FLIGHT, *, device: str = "cpu", write=None) -> dict:
    """Fly the fixture quad with its camera gimbal for `ticks` ticks on `device`, its first controller at
    `commands`. `write`, a tick and a setpoint, writes that setpoint to the gimbal's controller after that many
    ticks. Returns the camera's and the airframe's recorded orientations, one row per tick, and the error the
    run raised.
    """
    path = quad.declare_gimbal(quad.author(tmp_path / "quad.usda", camera=True, diagonal=True))
    controller = _reachable()
    registry = Registry({"NexusPx4API_1": _commanding(commands), "NexusGimbalControllerAPI": controller})
    try:
        with wp.ScopedDevice("cpu" if device == "cpu" else "cuda:0"):
            loop = sv.build(path, components=registry, device=device, fall_from=(0.0, 0.0, 2.0))
            with Sim._from_loop(loop) as sim:
                for tick in range(ticks):
                    if write is not None and tick == write[0]:
                        controller.made[0].write(*write[1])
                    if not sim.step():
                        break
                camera = np.array(sim["vehicle/body/camera"].history_arrays()["orientation"])
                airframe = np.array(sim["vehicle/body/body"].history_arrays()["orientation"])
    except Exception as exc:
        return {"camera": None, "airframe": None, "error": exc}
    return {"camera": camera, "airframe": airframe, "error": None}


def test_with_nothing_written_the_camera_stays_level_while_the_airframe_tips_under_it(tmp_path):
    """With nothing written to its setpoint, the camera stays level while the airframe tips under it.

    Given the fixture quad with a camera body behind three revolute joints, its center of mass 2 cm off the
    joints' axes, a torque motor on each joint connected to the gimbal's controller by `nexus:inputs:controls`,
    and a first controller whose unequal commands tip the airframe more than 20° in 125 ticks, when the run flies,
    then the camera's axis stays within 2° of level on every tick.
    """
    flown = _fly(tmp_path, TIP)

    tipped = flown["airframe"] is not None and quad.tilt(flown["airframe"][-1]) > 20.0
    level = flown["camera"] is not None and max(quad.tilt(q) for q in flown["camera"]) < 2.0
    assert (tipped, level) == (True, True), flown["error"]


def test_with_nothing_written_the_cameras_heading_follows_the_airframes(tmp_path):
    """With nothing written to its setpoint, the camera's heading follows the airframe's.

    Given the same vehicle and a first controller whose commands turn the airframe's heading more than 30° in 125
    ticks, when the run flies, then the camera's heading stays within 5° of the airframe's on every tick.
    """
    flown = _fly(tmp_path, TURN)

    turned, followed = False, False
    if flown["error"] is None:
        headings = [
            (_heading(camera), _heading(airframe))
            for camera, airframe in zip(*map(flown.get, ("camera", "airframe")), strict=True)
        ]
        turned = _apart(headings[-1][1], headings[0][1]) > 30.0
        followed = max(_apart(camera, airframe) for camera, airframe in headings) < 5.0
    assert (turned, followed) == (True, True), flown["error"]


@pytest.mark.parametrize("device", DEVICES)
def test_the_camera_turns_to_a_written_attitude_and_a_setpoint_written_during_a_run_takes_effect(tmp_path, device):
    """The camera turns to a written attitude, and a setpoint written during a run takes effect.

    Given the same vehicle, its airframe climbing straight, and an attitude of 30° down, with yaw following the
    vehicle, written after tick 25, when the run flies 150 ticks, then the camera stays level before the write,
    never passes 33° down, and is within 1° of 30° from tick 125 on. Once on the CPU device and once, marked
    `gpu`, on a CUDA device, where the loop captures the graph before the write.
    """
    half = math.radians(-30.0) / 2.0  # a pitch of −30° turns the view down
    setpoint = ([(0.0, math.sin(half), 0.0, math.cos(half))], [(0.0, 0.0, 0.0)], [False])

    flown = _fly(tmp_path, FULL, 150, device=device, write=(25, setpoint))

    before, most, settled = False, False, False
    if flown["error"] is None:
        down = [_down(q) for q in flown["camera"]]
        before = max(abs(d) for d in down[:25]) < 1.0
        most = max(down) < 33.0
        settled = max(abs(d - 30.0) for d in down[125:]) < 1.0
    assert (before, most, settled) == (True, True, True), flown["error"]


@pytest.mark.parametrize(
    ("case", "named"),
    [
        ("no camera", ("nexus:inputs:camera",)),
        ("two joints", ("nexus:inputs:joints", "three")),
        ("a prismatic joint", ("nexus:inputs:joints", quad.CAMERA_JOINTS[1])),
        ("a joint off the chain", ("nexus:inputs:joints", quad.GIMBAL_JOINT)),
    ],
)
def test_a_declaration_that_is_no_camera_behind_three_revolute_joints_fails_the_build(tmp_path, case, named):
    """A controller whose declaration isn't a camera body behind a chain of three revolute joints fails the build
    and names its prim.

    Given the fixture with, in turn, no camera connected, two joints connected, a prismatic joint among the three,
    and a joint off the chain to the camera's body, the airframe's other gimbal joint, in the tilt joint's place,
    when the vehicle builds, then it raises `ValueError` before any stage runs, and the message names the
    controller's prim and the input, and with it the fault: that the joints number three, or the one joint the
    chain rejects.
    """
    drawn = quad.author(
        tmp_path / "quad.usda", camera=True, gimbal=True, prismatic=1 if case == "a prismatic joint" else None
    )
    declared = {
        "no camera": {"camera": None},
        "two joints": {"joints": quad.CAMERA_JOINTS[:2]},
        "a prismatic joint": {},
        "a joint off the chain": {"joints": (*quad.CAMERA_JOINTS[:2], quad.GIMBAL_JOINT)},
    }[case]
    path = quad.declare_gimbal(drawn, motors=False, **declared)
    autopilot = _commanding(FULL)

    error = None
    try:
        sv.steps(sv.build(path, components=Registry({"NexusPx4API_1": autopilot}), fall_from=(0.0, 0.0, 2.0)), 1)
    except Exception as exc:
        error = exc

    names = [s in str(error) for s in (f"{quad.GIMBAL_CONTROLLER}:", *named)]
    assert (type(error), all(names), autopilot.ticks) == (ValueError, True, 0), error


SYSTEM = 3  # the stand-in manager's MAVLink system
GIMBAL = 154  # the component id of a gimbal device, MAV_COMP_ID_GIMBAL
IN_VEHICLE = 32  # GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME: the yaw is against the vehicle's heading
DOWN_30_WXYZ = (0.9659258, 0.0, -0.2588190, 0.0)  # a pitch of −30°, as MAVLink orders a quaternion


class _Manager:
    """A stand-in gimbal manager: a socket of its own that speaks MAVLink, as the autopilot of system 3, to the
    device of the run at `instance`.
    """

    def __init__(self, instance: int = 0):
        self._device = ("127.0.0.1", PORT + instance)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.setblocking(False)
        self._mav = mavlink.MAVLink(None, srcSystem=SYSTEM, srcComponent=1)

    def send(self, message) -> None:
        """Send `message` to the device's port."""
        self._socket.sendto(message.pack(self._mav), self._device)

    def heartbeat(self) -> None:
        """Send the autopilot's `HEARTBEAT`."""
        self.send(mavlink.MAVLink_heartbeat_message(mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_PX4, 0, 0, 0, 3))

    def heard(self) -> list:
        """Every message the socket has received since the last call, in order."""
        messages = []
        while True:
            try:
                datagram = self._socket.recv(65535)
            except BlockingIOError:
                return messages
            messages += mavlink.MAVLink(None).parse_buffer(datagram) or []

    def close(self) -> None:
        self._socket.close()


@pytest.fixture
def manager():
    """Makes a stand-in manager for the run at an instance, and closes each one's socket after the test."""
    made = []

    def make(instance: int = 0) -> _Manager:
        made.append(_Manager(instance))
        return made[-1]

    yield make
    for one in made:
        one.close()


def _at(instance: int) -> type:
    """The gimbal controller's class for a run at `instance`, the number a run takes when it starts a peer."""

    class At(GimbalController):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "instance": instance})

    return At


@contextlib.contextmanager
def _run(tmp_path, *, instance: int = 0, autopilot=None):
    """A run of the fixture quad with its camera gimbal at `instance`, on the CPU device, its airframe climbing
    straight under `autopilot`, a stand-in at full command by default.
    """
    path = quad.declare_gimbal(quad.author(tmp_path / f"quad_{instance}.usda", camera=True, diagonal=True))
    registry = Registry({"NexusPx4API_1": autopilot or _commanding(FULL), "NexusGimbalControllerAPI": _at(instance)})
    with wp.ScopedDevice("cpu"):
        loop = sv.build(path, components=registry, device="cpu", fall_from=(0.0, 0.0, 2.0))
        with Sim._from_loop(loop) as sim:
            yield sim


def _steps(sim, ticks: int) -> None:
    """Step `sim` `ticks` ticks."""
    for _ in range(ticks):
        sim.step()


def _heartbeats(messages) -> list[tuple]:
    """Each `HEARTBEAT` among `messages`: its sender's system and component, and the type it states."""
    return [(m.get_srcSystem(), m.get_srcComponent(), m.type) for m in messages if m.get_type() == "HEARTBEAT"]


def test_the_device_listens_on_its_port_and_answers_the_latest_sender(tmp_path, manager):
    """The device listens on UDP 13280 plus the run's instance and answers the address its last message came
    from, as component 154 of the sender's system.

    Given a started run of the fixture quad with its camera gimbal at instance 0 and at instance 2, and a stand-in
    manager of system 3 on a socket of its own, when it sends a `HEARTBEAT` and the run steps, then that socket
    receives a `HEARTBEAT` of type `MAV_TYPE_GIMBAL` from system 3, component 154; and when a second socket then
    sends, the device's next messages go to it.
    """
    seen = []
    for instance in (0, 2):
        first, second = manager(instance), manager(instance)
        with _run(tmp_path, instance=instance) as sim:
            _steps(sim, 1)  # the run binds its port at its start
            first.heartbeat()
            _steps(sim, 5)
            answered = _heartbeats(first.heard())
            second.heartbeat()
            _steps(sim, 300)  # 1.2 s: the device's next heartbeat falls due
            seen.append((answered[:1], _heartbeats(second.heard())[:1], first.heard()))

    gimbal = [(SYSTEM, GIMBAL, mavlink.MAV_TYPE_GIMBAL)]
    assert seen == [(gimbal, gimbal, [])] * 2


def test_the_device_sends_its_heartbeat_and_status_at_their_rates_of_sim_time(tmp_path, manager):
    """The device sends `HEARTBEAT` at 1 Hz and `GIMBAL_DEVICE_ATTITUDE_STATUS` at 20 Hz of sim time, and nothing
    before a message arrives.

    Given the fixture at a 4 ms tick, when the run steps 250 ticks with no sender, then no socket receives
    anything; and when a manager sends one message and the run steps 500 ticks more, then it receives 2 heartbeats
    and 40 statuses, each one more or less, and two statuses in a row differ by 50 ms in `time_boot_ms`, within
    one tick.
    """
    listener = manager()
    with _run(tmp_path) as sim:
        _steps(sim, 250)
        before = listener.heard()
        listener.heartbeat()
        _steps(sim, 500)
        after = listener.heard()

    heartbeats = len(_heartbeats(after))
    times = [m.time_boot_ms for m in after if m.get_type() == "GIMBAL_DEVICE_ATTITUDE_STATUS"]
    apart = [later - earlier for earlier, later in itertools.pairwise(times)]
    rates = (abs(heartbeats - 2) <= 1, abs(len(times) - 40) <= 1, bool(apart) and max(abs(d - 50) for d in apart) <= 4)
    assert (before, rates) == ([], (True, True, True)), (heartbeats, len(times), apart)


def test_the_camera_turns_to_a_setpoint_a_manager_sends(tmp_path, manager):
    """The camera turns to a setpoint a manager sends.

    Given the fixture climbing straight and a manager that sends 30° down with yaw following after tick 25, when
    the run flies 150 ticks, then the camera stays level before, never passes 33° down, and is within 1° of 30°
    down from tick 125 on.
    """
    sender = manager()
    setpoint = mavlink.MAVLink_gimbal_device_set_attitude_message(
        SYSTEM, GIMBAL, IN_VEHICLE, list(DOWN_30_WXYZ), math.nan, math.nan, math.nan
    )
    with _run(tmp_path) as sim:
        _steps(sim, 25)
        sender.send(setpoint)
        _steps(sim, 125)
        down = [_down(q) for q in np.array(sim["vehicle/body/camera"].history_arrays()["orientation"])]

    before = max(abs(d) for d in down[:25]) < 1.0
    most = max(down) < 33.0
    settled = max(abs(d - 30.0) for d in down[125:]) < 1.0
    assert (before, most, settled) == (True, True, True), down[124:]


def test_a_port_another_process_holds_fails_the_run_at_its_start_and_the_error_names_the_port(tmp_path):
    """A port another process holds fails the run at its start, and the error names the port.

    Given a socket bound to UDP 13280, when a run of the fixture at instance 0 starts, then it raises `RunError`
    before the first tick, and the message names UDP 13280.
    """
    autopilot = _commanding(FULL)
    error = None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as held:
        held.bind(("0.0.0.0", PORT))
        try:
            with _run(tmp_path, autopilot=autopilot) as sim:
                sim.step()
        except Exception as exc:
            error = exc

    assert (type(error), "UDP 13280" in str(error), autopilot.ticks) == (RunError, True, 0), error


def _plan(path, registry: Registry, caplog) -> str:
    """The stage plan the run of the vehicle at `path` logs at its start, or, for a run that fails to build, a plan
    of one host stage that holds the error.
    """
    caplog.clear()
    try:
        with (
            caplog.at_level(logging.INFO, logger="nexus"),
            Sim(str(path), scene=sv.SCENE, device="cpu", max_steps=2, registry=registry) as sim,
        ):
            sim.run()
    except ValueError as exc:
        return f"stage plan: host({exc})"
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("stage plan:"))


def test_the_devices_end_is_one_host_stage_beside_the_first_controllers_and_the_tick_gains_no_graph_segment(
    tmp_path, caplog
):
    """The device's end is one host stage beside the first controller's, and the tick gains no graph segment.

    Given the fixture with a host-stage first controller declared before the gimbal's, with and without the
    gimbal's controller, when each run starts, then the plan with it holds one host stage more, right after the
    first controller's, and the same count of graph segments. The law, `gimbal_law`, runs inside a graph.
    """
    registry = Registry({"NexusPx4API_1": _commanding(FULL)})
    drawn = [quad.author(tmp_path / name, camera=True, diagonal=True) for name in ("with.usda", "without.usda")]
    with_law = quad.declare_gimbal(drawn[0])
    without = quad.declare_gimbal(drawn[1], controller=False)

    def segments(plan: str) -> list[str]:
        """Each segment of `plan` in order: a host stage by its name, a device segment by its kind alone."""
        parts = [part.rstrip(")") for part in plan.removeprefix("stage plan: ").split(") ")]
        return [part if part.startswith("host(") else part.split("(")[0] for part in parts]

    plans = _plan(with_law, registry, caplog), _plan(without, registry, caplog)

    with_device, plain = segments(plans[0]), segments(plans[1])
    added = [i for i, segment in enumerate(with_device) if segment.startswith("host(") and segment not in plain]
    beside = [with_device[i - 1].startswith("host(") for i in added]
    rest = [segment for i, segment in enumerate(with_device) if i not in added]
    assert ("gimbal_law" in plans[0], len(added), beside, rest) == (True, 1, [True], plain), plans[0]
