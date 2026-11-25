"""A vehicle's gimbal controller holds its camera's attitude in a run, through its three joint motors.

Each test builds the fixture quad of ``tests/vehicle/quad.py`` with its camera gimbal: a camera body behind a pan,
a roll and a tilt joint, its center of mass 2 cm off each joint's axis, a torque motor on each joint that reads
the gimbal controller's controls, and a first controller, a stand-in at fixed rotor commands, that flies the
airframe. A flight starts in free flight 2 m up, so the camera starts level and at rest. The world's x axis
points North and its z axis up, and a body's frame has x forward and z down. Skipped without newton or pxr.
"""

import logging
import math
from typing import ClassVar

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import numpy as np
import warp as wp

from nexus_uav._src.api.sim import Sim
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.vehicle.controllers.gimbal import GimbalController
from tests.usd import sensor_vehicle as sv
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")

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
    setpoint = ([(0.0, math.sin(half), 0.0, math.cos(half))], [(0.0, 0.0, 0.0)], [(True, True, False)])

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


def test_the_gimbals_law_adds_no_host_stage_and_no_segment_to_the_tick(tmp_path, caplog):
    """The law runs inside the captured graph: a run with the gimbal's controller has no host stage and no graph
    segment more than the same run without it.

    Given the fixture quad with its camera gimbal and a host-stage first controller, once with the gimbal's
    controller and its motors and once without them, when each run starts, then the plan of the run with the
    controller holds its stage, `gimbal_law`, and the two plans hold the same segments in the same order, each
    host stage by its name.
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

    assert ("gimbal_law" in plans[0], segments(plans[0])) == (True, segments(plans[1])), plans[0]
