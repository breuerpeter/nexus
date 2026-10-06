"""The rotor chain splits into a command seam, a force seam and physics stepping Newton's actuators, and
the shipped vehicle flies as before.

Real builds on the Warp CPU backend: the fixture quad of ``tests/vehicle/quad.py`` around a stand-in
controller, with stand-in command stages and force elements beside the rotors' where a test needs them,
and the hosted ``astro_max_base`` for the trajectory ``main`` flew. Skipped without newton or pxr.
"""

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.api.sim import Sim
from nexus_sim._src.build.assembly import assemble, build_orchestrator, build_scenario, resolve_device
from nexus_sim._src.build.launch import resolve_to_vehicle_builder
from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import Controls
from nexus_sim._src.core.stages import peer_stages
from nexus_sim._src.physics import NewtonPhysics
from nexus_sim._src.physics.builders.usd import USDBuilder
from nexus_sim._src.scene.site import GRAVITY
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")

# The body poses main flew for the stream, per tick: `fly_shipped()` at `b5a1fb1`, before the split, so a
# change to the rotor chain shows against it. Never re-record it after the change.
MAIN_TRAJECTORY = Path(__file__).with_name("rotor_chain_main.npz")
STREAM_TICKS = 200
SERVO_RAD = 0.5  # the angle the servo tests use [rad]
SETTLED = 0.02  # how close to its target the servo's joint sits after the flight [rad]


def stream(step: int) -> np.ndarray:
    """The fixed per-rotor command stream: rest, climb, a slight tilt, then ease off."""
    if step < 40:
        return np.zeros(4, dtype=np.float32)
    if step < 120:
        return np.full(4, 0.8, dtype=np.float32)
    if step < 160:
        return np.array([0.82, 0.78, 0.82, 0.78], dtype=np.float32)
    return np.full(4, 0.6, dtype=np.float32)


class _Stream:
    """Answers the preroll at once and sends the command the stream holds for the tick."""

    def connect(self):
        pass

    def stages(self):
        return peer_stages(self)

    def exchange(self, meas, t, timeout=None):
        return Controls(command=stream(min(t.step_index, STREAM_TICKS - 1)))

    def close(self):
        pass


class _Narrow:
    """A device-native controller whose command buffer holds three values, one short of the quad's rotors."""

    def connect(self):
        self.buf = wp.zeros((1, 3), dtype=float)

    def stages(self):
        return [Stage("act", "device", lambda tick: setattr(tick, "controls", self.buf))]

    def close(self):
        pass


@wp.kernel
def _servo_target(cmd: wp.array2d(dtype=float), channel: int, scale: float, index: int, target: wp.array(dtype=float)):
    target[index] = cmd[0, channel] * scale


class _ServoCommand:
    """A command stage beside the rotors': it writes the servo's position target from one channel of the
    controller's command, `scale` radians per unit.
    """

    def __init__(self, physics, joint: str, channel: int, scale: float):
        model, self.control = physics.model, physics.control
        j = list(model.joint_label).index(joint)
        coord = self.control.joint_target_q.shape[0] == model.joint_coord_count
        self.index = int((model.joint_q_start if coord else model.joint_qd_start).numpy()[j])
        self.channel, self.scale = channel, scale

    def stages(self):
        return [Stage("servo", "device", self._write)]

    def _write(self, tick):
        wp.launch(
            _servo_target,
            dim=1,
            inputs=(tick.controls, self.channel, self.scale, self.index),
            outputs=(self.control.joint_target_q,),
        )


@wp.kernel
def _add_lift(bodies: wp.array(dtype=wp.int32), fz: float, body_f: wp.array(dtype=wp.spatial_vector)):
    wp.atomic_add(body_f, bodies[wp.tid()], wp.spatial_vector(0.0, 0.0, fz, 0.0, 0.0, 0.0))


class _Lift:
    """A force element that adds an upward world force of `fz` newtons on each of `bodies`, every tick."""

    def __init__(self, bodies, fz: float):
        self.bodies = wp.array(list(bodies), dtype=wp.int32)
        self.fz = float(fz)

    def stages(self):
        return [Stage("lift", "device", self._add)]

    def _add(self, tick):
        wp.launch(_add_lift, dim=len(self.bodies), inputs=(self.bodies, self.fz), outputs=(tick.state.body_f,))


def _weight(physics) -> float:
    """The vehicle's weight [N]."""
    return float(physics.model.body_mass.numpy().sum()) * GRAVITY


def _bodies(physics, name: str) -> list[int]:
    """The indices of the bodies whose label ends with `name`."""
    return [i for i, label in enumerate(physics.model.body_label) if label.endswith(name)]


def _turn_joint(physics, joint: str, angle: float) -> None:
    """Turn `joint` to `angle` [rad] on the built state, before the run starts. Newton's importer reads a
    joint's authored state only through its PhysX resolver, which the build doesn't use.
    """
    model, state = physics.model, physics.current_state
    q = state.joint_q.numpy()
    q[int(model.joint_q_start.numpy()[list(model.joint_label).index(joint)])] = angle
    state.joint_q.assign(q)


def _rotor_bodies(physics) -> list[int]:
    """The indices of the quad's four rotor bodies."""
    return [i for i, label in enumerate(physics.model.body_label) if "/rotor_" in label]


def _loop(path, controller, *, commands=(), forces=()):
    """The run the quad at `path` builds around `controller`, with stand-in command stages and force
    elements built by `commands` and `forces`, each a callable of the physics, beside the rotors' own.
    """
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    resolve_device(cfg)
    vb = USDBuilder({"usd_path": str(path)}, None)
    physics = NewtonPhysics(vehicle_builder=vb, cfg=cfg)
    a = assemble(physics, vb, cfg, controller=controller)
    return Orchestrator(
        clock=a.clock,
        physics=a.physics,
        commands=[*a.commands, *(make(physics) for make in commands)],
        forces=[*a.forces, *(make(physics) for make in forces)],
        sensors=a.sensors,
        controller=a.controller,
        max_steps=quad.FLIGHT_STEPS,
    )


def _flown(orch) -> tuple[float, float | None]:
    """The airframe's climb over the flight [m], and the gimbal joint's final angle [rad] when it has one."""
    with wp.ScopedDevice("cpu"), Sim.from_orchestrator(orch) as sim:
        sim.run()
        rows = sim.physics["body"].history_arrays()["position"]
        joint = sim.physics["gimbal_joint"].history()[-1].q[0] if "gimbal_joint" in sim.physics else None
    return float(rows[-1][2] - rows[0][2]), joint


def fly_shipped(steps: int = STREAM_TICKS) -> np.ndarray:
    """Fly `astro_max_base` through the command stream on the CPU device: every body's pose per tick,
    ``(steps, bodies, 7)``.
    """
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"))
    orch = build_orchestrator("astro_max_base", cfg, vb, controller=_Stream(), max_steps=steps)
    poses = []
    orch.on_tick = lambda view, t, n: poses.append(orch.physics.state0.body_q.numpy().copy())
    orch.run()
    return np.array(poses, dtype=np.float32)


def test_the_shipped_vehicle_flies_the_same_trajectory_for_the_same_commands():
    """The shipped vehicle flies the same trajectory for the same commands.

    Given `astro_max_base` on the CPU device and a fixed 200-tick per-rotor command stream, when the run
    flies it, then every body pose at every tick matches the poses `main` flew for that stream, within 1e-5.
    """
    flown = fly_shipped()
    main = np.load(MAIN_TRAJECTORY)["body_q"]
    np.testing.assert_allclose(flown, main, rtol=0, atol=1e-5)


@pytest.mark.parametrize(("outside", "bound"), [(2.0, 1.0), (-1.0, 0.0)], ids=["above 1", "below 0"])
def test_a_rotor_command_outside_0_to_1_flies_as_the_nearest_bound(tmp_path, outside, bound):
    """A rotor command outside 0 to 1 flies as the nearest bound.

    Given a fixture quad, when one run commands 2.0 on every rotor and another 1.0, then the two
    trajectories are equal, and the same holds for -1.0 and 0.0.
    """
    path = quad.author(tmp_path / "quad.usda")
    flights = [quad.fly(quad.build(path, quad.FLIGHT_STEPS, quad.Commands([value] * 4))) for value in (outside, bound)]
    assert np.array_equal(flights[0], flights[1])


def test_a_controller_with_fewer_commands_than_rotors_fails_the_run_naming_both_counts(tmp_path):
    """A controller whose command holds fewer values than the vehicle has rotors fails the run, and the
    error names both counts.

    Given a fixture quad with four rotors and a device-native controller whose command buffer holds three
    values, when the run starts, then it fails with an error that names 4 and 3.
    """
    orch = quad.build(quad.author(tmp_path / "quad.usda"), quad.FLIGHT_STEPS, _Narrow())
    with wp.ScopedDevice("cpu"), pytest.raises(ValueError) as e, Sim.from_orchestrator(orch) as sim:
        sim.run()
    assert re.search(r"\b4\b", str(e.value)) and re.search(r"\b3\b", str(e.value))


def test_a_rotor_whose_joint_no_newton_actuator_drives_fails_the_build_naming_the_rotor(tmp_path):
    """A rotor whose joint no Newton actuator drives fails the build, and the error names the rotor.

    Given a fixture quad with the `NewtonActuator` prim of one rotor removed, when the run builds, then it
    fails and the error names that rotor's prim.
    """
    assert f"{quad.ROOT}/rotor_0" in quad.build_error(quad.author(tmp_path / "quad.usda", no_motor=0))


def test_a_newton_actuator_on_a_joint_that_is_no_rotor_still_drives_its_joint(tmp_path):
    """A Newton actuator on a joint that's no rotor still drives its joint.

    Given a fixture quad with a fifth revolute joint, a `NewtonActuator` position servo on it and the joint
    set 0.5 rad off the servo's target, when the run flies 125 ticks, then the joint sits at the target.
    """
    orch = quad.build(quad.author(tmp_path / "quad.usda", servo=True), quad.FLIGHT_STEPS, quad.Commands([0.0] * 4))
    _turn_joint(orch.physics, quad.GIMBAL_JOINT, SERVO_RAD)
    with wp.ScopedDevice("cpu"), Sim.from_orchestrator(orch) as sim:
        sim.run()
        angles = [row.q[0] for row in sim.physics["gimbal_joint"].history()]
    assert (round(angles[0], 1), abs(angles[-1]) < SETTLED) == (SERVO_RAD, True)


def test_a_command_stage_beside_the_rotors_sets_the_target_of_the_actuator_it_commands(tmp_path):
    """A command stage beside the rotors' sets the target of the actuator it commands.

    Given that fixture and a stand-in command stage that writes the servo's target from the fifth value of
    the controller's command, when the controller sends 0.5 there at full throttle, then the joint settles
    at the angle the stand-in maps 0.5 to, and the vehicle still climbs.
    """
    path = quad.author(tmp_path / "quad.usda", servo=True)
    servo = lambda physics: _ServoCommand(physics, quad.GIMBAL_JOINT, channel=4, scale=1.0)  # noqa: E731
    climb, angle = _flown(_loop(path, quad.Commands([1.0, 1.0, 1.0, 1.0, SERVO_RAD]), commands=[servo]))
    assert (abs(angle - SERVO_RAD) < SETTLED, climb > 0.5) == (True, True)


def test_a_force_element_beside_the_propellers_adds_its_wrench_to_the_body_it_acts_on(tmp_path):
    """A force element beside the propellers adds its wrench to the body it acts on.

    Given a fixture quad at zero throttle and a stand-in force element that adds twice the vehicle's weight,
    upward, on the airframe, when the run flies 125 ticks, then the airframe climbs more than 0.5 m.
    """
    path = quad.author(tmp_path / "quad.usda")
    lift = lambda physics: _Lift(_bodies(physics, "/body"), 2.0 * _weight(physics))  # noqa: E731
    climb, _ = _flown(_loop(path, quad.Commands([0.0] * 4), forces=[lift]))
    assert climb > 0.5


def test_a_force_element_on_a_rotors_body_acts_with_the_propeller(tmp_path):
    """A force element on a rotor's body acts with the propeller: the body takes the sum.

    Given a fixture quad at full throttle and a stand-in force element that adds a quarter of the vehicle's
    weight, upward, on each rotor body, when the run flies 125 ticks, then it climbs higher than it does
    without the stand-in.
    """
    path = quad.author(tmp_path / "quad.usda")
    lift = lambda physics: _Lift(_rotor_bodies(physics), 0.25 * _weight(physics))  # noqa: E731
    climb, _ = _flown(_loop(path, quad.Commands(), forces=[lift]))
    assert climb > quad.climb(path)
