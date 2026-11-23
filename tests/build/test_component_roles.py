"""Each component's schema in the vehicle's Universal Scene Description (USD) file states the role the component
fills, and the build places it by that role.

A role is a schema with no attributes that a component schema includes as a built-in. The stand-in schemas
come from the project under ``tests/usd/stand_in``, which the root conftest registers. Real builds on the
Warp CPU backend of the local fixture vehicle in ``tests/usd/sensor_vehicle.py``, flown by its stand-in
controller. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from typing import ClassVar

import numpy as np
import warp as wp
from pxr import Sdf, Usd, UsdGeom

from nexus_uav._src.api.sim import Sim
from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.core.schema import PoseTwist
from nexus_uav._src.core.signals import Signal
from nexus_uav._src.scene.world import GRAVITY
from tests.usd import sensor_vehicle as sv
from tests.usd.stand_in.nexus_stand_in.estimator import StandInEstimator
from tests.usd.stand_in.nexus_stand_in.sensor import Gain
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")

ESTIMATOR = sv.prim("Estimator", "StandInEstimatorAPI", "float3 nexus:position = (1, 2, 3)", kind="Scope")


class _EstimateGuidance(Component):
    """A guidance whose host stage keeps the estimate it reads on each tick, and writes no new setpoint."""

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))
        self.seen = []

    def stages(self):
        return [Stage("guide", "host", self._guide, reads=(self.estimate,), writes=(self.setpoint,))]

    def _guide(self, tick):
        estimate = self.estimate.read()
        fields = ("position", "orientation", "linear_velocity", "angular_velocity")
        self.seen.append([float(x) for name in fields for x in getattr(estimate, name)])
        return True


class _SetpointController(sv.Controller):
    """The fixture's stand-in controller, whose exchange also reads the setpoint a guidance writes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))
        self.reads = (self.setpoint,)


class _ConnectingSensor(Component):
    """A sensor whose class also states `connect` and `close`, as a controller's does. Its host stage writes
    its gain to the signal `gain`.
    """

    def __init__(self, model, path, *, seed: int = 0, gain: float = 1.0):
        self.gain = float(gain)
        self.out = Signal("gain", Gain, shape=(1,))

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("stand_in", "host", self._sample, writes=(self.out,))]

    def _sample(self, tick):
        self.out.write([(self.gain,)])


def _on_root(tmp_path, vehicle: str, schema: str) -> str:
    """A layer over the vehicle at `vehicle` whose root prim applies `schema`; its path."""
    path = tmp_path / "root_schema.usda"
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "vehicle"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @{vehicle}@
    ]
)

over "vehicle" (
    prepend apiSchemas = ["{schema}"]
)
{{
}}
"""
    )
    return str(path)


def test_an_estimator_the_vehicle_declares_runs_in_the_estimators_place_and_the_guidance_reads_its_estimate(
    tmp_path,
):
    """An estimator that a vehicle's USD declares runs in the estimator's place in the tick, and a reader of the
    estimate reads what it writes.

    Given the fixture vehicle whose estimator scope applies a stand-in estimator schema from `tests/usd/stand_in`
    that writes a fixed pose, and a stand-in guidance that reads the estimate, when the run steps 3 ticks,
    eagerly on the CPU device, then in the start and on each tick the guidance reads the pose the estimator
    wrote just before.
    """
    path = sv.vehicle(tmp_path, root=ESTIMATOR)
    registry = sv.components(NexusPx4API_1=_SetpointController, StandInEstimatorAPI=StandInEstimator)
    loop = sv.build(path, components=registry)
    guidance = _EstimateGuidance()
    loop.guidances.append(guidance)
    sv.steps(loop, 3)
    pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert guidance.seen == [pose, pose, pose, pose]


def test_a_component_builds_in_the_role_its_schema_states_whatever_methods_its_class_has(tmp_path):
    """A component builds in the role its schema states, whatever methods its class has.

    Given the fixture vehicle with a stand-in schema that states the sensor role, on a mount prim under a body,
    whose class states `connect`, `close` and `stages`, and a stand-in estimator that reads its gain, when the
    run steps 2 ticks, then the component runs in the sensors' place in the tick, before the estimator, and
    the vehicle's PX4 controller still builds as its one controller. The estimator reads 0 in the start, which
    leaves a sensor's host stage out.
    """
    reader = sv.estimator(Signal("gain", Gain, shape=(1,)))
    path = sv.vehicle(tmp_path, sv.prim("Own", "StandInSensorAPI", "float nexus:gain = 2.5"), estimator="")
    loop = sv.build(path, components=sv.components(StandInSensorAPI=_ConnectingSensor, StandInEstimatorAPI=reader))
    controller = loop.controllers[0]
    sv.steps(loop, 2)
    gains = [float(gain.value) for _, gain in reader.kept]
    assert (gains, type(controller)) == ([0.0, 2.5, 2.5], sv.Controller)


def test_a_component_schema_that_states_no_role_fails_the_build_and_names_the_schema(tmp_path):
    """A component schema that states no role fails the build and names the schema.

    Given the fixture vehicle with a stand-in component schema that states no role, applied to a prim under a
    body, when the vehicle builds, then it fails before any stage runs, and the error names the schema and the
    prim and says the schema states no role.
    """
    path = sv.vehicle(tmp_path, sv.prim("Own", "StandInNoRoleAPI"))
    with pytest.raises(ValueError) as e:
        sv.build(path, components=sv.components(StandInNoRoleAPI=sv.StandInSensor))
    assert [s in str(e.value) for s in ("StandInNoRoleAPI", "/vehicle/body/Own", "no role")] == [True] * 3


def test_a_component_schema_that_states_two_roles_fails_the_build_and_names_both(tmp_path):
    """A component schema that states two roles fails the build and names the schema and both roles.

    Given the fixture vehicle with a stand-in schema that states the sensor role and the estimator role, when
    the vehicle builds, then it fails before any stage runs, and the error names the schema, the prim and both
    roles.
    """
    path = sv.vehicle(tmp_path, sv.prim("Both", "StandInTwoRolesAPI"))
    with pytest.raises(ValueError) as e:
        sv.build(path, components=sv.components(StandInTwoRolesAPI=sv.StandInSensor))
    named = [s in str(e.value) for s in ("StandInTwoRolesAPI", "/vehicle/body/Both", "sensor", "estimator")]
    assert named == [True] * 4


@pytest.mark.parametrize("where", ["root prim", "under a body"])
def test_an_estimator_off_its_scope_fails_the_build_and_says_where_an_estimator_sits(tmp_path, where):
    """An estimator off its scope fails the build and says where an estimator sits.

    Given the fixture vehicle with the stand-in estimator schema on the root prim, when the vehicle builds, then
    it fails before any stage runs, and the error names the prim and the schema and says where an estimator
    sits; and once with the schema on a prim under a body.
    """
    if where == "root prim":
        prim = "/vehicle"
        path = _on_root(tmp_path, sv.vehicle(tmp_path), "StandInEstimatorAPI")
    else:
        prim = "/vehicle/body/Estimator"
        path = sv.vehicle(tmp_path, ESTIMATOR)
    with pytest.raises(ValueError) as e:
        sv.build(path, components=sv.components(StandInEstimatorAPI=StandInEstimator))
    named = [s in str(e.value) for s in (prim, "StandInEstimatorAPI", "estimator", "Scope")]
    assert named == [True] * 4


# The quad's four rotors at full command.
FULL = (1.0, 1.0, 1.0, 1.0)


def _commanding(values) -> type:
    """A stand-in controller class that sends `values` every tick and counts its ticks in the class's `ticks`.

    Each call makes a class of its own, for one run.
    """

    class Commanding(quad.Commands):
        ticks = 0

        def __init__(self, **kwargs):
            super().__init__(values)  # the schema's keyword arguments, which it doesn't read

        def exchange(self, t, timeout=None):
            type(self).ticks += 1
            return super().exchange(t, timeout)

    return Commanding


@wp.kernel
def _servo_torque(
    controls: wp.array2d(dtype=float),
    channel: int,
    scale: float,
    q: int,
    qd: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
):
    """A position servo's torque, critically damped at the 4 ms tick for the gimbal body's 1e-4 kg·m²."""
    target = controls[0, channel] * scale
    wp.atomic_add(joint_f, qd, 0.5 * (target - joint_q[q]) - 0.01 * joint_qd[qd])


class _DeclaredServo(Component):
    """An actuator its joint's prim declares: it holds the joint at the angle one controls value commands, `scale`
    radians per unit, with a torque of its own into the joint's effort.
    """

    def __init__(self, model, path, *, channel: int = 0, scale: float = 1.0):
        j = list(model.joint_label).index(path)
        self.q = int(model.joint_q_start.numpy()[j])
        self.qd = int(model.joint_qd_start.numpy()[j])
        self.channel, self.scale = int(channel), float(scale)
        self.controls = Signal("controls", wp.float32, shape=(1, self.channel + 1))

    def stages(self):
        return [Stage("servo", "device", self._write, reads=(self.controls,))]

    def _write(self, tick):
        wp.launch(
            _servo_torque,
            dim=1,
            inputs=(
                self.controls.buffer,
                self.channel,
                self.scale,
                self.q,
                self.qd,
                tick.state.joint_q,
                tick.state.joint_qd,
            ),
            outputs=(tick.control.joint_f,),
        )


@wp.kernel
def _lift(body: int, fz: float, body_f: wp.array(dtype=wp.spatial_vector)):
    wp.atomic_add(body_f, body, wp.spatial_vector(0.0, 0.0, fz, 0.0, 0.0, 0.0))


class _DeclaredLift(Component):
    """A force element its body's prim declares: an upward world force of `weights` times the vehicle's weight."""

    def __init__(self, model, path, *, weights: float = 1.0):
        self.body = [str(label) for label in model.body_label].index(path)
        self.fz = float(weights) * float(model.body_mass.numpy().sum()) * GRAVITY

    def stages(self):
        return [Stage("lift", "device", self._add)]

    def _add(self, tick):
        wp.launch(_lift, dim=1, inputs=(self.body, self.fz), outputs=(tick.state.body_f,))


class _FixedGuidance(Component):
    """A guidance its scope declares: its host stage writes its setpoint on every tick."""

    def __init__(self, *, setpoint=(0.0, 0.0, 0.0)):
        self.value = [tuple(float(x) for x in setpoint)]
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))

    def stages(self):
        return [Stage("guide", "host", self._guide, writes=(self.setpoint,))]

    def _guide(self, tick):
        self.setpoint.write(self.value)
        return True


def _setpoint_reader() -> type:
    """A stand-in controller class whose exchange keeps, in the class's `seen`, the setpoint it reads each tick.

    Each call makes a class of its own, for one run.
    """

    class Reader(_SetpointController):
        seen: ClassVar[list] = []

        def exchange(self, t, timeout=None):
            type(self).seen.append(tuple(float(x) for x in np.asarray(self.setpoint.read()).reshape(-1)))
            return super().exchange(t, timeout)

    return Reader


def _declare(path, *, controller: str = "NexusPx4API_1", applied: dict | None = None) -> str:
    """Declare on the quad at `path` its controller, `controller` on the `Scope` `Controller`, and on each prim of
    `applied`, a path to `(schema, attributes)`, that schema with those attributes. Returns the path.
    """
    stage = Usd.Stage.Open(str(path))
    scope = UsdGeom.Scope.Define(stage, f"{quad.ROOT}/Controller").GetPrim()
    scope.AddAppliedSchema(controller)
    for prim_path, (schema, attrs) in (applied or {}).items():
        prim = stage.GetPrimAtPath(prim_path)
        prim.AddAppliedSchema(schema)
        for name, (kind, value) in attrs.items():
            prim.CreateAttribute(name, kind).Set(value)
    stage.GetRootLayer().Save()
    return str(path)


def _fly(path, registry: Registry, ticks: int = quad.FLIGHT_STEPS) -> dict:
    """Fly the vehicle at `path` through `Sim` with `registry` for `ticks` ticks on the CPU device. Returns the
    airframe's climb [m], the gimbal joint's last angle [rad] and rate [rad/s] when it has one, and the error the
    run raised.
    """
    try:
        with Sim(str(path), scene=sv.SCENE, device="cpu", max_steps=ticks, registry=registry) as sim:
            sim.run()
            rows = sim["vehicle/body/body"].history_arrays()["position"]
            gimbal = sim["vehicle/joints/gimbal_joint"] if "vehicle/joints/gimbal_joint" in sim else None
            angle = float(gimbal.history()[-1]["q"][0]) if gimbal is not None else None
            rate = float(gimbal.history()[-1]["qd"][0]) if gimbal is not None else None
    except Exception as exc:
        return {"climb": None, "angle": None, "rate": None, "error": exc}
    return {"climb": float(rows[-1][2] - rows[0][2]), "angle": angle, "rate": rate, "error": None}


def test_a_vehicle_that_declares_a_controller_other_than_px4_builds_and_flies_through_sim(tmp_path):
    """A vehicle that declares a controller other than PX4 builds and flies through `Sim`.

    Given the fixture quad whose controller `Scope` applies the stand-in controller schema `StandInControllerAPI`, and a
    `Registry` that maps it to a stand-in controller at full command, when `Sim(registry=...)` flies 125 ticks,
    then the stand-in's stage ran on each tick and the airframe climbed more than 0.5 m.
    """
    path = _declare(quad.author(tmp_path / "quad.usda"), controller="StandInControllerAPI")
    controller = _commanding(FULL)

    outcome = _fly(path, Registry({"StandInControllerAPI": controller}))

    climbed = outcome["climb"] is not None and outcome["climb"] > 0.5
    assert (controller.ticks >= quad.FLIGHT_STEPS, climbed) == (True, True), outcome["error"]


def test_a_guidance_the_vehicle_declares_builds_in_its_role_and_runs_before_the_controller(tmp_path):
    """A guidance the vehicle declares builds in its role and runs before the controller.

    Given the fixture vehicle whose `Guidance` scope applies the stand-in guidance schema with the setpoint
    (1, 2, 3), a stand-in controller that reads the setpoint, and `Registry` rows for both, when `Sim` steps 3
    ticks, then on each tick the controller reads the setpoint the guidance wrote that tick.
    """
    guidance = sv.prim("Guidance", "StandInGuidanceAPI", "float3 nexus:setpoint = (1, 2, 3)", kind="Scope")
    path = sv.vehicle(tmp_path, root=guidance)
    reader = _setpoint_reader()
    registry = sv.components(NexusPx4API_1=reader, StandInGuidanceAPI=_FixedGuidance)

    outcome = _fly(path, registry, ticks=3)

    assert reader.seen[-3:] == [(1.0, 2.0, 3.0)] * 3, outcome["error"]


def test_an_actuator_the_vehicle_declares_builds_through_the_registry(tmp_path):
    """An actuator the vehicle declares builds through the registry, in the actuator's place in the substep.

    Given the fixture quad whose gimbal joint's prim applies the stand-in actuator schema `StandInServoAPI` on the
    fifth controls value at 1 rad per unit, mapped in the `Registry` to a servo that holds its joint, when the run
    flies 125 ticks with 0.5 on that value, then the joint settles at 0.5 rad.
    """
    servo = {"nexus:channel": (Sdf.ValueTypeNames.Int, 4), "nexus:scale": (Sdf.ValueTypeNames.Float, 1.0)}
    path = _declare(
        quad.author(tmp_path / "quad.usda", gimbal=True), applied={quad.GIMBAL_JOINT: ("StandInServoAPI", servo)}
    )
    registry = Registry({"NexusPx4API_1": _commanding((*FULL, 0.5)), "StandInServoAPI": _DeclaredServo})

    outcome = _fly(path, registry)

    settled = outcome["angle"] is not None and abs(outcome["angle"] - 0.5) < 0.02
    assert settled, outcome


# The torque motor on the fifth controls value: 0.02 N·m/A at 0.5 A.
TORQUE_MOTOR = {
    "nexus:channel": (Sdf.ValueTypeNames.Int, 4),
    "nexus:torqueConstant": (Sdf.ValueTypeNames.Float, 0.02),
    "nexus:currentLimit": (Sdf.ValueTypeNames.Float, 0.5),
}


def test_a_joint_that_declares_the_motor_turns_at_the_rate_the_motors_torque_gives(tmp_path):
    """A vehicle whose joint declares the motor turns that joint under a constant command at the rate
    `Kt·u·currentLimit·t/J`.

    Given the fixture quad whose gimbal joint's prim applies `NexusTorqueMotorAPI` on the fifth controls value, at
    0.02 N·m/A and 0.5 A, and a stand-in controller at 0.2 on that value, so the torque is 0.002 N·m on the gimbal
    body's 1e-4 kg·m², when `Sim` flies 125 ticks, 0.5 s, then the joint's recorded rate is 10 rad/s within 2 %.
    The tolerance holds Newton's inertia check, which raises the body's inertia by 1 %, and the free airframe,
    which turns the other way.
    """
    path = _declare(
        quad.author(tmp_path / "quad.usda", gimbal=True),
        applied={quad.GIMBAL_JOINT: ("NexusTorqueMotorAPI", TORQUE_MOTOR)},
    )

    outcome = _fly(path, Registry({"NexusPx4API_1": _commanding((*FULL, 0.2))}))

    assert outcome["rate"] is not None and abs(outcome["rate"] - 10.0) < 0.2, outcome


def test_a_torque_motor_on_a_prim_that_is_no_joint_of_one_degree_of_freedom_fails_the_build(tmp_path):
    """A motor declared on a prim that's no joint of one degree of freedom fails the build, naming the prim.

    Given the fixture quad whose gimbal's rigid body, not its joint, applies `NexusTorqueMotorAPI`, when the vehicle
    builds, then it fails before any stage runs, and the message names the prim and the schema and says the schema
    sits on a joint.
    """
    body = f"{quad.ROOT}/gimbal"
    path = _declare(
        quad.author(tmp_path / "quad.usda", gimbal=True), applied={body: ("NexusTorqueMotorAPI", TORQUE_MOTOR)}
    )

    outcome = _fly(path, Registry({"NexusPx4API_1": _commanding((*FULL, 0.2))}), ticks=1)

    message = str(outcome["error"])
    assert [s in message for s in (f"{body}:", "NexusTorqueMotorAPI", "Joint")] == [True] * 3, message


def _plan(path, registry: Registry, caplog) -> str:
    """The stage plan the run of the vehicle at `path` logs at its start, or, for a run that fails to build, a plan
    of one host stage that holds the error.
    """
    import logging

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


def test_the_torque_motors_stage_adds_no_host_stage_and_no_segment_to_the_tick(tmp_path, caplog):
    """The motor's stage runs inside the substep's captured graph: a run with the motor has no host stage and no
    graph segment more than the same run without it.

    Given the fixture quad with a gimbal joint, once plain and once with `NexusTorqueMotorAPI` on the joint's
    prim, each flown by a stand-in controller, when each run starts, then the plan of the run with the motor holds
    the motor's stage, `torque_motor`, and the two plans hold the same segments in the same order, each host stage
    by its name.
    """
    registry = Registry({"NexusPx4API_1": _commanding((*FULL, 0.2))})
    plain = _declare(quad.author(tmp_path / "plain.usda", gimbal=True))
    motor = _declare(
        quad.author(tmp_path / "motor.usda", gimbal=True),
        applied={quad.GIMBAL_JOINT: ("NexusTorqueMotorAPI", TORQUE_MOTOR)},
    )

    def segments(plan: str) -> list[str]:
        """Each segment of `plan` in order: a host stage by its name, a device segment by its kind alone."""
        parts = [part.rstrip(")") for part in plan.removeprefix("stage plan: ").split(") ")]
        return [part if part.startswith("host(") else part.split("(")[0] for part in parts]

    with_motor, without = _plan(motor, registry, caplog), _plan(plain, registry, caplog)

    assert ("torque_motor" in with_motor, segments(with_motor)) == (True, segments(without)), with_motor


def test_a_force_element_the_vehicle_declares_builds_through_the_registry(tmp_path):
    """A force element the vehicle declares builds through the registry.

    Given the fixture quad at zero command whose airframe applies the stand-in force schema `StandInLiftAPI` at
    two of the vehicle's weights, mapped in the `Registry` to an element that adds that force upward, when the
    run flies 125 ticks, then the airframe climbs more than 0.5 m.
    """
    lift = {"nexus:weights": (Sdf.ValueTypeNames.Float, 2.0)}
    path = _declare(quad.author(tmp_path / "quad.usda"), applied={f"{quad.ROOT}/body": ("StandInLiftAPI", lift)})
    registry = Registry({"NexusPx4API_1": _commanding((0.0,) * 4), "StandInLiftAPI": _DeclaredLift})

    outcome = _fly(path, registry)

    assert outcome["climb"] is not None and outcome["climb"] > 0.5, outcome


class _Numbered(Component):
    """A force element that numbers an address of its own from the run's instance; the class keeps the instance
    the build hands each one in `given`.
    """

    needs_instance = True
    given: ClassVar[list] = []

    def __init__(self, model, path, *, instance: int, weights: float = 1.0):
        type(self).given.append(instance)

    def stages(self):
        return [Stage("numbered", "device", lambda tick: None)]


@pytest.mark.xdist_group("px4_ports")  # the fake PX4 dials its instance's port
def test_a_declared_force_element_that_needs_the_instance_takes_the_runs(tmp_path, warp_cpu):
    """A declared force element whose class needs the instance takes the run's, the one the run's peer takes.

    Given the fixture vehicle that declares the PX4 Software In The Loop (SITL) peer, sent to a fake that keeps
    its instance, and a stand-in force element on a prim under its body whose class states `needs_instance`, when
    the run builds, then the build hands the element the instance it hands the fake.
    """
    from nexus_uav._src.peers.px4_sitl.fake import Px4Fake

    kept: list = []

    class Kept(Px4Fake):
        def __init__(self, **run):
            kept.append(run["instance"])
            super().__init__(**run)

    numbered = type("Numbered", (_Numbered,), {"given": []})
    path = sv.vehicle(tmp_path, sv.prim("Lift", "StandInLiftAPI", kind="Xform"), px4=True)

    sv.build(path, components=sv.components(NexusPx4SitlAPI=Kept, StandInLiftAPI=numbered)).close()

    assert (numbered.given, len(kept)) == (kept, 1)


class _StandInEsc(Component):
    """A command element a rotor's body declares, `NexusEscAPI`: its device stage reads its controls channel and
    writes nothing; the class counts in `ticks` how often each instance's stage ran, by its prim.
    """

    ticks: ClassVar[dict] = {}

    def __init__(self, model, path, *, channel: int = 0, **_duty_map):
        self.path = path
        self.controls = Signal("controls", wp.float32, shape=(1, int(channel) + 1))

    def stages(self):
        return [Stage("esc", "device", self._run, reads=(self.controls,))]

    def _run(self, tick):
        type(self).ticks[self.path] = type(self).ticks.get(self.path, 0) + 1


class _StandInMotor(Component):
    """A command element a rotor's body declares, `NexusMotorAPI`: its device stage reads the voltage its ESC
    writes and writes nothing; the class counts in `ticks` how often each instance's stage ran, by its prim.
    """

    ticks: ClassVar[dict] = {}

    def __init__(self, model, path, *, kv: float = 0.0, resistance: float = 0.0, current_limit: float = 0.0):
        self.path = path
        self.voltage = Signal("voltage", wp.float32, shape=(1,), default=[0.0])

    def stages(self):
        return [Stage("motor", "device", self._run, reads=(self.voltage,))]

    def _run(self, tick):
        type(self).ticks[self.path] = type(self).ticks.get(self.path, 0) + 1


class _StandInPropeller(Component):
    """A force element a rotor's body declares, `NexusPropellerAPI`: its device stage adds no force; the class
    counts in `ticks` how often each instance's stage ran, by its prim.
    """

    ticks: ClassVar[dict] = {}

    def __init__(self, model, path, *, ct: float, cd: float, aero_h: float = 0.0, aero_hforce: float = 0.0):
        self.path = path

    def stages(self):
        return [Stage("propeller", "device", self._run)]

    def _run(self, tick):
        type(self).ticks[self.path] = type(self).ticks.get(self.path, 0) + 1


class _ReadingPropeller(_StandInPropeller):
    """A stand-in propeller that names the ESC's schema in `siblings`, so it would take the ESC's attributes."""

    siblings: ClassVar[dict[str, str]] = {"esc": "NexusEscAPI"}

    def __init__(self, model, path, *, esc: dict, **kwargs):
        super().__init__(model, path, **kwargs)


def test_a_component_class_that_names_another_schema_in_siblings_fails_the_build_and_names_the_rule(tmp_path):
    """A declared component class that names another schema in `siblings`, other than the shipped ESC, fails the
    build before any peer starts, and the message names the prim, the schema it reads and the rule.

    Given the fixture vehicle, whose rotor bodies apply `NexusPropellerAPI` and `NexusEscAPI`, and a registry that
    maps the propeller's schema to a stand-in that declares `siblings = {"esc": "NexusEscAPI"}`, when the run
    builds, then it fails with a message naming the rotor's prim, `NexusEscAPI` and that a component reads only
    its own schema's attributes.
    """
    path = sv.vehicle(tmp_path, ROTOR_COMMANDS)
    registry = sv.components(NexusPropellerAPI=_ReadingPropeller, NexusEscAPI=_StandInEsc)

    with pytest.raises(ValueError) as e:
        sv.build(path, components=registry)

    named = [s in str(e.value) for s in ("/vehicle/body/rotor_1", "NexusEscAPI", "only its own schema's attributes")]
    assert named == [True] * 3, str(e.value)


# Each rotor body of the fixture vehicle declares its command, on the controls channel of its number.
ROTOR_COMMANDS = "".join(
    f'        over "rotor_{i}" (\n            prepend apiSchemas = ["NexusEscAPI"]\n        )\n'
    f"        {{\n            int nexus:channel = {i - 1}\n        }}\n"
    for i in (1, 2, 3, 4)
)
# The stages a run owns beside its components': its clock, its physics and its recorder.
RUN_STAGES = {"clock", "clear", "step", "state", "record"}


@pytest.mark.xdist_group("px4_ports")  # the fake PX4 dials its instance's port
def test_every_class_a_run_flies_comes_from_the_registry_and_the_builder_names_none(tmp_path, caplog):
    """Every class a run flies comes from the registry, and the builder names none.

    Given a registry that maps every schema the fixture vehicle applies, each sensor's, the propeller's, the
    ESC's, the motor's, the controller's, the estimator's, the guidance's and the PX4 SITL peer's, to a stand-in,
    when `Sim(registry=...)` steps 5 ticks, then the receipt names the stand-in for every declaration and each
    tick's stages are the stand-ins' alone.
    """
    import logging

    sensor = sv.prim("Own", "StandInSensorAPI", "float nexus:gain = 2.5")
    guidance = sv.prim("Guidance", "StandInGuidanceAPI", "float3 nexus:setpoint = (1, 2, 3)", kind="Scope")
    path = sv.vehicle(tmp_path, sensor + ROTOR_COMMANDS, root=ESTIMATOR + guidance, px4=True)
    esc = type("Esc", (_StandInEsc,), {"ticks": {}})
    motor = type("Motor", (_StandInMotor,), {"ticks": {}})
    propeller = type("Propeller", (_StandInPropeller,), {"ticks": {}})
    registry = Registry(
        {
            "NexusPx4API_1": _setpoint_reader(),
            "NexusPropellerAPI": propeller,
            "NexusEscAPI": esc,
            "NexusMotorAPI": motor,
            "StandInSensorAPI": _ConnectingSensor,
            "StandInEstimatorAPI": StandInEstimator,
            "StandInGuidanceAPI": _FixedGuidance,
            "NexusPx4SitlAPI": sv.KeepingFake,
        }
    )

    try:
        with caplog.at_level(logging.INFO, logger="nexus"):
            loop = sv.build(path, components=registry)
            receipt = loop.settings
            ticks = sv.steps(loop, 5)
        plan = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("stage plan:"))
    except ValueError as exc:
        receipt, ticks, plan = {"declarations": []}, 0, f"stage plan: host({exc})"

    rotors = [f"/vehicle/body/rotor_{i}" for i in (1, 2, 3, 4)]
    resolved = {
        (d["prim"], d["schema"]): d["class"].rsplit(":", 1)[-1].rsplit(".", 1)[-1] for d in receipt["declarations"]
    }
    expected = {
        ("/vehicle/Controller", "NexusPx4API_1"): "Reader",
        ("/vehicle/body/Own", "StandInSensorAPI"): "_ConnectingSensor",
        ("/vehicle/Estimator", "StandInEstimatorAPI"): "StandInEstimator",
        ("/vehicle/Guidance", "StandInGuidanceAPI"): "_FixedGuidance",
        ("/vehicle/Px4Sitl", "NexusPx4SitlAPI"): "KeepingFake",
        **{(prim, "NexusPropellerAPI"): "Propeller" for prim in rotors},
        **{(prim, "NexusEscAPI"): "Esc" for prim in rotors},
        **{(prim, "NexusMotorAPI"): "Motor" for prim in rotors},
    }
    stages = {
        name
        for part in plan.removeprefix("stage plan: ").split(") ")
        for name in part.split("(")[-1].rstrip(")").split(" -> ")
    }
    ran = (esc.ticks, motor.ticks, propeller.ticks)
    assert (resolved, stages - RUN_STAGES, ran, ticks) == (
        expected,
        {"stand_in", "estimate", "guide", "exchange", "esc", "motor", "propeller"},
        (dict.fromkeys(rotors, 5), dict.fromkeys(rotors, 5), dict.fromkeys(rotors, 5)),
        5,
    )


# --- one or more components of each role -----------------------------------------------------------

# The fixture's own controller scope, dropped, so a test declares every controller of the vehicle itself.
NO_CONTROLLER = 'over "Controller" (\n    active = false\n)\n{\n}\n'
# Each role's stand-in schema, the attribute whose value differs on its two prims, its keyword, and the signal
# the role writes: its name, its type and its shape. `ROLE_VALUES` holds one value of each signal.
ROLE_SIGNALS = {
    "controller": ("StandInControllerAPI", "float nexus:gain = {n}", "gain", ("controls", wp.float32, (1, 4))),
    "estimator": (
        "StandInEstimatorAPI",
        "float3 nexus:position = ({n}, 0, 0)",
        "position",
        ("estimate", PoseTwist, (1,)),
    ),
    "guidance": ("StandInGuidanceAPI", "float3 nexus:setpoint = ({n}, 0, 0)", "setpoint", ("setpoint", wp.vec3, (1,))),
}
ROLE_VALUES = {
    "controls": [[0.0, 0.0, 0.0, 0.0]],
    "estimate": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
    "setpoint": [(0.0, 0.0, 0.0)],
}


def _escs_read(target: str) -> str:
    """Each rotor's ESC of the fixture vehicle reads the controls of the controller on `target`."""
    return "".join(f'over "rotor_{i}"\n{{\n    rel nexus:inputs:controls = <{target}>\n}}\n' for i in (1, 2, 3, 4))


def _ordered(key: str, output: tuple) -> type:
    """A stand-in class of any role, told apart by its schema's attribute `key`, 1 on the first prim and 2 on the
    second: its host stage writes the signal `output` and keeps, in the class's `ran`, the tick's number and
    `first` or `second`.

    Each call makes a class of its own, for one run.
    """
    name, kind, shape = output

    class Ordered(Component):
        ran: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.which = {1: "first", 2: "second"}[round(float(np.ravel(kwargs[key])[0]))]
            self.out = Signal(name, kind, shape=shape)

        def stages(self):
            return [Stage("ordered", "host", self._run, writes=(self.out,))]

        def _run(self, tick):
            type(self).ran.append((tick.t.step_index, self.which))
            self.out.write(ROLE_VALUES[name])
            return True

    return Ordered


def _reading(output: tuple) -> type:
    """A stand-in controller class whose exchange also reads the signal `output`."""
    name, kind, shape = output

    class Reading(sv.Controller):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reads = (Signal(name, kind, shape=shape),)

    return Reading


@pytest.mark.parametrize("role", ["controller", "estimator", "guidance"])
def test_a_vehicle_declares_two_components_of_one_role_and_each_ones_stages_run_every_tick_in_prim_order(
    tmp_path, role
):
    """A vehicle declares two components of one role, a controller, an estimator or a guidance, each on a `Scope`
    of its own under the root prim, and each one's stages run every tick, in the order of their prims.

    Given the fixture vehicle with two stand-ins of the role on the `Scope`s `First` and `Second`, each reader of
    their signal connected to the first, when the run steps 5 ticks on the CPU device, then each stand-in ran 5
    times, the first before the second on every tick, and the receipt names both declarations. Once per role,
    since each role has its own count check and its own slot on the loop.
    """
    schema, attribute, key, output = ROLE_SIGNALS[role]
    two = "".join(
        sv.prim(name, schema, attribute.format(n=n), kind="Scope") for n, name in ((1, "First"), (2, "Second"))
    )
    ordered = _ordered(key, output)
    if role == "controller":
        path = sv.vehicle(tmp_path, _escs_read("/vehicle/First"), root=NO_CONTROLLER + two)
        registry = sv.components(StandInControllerAPI=ordered)
    else:
        connection = f"rel nexus:inputs:{output[0]} = </vehicle/First>"
        reader = sv.prim("Reader", "StandInControllerAPI", connection, kind="Scope")
        path = sv.vehicle(tmp_path, root=NO_CONTROLLER + two + reader)
        registry = sv.components(**{schema: ordered, "StandInControllerAPI": _reading(output)})

    try:
        loop = sv.build(path, components=registry)
        declared = sorted(d["prim"] for d in loop.settings["declarations"] if d["schema"] == schema)
        sv.steps(loop, 5)
        outcome = ([which for step, which in ordered.ran if step > 0], declared)
    except Exception as exc:
        outcome = exc

    assert outcome == (["first", "second"] * 5, ["/vehicle/First", "/vehicle/Second"])


def _two_controllers(
    path, *, order=("Autopilot", "GimbalController"), connect: bool = True, servo: bool = True, wired: bool = True
):
    """Declare on the quad at `path` two controllers, `NexusPx4API_1` on the `Scope` `Autopilot` and
    `StandInControllerAPI` on the other `Scope` `order` names, `GimbalController`, defined in `order`. With
    `connect`, each rotor's ESC reads the controls of `Autopilot`. With `servo`, the gimbal joint applies the
    stand-in actuator on the first controls value, at 1 rad per unit, and with `wired` its connection names the
    other `Scope`. Returns the path.
    """
    stage = Usd.Stage.Open(str(path))
    (other,) = (name for name in order if name != "Autopilot")
    schemas = {"Autopilot": "NexusPx4API_1", other: "StandInControllerAPI"}
    for name in order:
        UsdGeom.Scope.Define(stage, f"{quad.ROOT}/{name}").GetPrim().AddAppliedSchema(schemas[name])
    if connect:
        for i in range(4):
            rotor = stage.GetPrimAtPath(f"{quad.ROOT}/rotor_{i}")
            rotor.CreateRelationship("nexus:inputs:controls", custom=False).SetTargets([f"{quad.ROOT}/Autopilot"])
    if servo:
        joint = stage.GetPrimAtPath(quad.GIMBAL_JOINT)
        joint.AddAppliedSchema("StandInServoAPI")
        joint.CreateAttribute("nexus:channel", Sdf.ValueTypeNames.Int).Set(0)
        joint.CreateAttribute("nexus:scale", Sdf.ValueTypeNames.Float).Set(1.0)
        if wired:
            joint.CreateRelationship("nexus:inputs:controls", custom=False).SetTargets([f"{quad.ROOT}/{other}"])
    stage.GetRootLayer().Save()
    return str(path)


@pytest.mark.parametrize("order", [("Autopilot", "GimbalController"), ("GimbalController", "Autopilot")])
def test_each_reader_of_the_controls_reads_the_controller_its_connection_names(tmp_path, order):
    """Each reader of the controls reads the controller its `nexus:inputs:controls` names.

    Given the fixture quad with its gimbal joint, a first controller at full command on four channels and a
    second that sends 0.5 on one channel, each ESC connected to the first and the gimbal joint's stand-in actuator
    to the second, when the run flies 125 ticks, then the airframe climbs more than 0.5 m and the joint settles at
    0.5 rad; and once with the two scopes in the other order in the file, the same.
    """
    path = _two_controllers(quad.author(tmp_path / "quad.usda", gimbal=True), order=order)
    registry = Registry(
        {
            "NexusPx4API_1": _commanding(FULL),
            "StandInControllerAPI": _commanding((0.5,)),
            "StandInServoAPI": _DeclaredServo,
        }
    )

    outcome = _fly(path, registry)

    climbed = outcome["climb"] is not None and outcome["climb"] > 0.5
    settled = outcome["angle"] is not None and abs(outcome["angle"] - 0.5) < 0.02
    assert (climbed, settled) == (True, True), outcome


def test_with_two_controllers_a_reader_of_the_controls_that_names_neither_fails_before_any_stage_runs(tmp_path):
    """With two controllers, a reader of the controls that names neither fails the run before any stage runs.

    Given the fixture quad with two controllers and no connection on its ESCs, when the run takes its first step,
    then it raises `ValueError` that names both controllers, the reader and `nexus:inputs:controls`.
    """
    path = _two_controllers(quad.author(tmp_path / "quad.usda"), connect=False, servo=False)
    autopilot, gimbal = _commanding(FULL), _commanding((0.5,))

    outcome = _fly(path, Registry({"NexusPx4API_1": autopilot, "StandInControllerAPI": gimbal}), ticks=5)

    message = str(outcome["error"])
    named = [s in message for s in ("autopilot", "gimbalcontroller", "rotor_", "nexus:inputs:controls")]
    assert (type(outcome["error"]), named, autopilot.ticks, gimbal.ticks) == (ValueError, [True] * 4, 0, 0), message


def test_a_reader_takes_no_controller_for_sharing_the_name_of_its_prim(tmp_path):
    """A reader of the controls takes no controller for sharing the name of its prim: only a component on the
    reader's own prim is its writer with nothing authored.

    Given the fixture quad with its gimbal joint, `/Vehicle/gimbal_joint`, a controller on the `Scope`
    `/Vehicle/Gimbal_Joint`, whose name in lower case is the joint's, a second controller, each ESC connected to
    the second, and the joint's stand-in actuator with no connection, when the run takes its first step, then it
    raises `ValueError` that names both controllers and `nexus:inputs:controls`, and no stage has run.
    """
    path = _two_controllers(
        quad.author(tmp_path / "quad.usda", gimbal=True), order=("Autopilot", "Gimbal_Joint"), wired=False
    )
    autopilot, gimbal = _commanding(FULL), _commanding((0.5,))
    registry = Registry({"NexusPx4API_1": autopilot, "StandInControllerAPI": gimbal, "StandInServoAPI": _DeclaredServo})

    outcome = _fly(path, registry, ticks=5)

    message = str(outcome["error"])
    named = [s in message for s in ("autopilot", "gimbal_joint", "nexus:inputs:controls")]
    assert (type(outcome["error"]), named, autopilot.ticks, gimbal.ticks) == (ValueError, [True] * 3, 0, 0), outcome


class _DeviceLaw(Component):
    """A controller of one device stage, which launches nothing and writes a controls signal one wide."""

    def __init__(self, **kwargs):
        self.controls = Signal("controls", wp.float32, shape=(1, 1))

    def stages(self):
        return [Stage("law", "device", lambda tick: None, writes=(self.controls,))]


def _segments(caplog, path, registry: Registry) -> int:
    """How many segments the stage plan of the run of the vehicle at `path` has, or 0 when the run fails."""
    import logging
    import re

    caplog.clear()
    try:
        with caplog.at_level(logging.INFO, logger="nexus"):
            sv.steps(sv.build(path, components=registry), 1)
    except Exception:
        return 0
    plan = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("stage plan:"))
    return len(re.findall(r"(?:graph|eager|host)\(", plan))


@pytest.mark.parametrize("device_first", [True, False])
def test_a_second_controller_whose_stages_are_device_stages_adds_no_segment_to_the_tick(tmp_path, caplog, device_first):
    """A second controller whose stages are device stages adds no segment to the tick.

    Given the fixture vehicle with a host-stage controller and a second stand-in controller of one device stage,
    when the run starts, then the plan has as many segments as the same run with one controller, three; once with
    the second's scope before the first's and once after.
    """
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    one = _segments(caplog, sv.vehicle(tmp_path / "one"), sv.components())
    # The fixture's `Controller` scope comes first in the file: the device law sits on it, or on the scope after.
    host = "/vehicle/Second" if device_first else "/vehicle/Controller"
    second = sv.prim("Second", "StandInControllerAPI", kind="Scope")
    path = sv.vehicle(tmp_path / "two", _escs_read(host), root=second)
    classes = (_DeviceLaw, sv.Controller) if device_first else (sv.Controller, _DeviceLaw)
    registry = Registry({"NexusPx4API_1": classes[0], "StandInControllerAPI": classes[1]})

    two = _segments(caplog, path, registry)

    assert (two, one) == (3, 3)


class _HalfSetpoint(_SetpointController):
    """A stand-in controller that reads the setpoint and sends 0.5 on its one channel."""

    width = 1

    def exchange(self, t, timeout=None):
        return np.full(1, 0.5, dtype=np.float32)


def test_a_declared_controller_estimator_or_guidance_records_under_the_name_of_its_prim(tmp_path):
    """A declared controller, estimator or guidance records under the name of its prim, in lower case, as a
    sensor does.

    Given the fixture vehicle with controllers on the `Scope`s `Controller` and `GimbalController`, a stand-in
    estimator on `Estimator` and a stand-in guidance on `Guidance`, when the run steps 5 ticks, then it holds the
    histories `vehicle/controllers/controller/controls`, `vehicle/controllers/gimbalcontroller/controls`,
    `vehicle/estimators/estimator/estimate` and `guidance/guidance/setpoint`, each with its writer's values.
    """
    scopes = (
        sv.prim("GimbalController", "StandInControllerAPI", kind="Scope")
        + ESTIMATOR
        + sv.prim("Guidance", "StandInGuidanceAPI", "float3 nexus:setpoint = (4, 5, 6)", kind="Scope")
    )
    path = sv.vehicle(tmp_path, _escs_read("/vehicle/Controller"), root=scopes)
    registry = sv.components(
        StandInControllerAPI=_HalfSetpoint, StandInEstimatorAPI=StandInEstimator, StandInGuidanceAPI=_FixedGuidance
    )
    fields = {
        "vehicle/controllers/controller/controls": "controls",
        "vehicle/controllers/gimbalcontroller/controls": "controls",
        "vehicle/estimators/estimator/estimate": "position",
        "guidance/guidance/setpoint": "setpoint",
    }

    try:
        with Sim(path, scene=sv.SCENE, device="cpu", max_steps=5, registry=registry) as sim:
            sim.run()
            newest = {
                key: [float(x) for x in np.ravel(sim[key].history()[-1][field])] if key in sim else None
                for key, field in fields.items()
            }
    except Exception as exc:
        newest = exc

    assert newest == {
        "vehicle/controllers/controller/controls": [0.0, 0.0, 0.0, 0.0],
        "vehicle/controllers/gimbalcontroller/controls": [0.5],
        "vehicle/estimators/estimator/estimate": [1.0, 2.0, 3.0],
        "guidance/guidance/setpoint": [4.0, 5.0, 6.0],
    }
