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
    loop.guidance = guidance
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
    controller = loop.controller
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


def test_a_vehicle_that_declares_two_estimators_fails_the_build_and_names_both(tmp_path):
    """A vehicle that declares two estimators fails the build and names both.

    Given the fixture vehicle with the stand-in estimator schema on two prims, when the vehicle builds, then it
    fails before any stage runs, and the error names both prims and says a vehicle declares one estimator.
    """
    backup = sv.prim("Backup", "StandInEstimatorAPI", kind="Scope")
    path = sv.vehicle(tmp_path, root=ESTIMATOR + backup)
    with pytest.raises(ValueError) as e:
        sv.build(path, components=sv.components(StandInEstimatorAPI=StandInEstimator))
    named = [s in str(e.value) for s in ("/vehicle/Estimator", "/vehicle/Backup", "one estimator")]
    assert named == [True] * 3


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
def _target(controls: wp.array2d(dtype=float), channel: int, scale: float, index: int, out: wp.array(dtype=float)):
    out[index] = controls[0, channel] * scale


class _DeclaredServo(Component):
    """A command element its joint's prim declares: it writes the joint's target from one controls value, `scale`
    radians per unit.
    """

    def __init__(self, model, path, *, channel: int = 0, scale: float = 1.0):
        j = list(model.joint_label).index(path)
        self.q = int(model.joint_q_start.numpy()[j])
        self.qd = int(model.joint_qd_start.numpy()[j])
        self.coords = model.joint_coord_count
        self.channel, self.scale = int(channel), float(scale)
        self.controls = Signal("controls", wp.float32, shape=(1, self.channel + 1))

    def stages(self):
        return [Stage("servo", "device", self._write, reads=(self.controls,))]

    def _write(self, tick):
        target = tick.control.joint_target_q
        index = self.q if target.shape[0] == self.coords else self.qd
        wp.launch(_target, dim=1, inputs=(self.controls.buffer, self.channel, self.scale, index), outputs=(target,))


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
    airframe's climb [m], the gimbal joint's last angle [rad] when it has one, and the error the run raised.
    """
    try:
        with Sim(str(path), scene=sv.SCENE, device="cpu", max_steps=ticks, registry=registry) as sim:
            sim.run()
            rows = sim["vehicle/body/body"].history_arrays()["position"]
            gimbal = sim["vehicle/joints/gimbal_joint"] if "vehicle/joints/gimbal_joint" in sim else None
            angle = float(gimbal.history()[-1]["q"][0]) if gimbal is not None else None
    except Exception as exc:
        return {"climb": None, "angle": None, "error": exc}
    return {"climb": float(rows[-1][2] - rows[0][2]), "angle": angle, "error": None}


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


def test_a_command_element_the_vehicle_declares_builds_through_the_registry(tmp_path):
    """A command element the vehicle declares builds through the registry.

    Given the fixture quad whose gimbal joint's prim applies the stand-in command schema `StandInServoAPI` on the
    fifth controls value at 1 rad per unit, mapped in the `Registry` to a servo command element, when the run
    flies 125 ticks with 0.5 on that value, then the joint settles at 0.5 rad.
    """
    servo = {"nexus:channel": (Sdf.ValueTypeNames.Int, 4), "nexus:scale": (Sdf.ValueTypeNames.Float, 1.0)}
    path = _declare(
        quad.author(tmp_path / "quad.usda", servo=True), applied={quad.GIMBAL_JOINT: ("StandInServoAPI", servo)}
    )
    registry = Registry({"NexusPx4API_1": _commanding((*FULL, 0.5)), "StandInServoAPI": _DeclaredServo})

    outcome = _fly(path, registry)

    settled = outcome["angle"] is not None and abs(outcome["angle"] - 0.5) < 0.02
    assert settled, outcome


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
