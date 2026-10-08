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

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import PoseTwist, PositionGoal
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.core.stages import peer_stages
from tests.usd import sensor_vehicle as sv
from tests.usd.stand_in.nexus_stand_in.estimator import StandInEstimator

pytestmark = pytest.mark.usefixtures("warp_cpu")

ESTIMATOR = sv.prim("Estimator", "StandInEstimatorAPI", "float3 nexus:position = (1, 2, 3)", kind="Scope")


class _EstimateGuidance:
    """A guidance whose host stage keeps the estimate it reads on each tick, and writes no new setpoint."""

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1, 13))
        self.setpoint = Signal("setpoint", PositionGoal, shape=(1,))
        self.seen = []

    def stages(self):
        return [Stage("guide", "host", self._guide, warm=False, reads=(self.estimate,), writes=(self.setpoint,))]

    def _guide(self, tick):
        self.seen.append([float(x) for x in self.estimate.read()[0]])
        return True


class _SetpointController(sv.Controller):
    """The fixture's stand-in controller, whose exchange also reads the setpoint a guidance writes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setpoint = Signal("setpoint", PositionGoal, shape=(1,))

    def stages(self):
        return peer_stages(self, reads=(self.setpoint,))


class _ConnectingSensor:
    """A sensor whose class also states `connect` and `close`, as a controller's does. Its host stage writes
    its gain into the `Measurement` the controller reads.
    """

    def __init__(self, run, gain: float = 1.0):
        self.gain = float(gain)

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("stand_in", "host", self._sample)]

    def _sample(self, tick):
        tick.meas.eph = self.gain


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
    eagerly on the CPU device, then on each tick the guidance reads the pose the estimator wrote on that tick.
    """
    path = sv.vehicle(tmp_path, root=ESTIMATOR)
    registry = sv.components(NexusPx4API=_SetpointController, StandInEstimatorAPI=StandInEstimator)
    loop = sv.build(path, components=registry)
    guidance = _EstimateGuidance()
    loop.guidance = guidance
    sv.fly(loop, 3)
    pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert guidance.seen == [pose, pose, pose]


def test_a_component_builds_in_the_role_its_schema_states_whatever_methods_its_class_has(tmp_path):
    """A component builds in the role its schema states, whatever methods its class has.

    Given the fixture vehicle with a stand-in schema that states the sensor role, on a mount prim under a body,
    whose class states `connect`, `close` and `stages`, when the run steps 2 ticks, then the component runs in
    the sensors' place in the tick, and the vehicle's PX4 controller still builds as its one controller.
    """
    path = sv.vehicle(tmp_path, sv.prim("Own", "StandInSensorAPI", "float nexus:gain = 2.5"))
    loop = sv.build(path, components=sv.components(StandInSensorAPI=_ConnectingSensor))
    controller = loop.controller
    received = sv.fly(loop, 2)
    assert ([meas.eph for meas in received], type(controller)) == ([2.5, 2.5], sv.Controller)


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
