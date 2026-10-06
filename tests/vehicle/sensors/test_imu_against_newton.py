"""The Inertial Measurement Unit (IMU) a vehicle declares, against Newton's ``SensorIMU`` on the same model.

Each test writes a small rig as a Universal Scene Description (USD) file: a free base body, and for one
test an arm on a hinge. The rig declares the IMU with ``NexusImuAPI`` on a prim under a body, and a Newton
site beside it with the same transform. Newton's ``ModelBuilder`` loads the rig, ``build_sensors`` builds
the IMU from its declaration, and ``SensorIMU`` reads the site. The MuJoCo solver steps the model on the
Warp CPU backend, and each tick samples both sensors from the same state.

The IMU reports the mean acceleration over the tick before, and ``SensorIMU`` the solver's acceleration
at the tick's end. So the accelerometers can differ by half a tick of the reading's rate of change, and the
gyros agree. Skipped without newton or pxr.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import newton
from newton.sensors import SensorIMU

from nexus._src.core.schema import Measurement, SimTime
from nexus._src.core.seedtree import SeedTree
from nexus._src.scene import Site
from nexus._src.vehicle.sensors.declared import build_sensors, sensor_specs

pytestmark = pytest.mark.usefixtures("warp_cpu")

DT = 0.004
TICKS = 250
GYRO_TOL = 1e-5  # rad/s
ACC_FLOOR = 1e-3  # m/s^2, what single precision leaves of a velocity difference over one tick

# A mount turned 90 degrees about z, then tilted 30 degrees about x.
TURNED = (
    "float xformOp:rotateZ = 90\n"
    "float xformOp:rotateX = 30\n"
    'uniform token[] xformOpOrder = ["xformOp:rotateZ", "xformOp:rotateX"]'
)
# A mount 0.1 m from its body's origin.
MOVED = 'double3 xformOp:translate = (0.08, 0.06, 0)\nuniform token[] xformOpOrder = ["xformOp:translate"]'

# An arm on a hinge at the base body's origin, with its center of mass on the hinge, so its weight puts no
# torque on the base and the hinge turns at a steady rate.
_ARM = """
    def Xform "arm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.2
        point3f physics:centerOfMass = (0, 0, 0)
        float3 physics:diagonalInertia = (0.001, 0.002, 0.002)
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
__ARM_PRIMS__
    }

    def PhysicsRevoluteJoint "hinge"
    {
        rel physics:body0 = </rig/base>
        rel physics:body1 = </rig/arm>
        uniform token physics:axis = "Y"
        point3f physics:localPos0 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }
"""


def _mount(transform: str) -> str:
    """The IMU's prim, with no noise, and a Newton site beside it, both with the xform ops of `transform`."""
    ops = "".join(f"            {line}\n" for line in transform.splitlines())
    return f"""
        def Xform "Imu" (
            prepend apiSchemas = ["NexusImuAPI"]
        )
        {{
            float nexus:accNoise = 0
            float nexus:gyroNoise = 0
{ops}        }}

        def Sphere "ImuSite" (
            prepend apiSchemas = ["NewtonSiteAPI"]
        )
        {{
            double radius = 0.01
{ops}        }}
"""


def _rig(tmp_path: Path, *, mount: str = "", com=(0.0, 0.0, 0.0), on_arm: bool = False) -> str:
    """Write the rig under `tmp_path` and return its path.

    `mount` is the xform ops of the IMU's prim, `com` the base body's center of mass from its origin, and
    `on_arm` puts the IMU under the arm, which the rig then has.
    """
    prims = _mount(mount)
    path = tmp_path / "rig.usda"
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "rig"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "rig" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{{
    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 2
        point3f physics:centerOfMass = {tuple(com)}
        float3 physics:diagonalInertia = (0.02, 0.03, 0.04)
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{"" if on_arm else prims}
    }}
{_ARM.replace("__ARM_PRIMS__", prims) if on_arm else ""}
}}
"""
    )
    return str(path)


def _gaps(usd_path: str, *, spin=(0.0, 0.0, 0.0), hinge_rate: float = 0.0) -> tuple[float, list, list]:
    """Step the rig at `usd_path` for 250 ticks with both sensors on it, and compare their readings.

    `spin` is the base body's angular velocity at the start, rad/s, and `hinge_rate` the arm's, when the rig
    has one. A wrench that varies over time acts on the base body.

    Returns:
        The largest difference between the gyros, rad/s; for each accelerometer axis, the largest difference,
        m/s^2; and each axis's bound: half a tick of that axis's peak rate of change in the reference, over
        `ACC_FLOOR`.
    """
    builder = newton.ModelBuilder()
    builder.add_usd(usd_path, floating=True)
    builder.gravity = -9.81
    model = builder.finalize()
    reference = SensorIMU(model, sites="*ImuSite")
    (imu,) = build_sensors(
        sensor_specs(usd_path),
        usd_path=usd_path,
        model=model,
        seedtree=SeedTree(42),
        dt=DT,
        site=Site(lat=47.6, lon=-122.3, alt=5.0, mag_ned=(0.21, 0.05, 0.43)),
    )
    solver = newton.solvers.SolverMuJoCo(model)
    state, other = model.state(), model.state()
    rates = np.zeros(model.joint_dof_count, dtype=np.float32)
    rates[3:6] = spin
    rates[6:] = hinge_rate
    state.joint_qd.assign(rates)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    meas = Measurement()
    imu.sample(state, SimTime(0.0, 0), meas)  # the first sample only seeds the IMU's earlier velocities
    ours, theirs = [], []
    for k in range(TICKS):
        t = k * DT
        wrench = np.zeros((model.body_count, 6), dtype=np.float32)
        wrench[0] = (np.sin(3.0 * t), 0.5, 22.0 + 2.0 * np.cos(2.0 * t), 0.02, 0.03 * np.sin(4.0 * t), 0.01)
        state.clear_forces()
        state.body_f.assign(wrench)
        solver.step(state, other, None, None, DT)
        state, other = other, state
        reference.update(state)
        imu.sample(state, SimTime((k + 1) * DT, k + 1), meas)
        ours.append((meas.xgyro, meas.ygyro, meas.zgyro, meas.xacc, meas.yacc, meas.zacc))
        theirs.append((*reference.gyroscope.numpy()[0], *reference.accelerometer.numpy()[0]))
    ours, theirs = np.array(ours), np.array(theirs)
    gap = np.abs(ours - theirs)
    bound = 0.5 * np.abs(np.diff(theirs[:, 3:], axis=0)).max(axis=0) + ACC_FLOOR
    return float(gap[:, :3].max()), gap[:, 3:].max(axis=0).tolist(), bound.tolist()


def _agrees(gaps: tuple[float, list, list]) -> dict[str, bool]:
    """Whether the gyros agree to `GYRO_TOL`, and whether each accelerometer axis agrees to its own bound."""
    gyro, acc, bound = gaps
    return {"gyro": gyro <= GYRO_TOL, "accelerometer": all(a <= b for a, b in zip(acc, bound, strict=True))}


def test_an_imu_at_its_bodys_origin_with_no_rotation_reads_what_newtons_sensor_reads(tmp_path):
    """An IMU at its body's origin with no rotation reads what Newton's `SensorIMU` reads, apart from the tick-mean lag.

    Given the rig with the IMU's prim at the base body's origin, a `SensorIMU` on a site at the same
    transform, no noise and a wrench that varies over time, when the solver steps 250 ticks, then the gyros
    agree to 1e-5 rad/s and each accelerometer axis differs by no more than half a tick of the reference's
    peak rate of change plus 1e-3 m/s^2.
    """
    gaps = _gaps(_rig(tmp_path))

    assert _agrees(gaps) == {"gyro": True, "accelerometer": True}, gaps


def test_an_imu_on_a_rotated_mount_reports_in_the_mounts_axes(tmp_path):
    """An IMU on a rotated mount reports the accelerometer and the gyro in the mount's axes.

    Given the rig with the IMU's prim turned 90 degrees about z and tilted 30 degrees about x, the site
    turned the same, and a slow tumble, when the solver steps 250 ticks, then both readings agree with
    `SensorIMU` within the same bounds.
    """
    gaps = _gaps(_rig(tmp_path, mount=TURNED), spin=(0.4, -0.3, 0.5))

    assert _agrees(gaps) == {"gyro": True, "accelerometer": True}, gaps


def test_an_imu_off_its_bodys_origin_feels_the_lever_arm_from_the_center_of_mass(tmp_path):
    """An IMU off its body's origin feels the lever arm from the body's center of mass.

    Given the rig with the base body's center of mass 0.05 m off its origin, the IMU's prim 0.1 m off the
    origin and a tumble of 5 rad/s, when the solver steps 250 ticks, then both readings agree with
    `SensorIMU` within the same bounds.
    """
    gaps = _gaps(_rig(tmp_path, mount=MOVED, com=(0.03, 0.0, 0.04)), spin=(3.0, 0.0, 4.0))

    assert _agrees(gaps) == {"gyro": True, "accelerometer": True}, gaps


def test_an_imu_reads_the_body_its_prim_sits_under_not_the_first_body(tmp_path):
    """An IMU reads the body its prim sits under, not the first body.

    Given the rig with an arm on a hinge that turns against the base at 3 rad/s, the IMU's prim under the
    arm and the site beside it, when the solver steps 250 ticks, then both readings agree with `SensorIMU`
    on the arm within the same bounds.
    """
    gaps = _gaps(_rig(tmp_path, on_arm=True), hinge_rate=3.0)

    assert _agrees(gaps) == {"gyro": True, "accelerometer": True}, gaps
