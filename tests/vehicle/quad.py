"""The fixture quad the tests of the rotor chain share, and the helpers that fly it.

A quad authored per test: four rotor bodies on revolute joints under one airframe, each body carrying the
nexus propeller schema and each joint driven by a ``NewtonActuator`` prim with Newton's velocity servo and
DC motor clamp. The motor's no-load speed, 398 rad/s, is the rotor speed at full command. A test flies it
on the Warp CPU backend around a stand-in controller that sends one fixed command every tick.
"""

import numpy as np
import pytest
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from nexus_sim._src.api.sim import Sim
from nexus_sim._src.build.assembly import build_orchestrator, build_scenario
from nexus_sim._src.core.schema import Controls
from nexus_sim._src.core.stages import peer_stages
from nexus_sim._src.physics.vehicle import VehicleUsd

ROOT = "/Vehicle"
ARM = 0.2  # rotor offset from the airframe's center on each axis [m]
LIFT_CT = 4e-7  # N/rpm²: four rotors at full command make about twice the 1.08 kg vehicle's weight
WEAK_CT = 1e-7  # N/rpm²: four rotors at full command make about half its weight
FLIGHT_STEPS = 125  # 0.5 s at the default 4 ms tick
PROPELLER = {"nexus:cd": 0.05, "nexus:aeroH": 0.0, "nexus:aeroHforce": 0.0}
GIMBAL_JOINT = f"{ROOT}/gimbal_joint"  # the fifth revolute joint, with `gimbal` or `servo`


def author(
    path,
    *,
    ct=LIFT_CT,
    propeller=True,
    odd_ct=None,
    legacy_attr=False,
    pod=False,
    gimbal=False,
    reparent=False,
    servo=False,
    no_motor=None,
):
    """A quad whose rotor bodies each declare the propeller schema, with Newton's motor on each rotor joint.

    Args:
        path: Where to write the vehicle.
        ct: The thrust coefficient every rotor declares.
        propeller: Whether the rotor bodies apply the propeller schema.
        odd_ct: A thrust coefficient rotor 2 declares instead of `ct`.
        legacy_attr: Also author `propeller:ct` on rotor 0's joint.
        pod: Add a fifth body on a fixed joint that applies the propeller schema.
        gimbal: Add a fifth body on a revolute joint that applies no propeller schema.
        reparent: Hang rotor 3's joint off rotor 0's body instead of the airframe.
        servo: Add the gimbal joint and a `NewtonActuator` position servo on it, with its target at 0.
        no_motor: The index of a rotor whose joint gets no `NewtonActuator` prim.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    body = UsdGeom.Cube.Define(stage, f"{ROOT}/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)
    imu = UsdGeom.Xform.Define(stage, f"{ROOT}/body/Imu").GetPrim()
    imu.CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("imu")

    def child_body(name, offset, flipped=False):
        prim = UsdGeom.Cylinder.Define(stage, f"{ROOT}/{name}")
        prim.GetRadiusAttr().Set(0.1)
        prim.GetHeightAttr().Set(0.01)
        prim.AddTranslateOp().Set(Gf.Vec3d(*offset))
        if flipped:  # the spin axis points the other way: the rotor turns the other way for the same command
            prim.AddRotateXOp().Set(180.0)
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(prim.GetPrim())
        mass.GetMassAttr().Set(0.02)
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(5e-5, 5e-5, 1e-4))
        return prim

    def revolute(name, parent, child, offset, flipped=False):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/{name}")
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if flipped:
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
        joint.CreateAxisAttr().Set("Z")
        return joint.GetPrim()

    def apply_propeller(prim, rotor_ct):
        prim.AddAppliedSchema("NexusPropellerAPI")
        prim.CreateAttribute("nexus:ct", Sdf.ValueTypeNames.Float).Set(rotor_ct)
        for name, value in PROPELLER.items():
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)

    for i, (x, y) in enumerate([(ARM, ARM), (-ARM, -ARM), (ARM, -ARM), (-ARM, ARM)]):
        offset = (x, y, -0.12)  # rotors on the airframe's -z, up once the run spawns it upright
        flipped = bool(i % 2)
        rotor = child_body(f"rotor_{i}", offset, flipped)
        parent = f"{ROOT}/rotor_0" if reparent and i == 3 else f"{ROOT}/body"
        joint = revolute(f"rotor_{i}_joint", parent, rotor.GetPath(), offset, flipped)
        if legacy_attr and i == 0:
            joint.CreateAttribute("propeller:ct", Sdf.ValueTypeNames.Float, custom=True).Set(ct)
        if i != no_motor:
            motor = stage.DefinePrim(f"{ROOT}/rotor_{i}_motor", "NewtonActuator")
            motor.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
            motor.ApplyAPI("NewtonPIDControlAPI")
            motor.ApplyAPI("NewtonDCMotorClampingAPI")
            motor.GetAttribute("newton:kd").Set(0.025)
            motor.GetAttribute("newton:saturationEffort").Set(8.0)
            motor.GetAttribute("newton:maxMotorEffort").Set(8.0)
            motor.GetAttribute("newton:velocityLimit").Set(398.0)
        if propeller:
            apply_propeller(rotor.GetPrim(), odd_ct if odd_ct is not None and i == 2 else ct)
    if gimbal or servo:
        offset = (0.0, 0.0, 0.12)
        mount = child_body("gimbal", offset)
        joint = revolute("gimbal_joint", f"{ROOT}/body", mount.GetPath(), offset)
    if servo:
        motor = stage.DefinePrim(f"{ROOT}/gimbal_motor", "NewtonActuator")
        motor.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        motor.ApplyAPI("NewtonPDControlAPI")  # critically damped at the 4 ms tick for the body's 1e-4 kg·m²
        motor.GetAttribute("newton:kp").Set(0.5)
        motor.GetAttribute("newton:kd").Set(0.01)
    if pod:
        offset = (0.0, 0.0, -0.12)
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/pod_joint")
        fixed.CreateBody0Rel().SetTargets([f"{ROOT}/body"])
        fixed.CreateBody1Rel().SetTargets([child_body("pod", offset).GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        apply_propeller(stage.GetPrimAtPath(f"{ROOT}/pod"), ct)
    stage.GetRootLayer().Save()
    return path


class Commands:
    """Answers the preroll at once and sends the same command every tick: one value per channel."""

    def __init__(self, values=(1.0, 1.0, 1.0, 1.0)):
        self.values = np.asarray(values, dtype=np.float32)

    def connect(self):
        pass

    def stages(self):
        return peer_stages(self)

    def exchange(self, t, timeout=None):
        return Controls(command=self.values.copy())

    def close(self):
        pass


def build(path, steps=0, controller=None):
    """The run the vehicle at `path` builds, around `controller`, full throttle with no controller."""
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    return build_orchestrator(
        "quad", cfg, VehicleUsd({"usd_path": str(path)}), controller=controller or Commands(), max_steps=steps
    )


def fly(orch):
    """Fly the built run to its end and return the airframe's recorded positions, one row per tick [m]."""
    with wp.ScopedDevice("cpu"), Sim.from_orchestrator(orch) as sim:
        sim.run()
        return sim.physics["body"].history_arrays()["position"]


def climb(path, controller=None):
    """How far the airframe rises over the flight [m], at full throttle with no controller."""
    with wp.ScopedDevice("cpu"):
        rows = fly(build(path, FLIGHT_STEPS, controller))
    return float(rows[-1][2] - rows[0][2])


def build_error(path):
    """The message of the error the build of the vehicle at `path` raises."""
    with wp.ScopedDevice("cpu"), pytest.raises(ValueError) as e:
        build(path)
    return str(e.value)
