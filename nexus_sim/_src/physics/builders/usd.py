import math
from dataclasses import asdict
from pathlib import Path

import newton
import warp as wp

from nexus_sim._src.usd.reader import read_declarations
from nexus_sim._src.vehicle.forces.propellers import Propeller
from nexus_sim._src.vehicle.rotors import RPM_PER_RADS

from .builder_base import BuilderBase

PROPELLER_SCHEMA = "NexusPropellerAPI"  # the schema a rotor's rigid body applies to declare its propeller
_MOTOR_PREFIX = "motor:"  # the joint attributes of the single-body examples' motor model, such as motor:tau


def parse_rotors(usd_path: str | Path) -> tuple[dict, list[str]]:
    """Read the rotors the vehicle's Universal Scene Description (USD) file declares.

    A rotor is a rigid body that applies ``NexusPropellerAPI``. Its joint is the revolute joint that has it
    as child body, and its motor is the ``NewtonActuator`` prim that drives that joint. The motor's no-load speed,
    ``newton:velocityLimit``, is the rotor speed at full command. The propeller model is a lumped-scalar one,
    so the rotors must declare the same values.

    Returns:
        The flat map every consumer reads and each rotor's joint path. The map holds the propeller's ``ct``,
        ``cd``, ``aero_h`` and ``aero_hforce``, the speed at full command as ``rpm_max``, and each ``motor:*``
        attribute of the rotor joints under its bare name, such as ``tau``.

    Raises:
        ValueError: The vehicle declares no rotor; a propeller sits on a prim that isn't the child body of a
            revolute joint; the rotors don't share one parent body or one set of values; no motor with a finite
            ``newton:velocityLimit`` drives a rotor's joint; or a prim still authors a ``propeller:*`` attribute,
            which the schema replaces. The message names the prim.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    joint_of, motor_of = {}, {}  # a child body's revolute joint, and the actuator prim that drives a joint
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if legacy := [prop.GetName() for prop in prim.GetAuthoredPropertiesInNamespace("propeller")]:
            raise ValueError(
                f"{prim.GetPath()}: {', '.join(legacy)} is no longer read; declare the propeller with "
                f"{PROPELLER_SCHEMA} on the rotor's rigid body"
            )
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            for body in prim.GetRelationship("physics:body1").GetTargets():
                joint_of[str(body)] = prim
        elif prim.GetTypeName() == "NewtonActuator":
            for joint in prim.GetRelationship("newton:targets").GetTargets()[:1]:  # Newton honors the first
                motor_of[str(joint)] = prim

    rotors = []  # each rotor's body path, its parent body's path, its joint's path and its values
    for body, schema, kwargs in read_declarations(usd_path):
        if schema != PROPELLER_SCHEMA:
            continue
        joint = joint_of.get(body)
        if joint is None:
            raise ValueError(
                f"{body}: {PROPELLER_SCHEMA} sits on a prim that isn't the child body of a revolute joint; "
                "apply it to a rotor's rigid body"
            )
        motor = motor_of.get(str(joint.GetPath()))
        if motor is None:
            raise ValueError(f"{body}: no NewtonActuator prim drives its joint {joint.GetPath()}, so it has no motor")
        attr = motor.GetAttribute("newton:velocityLimit")
        speed = attr.Get() if attr else None
        if speed is None or not math.isfinite(speed):
            raise ValueError(
                f"{motor.GetPath()}: newton:velocityLimit is the rotor speed at full command, and it is {speed}; "
                "author the motor's no-load speed"
            )
        values = {**asdict(Propeller(**kwargs)), "rpm_max": speed * RPM_PER_RADS}
        for a in joint.GetAttributes():
            if a.GetName().startswith(_MOTOR_PREFIX):
                values[a.GetName().removeprefix(_MOTOR_PREFIX)] = a.Get()
        parent = next(iter(joint.GetRelationship("physics:body0").GetTargets()), None)
        rotors.append((body, str(parent), str(joint.GetPath()), {k: float(v) for k, v in values.items()}))

    if not rotors:
        root = stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else usd_path
        raise ValueError(f"{root}: the vehicle declares no rotor; apply {PROPELLER_SCHEMA} to each rotor's rigid body")
    if len({parent for _, parent, _, _ in rotors}) > 1:
        placed = ", ".join(f"{body} on {parent}" for body, parent, _, _ in rotors)
        raise ValueError(f"the rotors don't share one parent body: {placed}")
    first = rotors[0][3]
    if odd := [f"{body} {values}" for body, _, _, values in rotors if values != first]:
        raise ValueError(
            "the rotors declare different values, and the propeller model takes one set: "
            f"{rotors[0][0]} {first}, {', '.join(odd)}"
        )
    return first, [joint for _, _, joint, _ in rotors]


class USDBuilder(BuilderBase):
    """Build a vehicle from a **USD** asset: the unified vehicle definition.

    The resolved, content-addressed and sha-verified USD path arrives as ``cfg['usd_path']``. In
    production that path comes from :mod:`nexus_sim._src.config`'s resolver, so the *same* USD asset serves
    every consumer: the standalone runtime, here, via ``ModelBuilder.add_usd``; the Isaac **Sim**
    runtime, RTX and photoreal, which also runs Newton physics, so it can build from the *same* stage
    via ``add_usd(source=stage)`` and then reference the USD onto the Kit stage for
    rendering; and the Isaac **Lab** RL app, ``nexus-rl``, via ``UsdFileCfg``. ``add_usd`` reads
    geometry, mass, inertia and joints natively.

    USD support needs ``usd-core`` to parse and ``newton-usd-schemas`` for the schema resolvers: newton's
    ``importers`` deps *minus* the open3d remesh and convex-decomp stack the lean runtime skips, see the
    pyproject. Each rotor declares its propeller with the ``NexusPropellerAPI`` schema on its rigid body, beside
    the ``NewtonActuator`` prim that declares its motor, which :func:`parse_rotors` reads at build time.

    Start pose: ``cfg['spawn_pos']``; the default is support-height placement, where the lowest point
    of the USD bounds, in the start attitude, lands 1 cm off the ground, so no more 2 m settle drop.
    Orientation is the FRD→world ``init_att``, 180° about X: the vehicle body's authored frame is
    **Forward-Right-Down (FRD)**, rotors on body −Z, so a 180° X rotation lands it **upright**, rotors
    up, in Newton's Z-up world *and* makes the Inertial Measurement Unit (IMU) and mag read the FRD
    frame PX4 Hardware In The Loop (HIL) expects. Placing it at identity instead leaves the FRD body
    **inverted**, rotors down: the vehicle reads an upside-down attitude, ``zacc ≈ +9.81``, and PX4
    refuses to arm and reports "Attitude failure (roll)" as the reason. Override with
    ``cfg['spawn_att']``, a wp.quat, if a future USD's authored frame differs.
    """

    def actuator_params(self) -> dict:
        """The values the rotors the USD declares share; see :func:`parse_rotors`."""
        return parse_rotors(self.cfg["usd_path"])[0]

    def rotor_joints(self) -> list[str]:
        """The joint path of each rotor the USD declares; see :func:`parse_rotors`."""
        return parse_rotors(self.cfg["usd_path"])[1]

    @staticmethod
    def _ground_spawn(usd_path, att) -> tuple:
        """Support-height placement: place the base so the model's lowest point, its USD bounds
        rotated into the start attitude, touches the ground plane. This is the conventional
        floating-base placement, a bounds-based snap to ground, for any vehicle USD. Replaces the old
        fixed 2 m drop; the contact settle then only seats the feet instead of breaking a fall.
        """
        from pxr import Gf, Usd, UsdGeom

        try:
            stage = Usd.Stage.Open(str(usd_path))
            root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
            lo, hi = rng.GetMin(), rng.GetMax()
            rot = Gf.Rotation(Gf.Quatd(float(att[3]), float(att[0]), float(att[1]), float(att[2])))
            min_z = min(
                rot.TransformDir(Gf.Vec3d(x, y, z))[2]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            )
            return (0.0, 0.0, -min_z + 0.01)  # lowest point 1 cm off the ground; settle seats contact
        except Exception:
            return (0.0, 0.0, 2.0)  # unreadable bounds: the old conservative drop

    def spawn_pose(self):
        """The resolved start pose, position and attitude: *the* one placement seam. :meth:`build`
        consumes it, and the Kit render peer places the vehicle's render root with it. The default
        attitude is the FRD flip; the default position is support-height placement, :meth:`_ground_spawn`.
        """
        usd_path = self.cfg.get("usd_path")
        if not usd_path:
            raise FileNotFoundError("USDBuilder requires cfg['usd_path'] (a resolved local USD path)")
        if not Path(usd_path).exists():
            raise FileNotFoundError(f"USD asset not found: {usd_path}")
        # FRD body → Newton Z-up world: rotors up plus the FRD sensor frame. See the class docstring.
        att = self.cfg.get("spawn_att", wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi))
        pos = self.cfg.get("spawn_pos") or self._ground_spawn(usd_path, att)
        return pos, att

    def build(self, builder: newton.ModelBuilder) -> None:
        pos, att = self.spawn_pose()
        usd_path = self.cfg.get("usd_path")
        xform = wp.transform(wp.vec3(*pos), att)

        n0 = builder.joint_count
        # `source` is the first positional arg in current Newton; older builds named it differently.
        try:
            builder.add_usd(str(usd_path), xform=xform, floating=True)
        except TypeError:
            builder.add_usd(source=str(usd_path), xform=xform, floating=True)

        # Guard: a free-flying drone needs a free base joint. `floating=True` only adds one for a
        # root body that isn't already joint-connected to the world, so a USD that authors a fixed
        # world->base joint, a common "fix base link" export, silently yields a vehicle with zero
        # degrees of freedom, pinned to the ground. Fail loudly instead of shipping a dead sim.
        added = [int(t) for t in builder.joint_type[n0:]]
        if int(newton.JointType.FREE) not in added:
            raise ValueError(
                f"USD {usd_path!r} did not yield a floating base: no FREE base joint was created "
                f"(joint types added: {added}). The vehicle would be pinned to the world and cannot "
                f"fly. Re-export the USD with a free/floating base (no fixed world->base joint)."
            )
