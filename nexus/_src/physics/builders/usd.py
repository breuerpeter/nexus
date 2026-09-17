from pathlib import Path

import newton
import warp as wp

from .builder_base import BuilderBase

_ROTOR_ATTR_PREFIXES = ("motor:", "propeller:")  # the unified motor + propeller param namespaces


def parse_rotor_joint_params(usd_path: str | Path) -> dict:
    """Read the ``motor:*`` and ``propeller:*`` params authored on the rotor revolute joints of the vehicle's
    Universal Scene Description (USD) file.

    The unified actuator's per-rotor params ride in the vehicle USD as ``motor:*`` for the motor model plus
    ``propeller:*`` for the propeller model, custom attributes on the per-rotor ``PhysicsRevoluteJoint``
    prims, so the rotor joint is the single home for all actuator params, with no YAML and USD duplication.
    This replaces the old ``NewtonActuator`` prims plus the ``freefly:actuator:*`` and ``newton:*`` schema,
    dropped with ``newton.actuators``. This strips the namespace prefix, so it returns the flat aero, thrust
    and motor map every consumer reads: ``ct``, ``cd``, ``rpm_max``, ``aero_h``, ``aero_hforce``, ``tau``.
    The propeller model is a lumped-scalar one, so the rotors must declare the same params; this asserts
    that and returns the shared dict, or ``{}`` if none of the joints authors any.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    per_rotor: list[dict] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        params = {}
        for a in prim.GetAttributes():
            name = a.GetName()
            for prefix in _ROTOR_ATTR_PREFIXES:
                if name.startswith(prefix):
                    params[name[len(prefix) :]] = a.Get()
        if params:
            per_rotor.append(params)
    if not per_rotor:
        return {}
    first = per_rotor[0]
    for other in per_rotor[1:]:
        if other != first:
            raise ValueError(
                f"non-uniform motor:*/propeller:* params across rotor joints ({per_rotor}); the lumped "
                "propeller model requires identical rotors"
            )
    return {k: float(v) for k, v in first.items()}


class USDBuilder(BuilderBase):
    """Build a vehicle from a **USD** asset: the unified vehicle definition.

    The resolved, content-addressed and sha-verified USD path arrives as ``cfg['usd_path']``. In
    production that path comes from ``newton-config``'s resolver, so the *same* USD asset serves
    every consumer: the standalone runtime, here, via ``ModelBuilder.add_usd``; the Isaac **Sim**
    runtime, RTX and photoreal, which also runs Newton physics, so it can build from the *same* stage
    via ``add_usd(source=stage)`` and then reference the USD onto the Kit stage for
    rendering; and the Isaac **Lab** RL app, ``nexus-rl``, via ``UsdFileCfg``. ``add_usd`` reads
    geometry, mass, inertia and joints natively.

    USD support needs ``usd-core`` to parse and ``newton-usd-schemas`` for the schema resolvers: newton's
    ``importers`` deps *minus* the open3d remesh and convex-decomp stack the lean runtime skips, see the
    pyproject. The unified motor plus propeller params ride as ``motor:*`` and ``propeller:*`` USD custom
    attributes on the rotor revolute joints, which :func:`parse_rotor_joint_params` reads at build time.

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
        """The motor plus propeller params authored on the USD rotor joints; see :func:`parse_rotor_joint_params`."""
        return parse_rotor_joint_params(self.cfg["usd_path"])

    def sensor_specs(self) -> list:
        """The analytic sensors authored on the vehicle USD as ``sensor:*`` prims; see
        :func:`nexus._src.vehicle.sensors.usd.parse_sensor_prims`.
        """
        from nexus._src.vehicle.sensors.usd import parse_sensor_prims

        return parse_sensor_prims(self.cfg["usd_path"])

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
        """The resolved start pose, position and attitude: *the* one placement seam, shared by both
        runtimes. Standalone consumes it in :meth:`build`; the isaacsim runtime authors the Kit stage's
        vehicle root from it. The default attitude is the FRD flip; the default position is
        support-height placement, :meth:`_ground_spawn`.
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
