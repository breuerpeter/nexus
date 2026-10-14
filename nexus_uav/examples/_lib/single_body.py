"""Collapse a quad-X vehicle Universal Scene Description (USD) into a single rigid body: the model the
sampling Model Predictive Control (MPC) planner plans its rollouts on. The plant every run flies stays the
articulated vehicle.

Core framework rule: **models are only ever created from USDs**, never synthesized in code. So instead of
authoring a second "collapsed" asset, this module **collapses the real articulated USD at load time**: retype
the rotor REVOLUTE joints to fixed joints, then ``ModelBuilder.collapse_fixed_joints()`` merges the rotors
into the airframe. Newton computes the correct lumped mass, inertia, and Center Of Mass (COM), and keeps the
rotor disk shapes, re-parented to the single body.

The planner's rotors act through one summed base-body wrench, :mod:`~nexus_uav.examples._lib.coupling`,
built from the rotor *layout*, offsets + spins + thrust map, which :func:`collapse_to_single_body` reads from
the articulated USD **before** the joints merge away.

Quad-X only, the framework's vehicle class; a different quad is a different USD + a retune, no code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# The rotor values the vehicle USD declares and the rotor geometry: the single-body model reads them, and
# so does nexus-rl, which imports them from here, since the core reads them too and never imports the examples.
from nexus_uav._src.physics.vehicle import parse_rotors  # noqa: F401
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints, quat_to_R  # noqa: F401


@dataclass
class SingleBody:
    """A quad-X vehicle USD collapsed to ``count`` free single rigid bodies, + the rotor layout and params
    the planner's coupling, :mod:`~nexus_uav.examples._lib.coupling`, needs to drive it.

    Attributes:
        model: The finalized collapsed model: ``count`` independent free single bodies, each with the
            lumped mass/inertia/COM Newton computed on collapse + the static rotor-disk visuals.
        mass: The lumped per-body mass [kg].
        offsets: ``(nr, 3)`` rotor offsets in the base-body frame, the ``r×F`` moment arms.
        dirs: ``(nr,)`` cw/ccw spin sign per rotor, the yaw-reaction sign.
        act: The rotors' values the USD declares, ``ct``/``cd``/``rpm_max``/``aero_*``. Feeds
            :func:`~nexus_uav.examples._lib.mixer.build_rotor_mixer_from_layout` + the coupling + the
            hover-thrust calc.
    """

    model: Any
    mass: float
    offsets: np.ndarray
    dirs: np.ndarray
    act: dict


def _fix_rotor_joints(builder) -> int:
    """Retype every rotor REVOLUTE joint in ``builder`` to fixed joints so the next ``collapse_fixed_joints()``
    merges the rotors into the airframe. Returns the count retyped. Newton reconciles the joint DOFs itself
    on collapse; only ``joint_type`` needs changing.
    """
    import newton

    rev, fixed = int(newton.JointType.REVOLUTE), int(newton.JointType.FIXED)
    n = 0
    for j in range(len(builder.joint_type)):
        if int(builder.joint_type[j]) == rev:
            builder.joint_type[j] = fixed
            n += 1
    return n


def _read_rotor_layout(vehicle_usd):
    """Read ``(mass, offsets(nr,3), dirs(nr,))`` from the articulated USD, its declared rotor joints, before
    the collapse: offsets in the base-body frame, spin signs from each rotor-z compared to the base-z,
    USD-authored.
    """
    import newton

    b = newton.ModelBuilder()
    b.add_ground_plane()
    vehicle_usd.build(b)
    m = b.finalize()
    st = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    bq = st.body_q.numpy()
    _vel, _pos, bodies, base = find_rotor_joints(m, vehicle_usd.rotor_joints())
    rb = quat_to_R(bq[base, 3:7])
    bz = rb @ np.array([0.0, 0.0, 1.0])
    offsets = np.array([rb.T @ (bq[r, :3] - bq[base, :3]) for r in bodies], dtype=np.float32)  # base-frame
    dirs = np.array(
        [-np.sign(np.dot(quat_to_R(bq[r, 3:7]) @ np.array([0.0, 0.0, 1.0]), bz)) for r in bodies], dtype=np.float32
    )
    mass = float(m.body_mass.numpy().sum())
    return mass, offsets, dirs


def collapse_to_single_body(
    vehicle_usd,
    *,
    count: int = 1,
    requires_grad: bool = False,
    cfg: dict | None = None,
) -> SingleBody:
    """Collapse a quad-X vehicle USD to ``count`` free single rigid bodies: the one single-body path.

    Reads the rotor layout + actuator params from the articulated USD, its rotor joints, stamps ``count``
    vehicle copies into one model, retypes each rotor REVOLUTE joint to a fixed joint and ``collapse_fixed_joints()``
    merges each vehicle's rotors into its airframe, with the correct lumped mass/inertia/COM and the
    rotor-disk visuals kept, loads the scene USD exactly as the full build path does, through
    :func:`~nexus_uav._src.scene.add_scene`, post-collapse so the scene's shape indices are
    stable, then finalizes. Pair the result with the joint-agnostic
    :class:`~nexus_uav.examples._lib.rotors.Rotors` actuator, building its mixer from
    ``sb.offsets``/``sb.dirs``: it drives the collapsed body exactly as the articulated model.

    Args:
        vehicle_usd: The articulated quad-X vehicle's ``VehicleUsd``, such as the catalog vehicle's.
        count: Number of independent single bodies to stamp: ``1`` for the real sim, ``num_rollouts`` for
            the sampling-MPC batched differentiable rollout.
        requires_grad: Build the model with gradients, for the differentiable rollout model, or without,
            for the real sim.
        cfg: The scenario cfg, read for the resolved scene, ``scene_usd_path``/``scene_start``, as
            ``NewtonPhysics``'s own build path does. ``None`` or no scene = the vehicle alone.

    Returns:
        SingleBody: the collapsed ``model`` + the rotor layout, ``mass``/``offsets``/``dirs``, + ``act``.
    """
    import newton

    from nexus_uav._src.scene import add_scene

    mass, offsets, dirs = _read_rotor_layout(vehicle_usd)
    act = dict(vehicle_usd.actuator_params())
    b = newton.ModelBuilder()
    b.add_ground_plane()
    for _ in range(int(count)):  # count copies of the catalog vehicle USD
        vehicle_usd.build(b)
    # Drop the USD-authored NewtonActuator entries, the plant's rotor motors: the collapse destroys the
    # rotor joints they target, and the planner's coupling models the motors instead.
    b.actuator_entries.clear()
    _fix_rotor_joints(b)
    b.collapse_fixed_joints()  # merge each vehicle's rotors into its airframe → count single bodies
    add_scene(b, cfg or {})  # scene USD -> builder, exactly as for the vehicle USD
    model = b.finalize(requires_grad=requires_grad)
    return SingleBody(model=model, mass=mass, offsets=offsets, dirs=dirs, act=act)
