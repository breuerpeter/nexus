"""Collapse a quad-X vehicle Universal Scene Description (USD) into a single rigid body: the one framework
seam for the single-body model path, the differentiable sampling Model Predictive Control (MPC) +
design-optimization rollouts, and their real sim.

Core framework rule: **models are only ever created from USDs**, never synthesized in code. The
differentiable controllers need one rigid body, because back-propagating through the articulated
multi-body, airframe + rotor revolute joints, gives a truncated or unstable
Backpropagation Through Time (BPTT) gradient. So instead of authoring a second "collapsed" asset, this
module **collapses the real articulated USD at load time**: retype the rotor REVOLUTE joints to fixed joints, then
``ModelBuilder.collapse_fixed_joints()`` merges the rotors into the airframe. Newton computes the correct
lumped mass, inertia, and Center Of Mass (COM), and keeps the rotor disk shapes, re-parented to the single
body, so the drone still renders with its static rotors.

The **actuator is joint-agnostic**, :class:`~nexus.examples._lib.rotors.Rotors`: it drives the base
body with a single summed wrench built from the rotor *layout*, offsets + spins + thrust map, which
:func:`collapse_to_single_body` reads from the articulated USD **before** the joints merge away.
So the one ``Rotors`` actuator flies the collapsed single body identically to the articulated model; see
the coupling docstring, ``actuators/coupling.py``. One asset, one actuator; the collapse is the only
single-body-specific step, and it lives here.

Quad-X only, the framework's vehicle class; a different quad is a different USD + a retune, no code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nexus._src.vehicle.actuators.layout import find_rotor_joints
from nexus._src.vehicle.actuators.layout import quat_to_R as _quat_to_R


@dataclass
class SingleBody:
    """A quad-X vehicle USD collapsed to ``count`` free single rigid bodies, + the rotor layout and params
    the joint-agnostic :class:`~nexus.examples._lib.rotors.Rotors` actuator needs to drive it.

    Attributes:
        model: The finalized collapsed model: ``count`` independent free single bodies, each with the
            lumped mass/inertia/COM Newton computed on collapse + the static rotor-disk visuals.
        mass: The lumped per-body mass [kg].
        offsets: ``(nr, 3)`` rotor offsets in the base-body frame, the ``r×F`` moment arms.
        dirs: ``(nr,)`` cw/ccw spin sign per rotor, the yaw-reaction sign.
        act: The USD actuator param dict, ``ct``/``cd``/``rpm_max``/``tau``/``aero_*``. Feeds
            :func:`~nexus._src.vehicle.actuators.mixer.build_rotor_mixer_from_layout` + the ``Rotors`` motor
            model + the hover-thrust calc.
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


def _read_rotor_layout(vehicle_builder):
    """Read ``(mass, offsets(nr,3), dirs(nr,))`` from the articulated USD, its rotor REVOLUTE joints, before
    the collapse: offsets in the base-body frame, spin signs from each rotor-z compared to the base-z,
    USD-authored.
    """
    import newton

    b = newton.ModelBuilder()
    b.add_ground_plane()
    vehicle_builder.build(b)
    m = b.finalize()
    st = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    bq = st.body_q.numpy()
    _vel, _pos, bodies, base = find_rotor_joints(m)
    rb = _quat_to_R(bq[base, 3:7])
    bz = rb @ np.array([0.0, 0.0, 1.0])
    offsets = np.array([rb.T @ (bq[r, :3] - bq[base, :3]) for r in bodies], dtype=np.float32)  # base-frame
    dirs = np.array(
        [-np.sign(np.dot(_quat_to_R(bq[r, 3:7]) @ np.array([0.0, 0.0, 1.0]), bz)) for r in bodies], dtype=np.float32
    )
    mass = float(m.body_mass.numpy().sum())
    return mass, offsets, dirs


def collapse_to_single_body(
    vehicle_builder,
    *,
    count: int = 1,
    requires_grad: bool = False,
    cfg: dict | None = None,
) -> SingleBody:
    """Collapse a quad-X vehicle USD to ``count`` free single rigid bodies: the one single-body seam.

    Reads the rotor layout + actuator params from the articulated USD, its rotor joints, stamps ``count``
    vehicle copies into one model, retypes each rotor REVOLUTE joint to a fixed joint and ``collapse_fixed_joints()``
    merges each vehicle's rotors into its airframe, with the correct lumped mass/inertia/COM and the
    rotor-disk visuals kept, loads the scene USD exactly as the full build path does, through
    :func:`~nexus._src.scene.add_scene`, post-collapse so the scene's shape indices are
    stable, then finalizes. Pair the result with the joint-agnostic
    :class:`~nexus.examples._lib.rotors.Rotors` actuator, building its mixer from
    ``sb.offsets``/``sb.dirs``: it drives the collapsed body exactly as the articulated model.

    Args:
        vehicle_builder: The articulated quad-X vehicle builder, for example the registry ``USDBuilder``.
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

    from nexus._src.scene import add_scene

    mass, offsets, dirs = _read_rotor_layout(vehicle_builder)
    act = dict(vehicle_builder.actuator_params())
    b = newton.ModelBuilder()
    b.add_ground_plane()
    for _ in range(int(count)):  # count copies of the registry vehicle USD
        vehicle_builder.build(b)
    # Drop any USD-authored NewtonActuator entries, the core path's rotor motors: the collapse
    # destroys the rotor joints they target, and the single-body regime runs the joint-agnostic
    # Rotors actuator instead.
    b.actuator_entries.clear()
    _fix_rotor_joints(b)
    b.collapse_fixed_joints()  # merge each vehicle's rotors into its airframe → count single bodies
    add_scene(b, cfg or {})  # scene USD -> builder, exactly as for the vehicle USD
    model = b.finalize(requires_grad=requires_grad)
    return SingleBody(model=model, mass=mass, offsets=offsets, dirs=dirs, act=act)
