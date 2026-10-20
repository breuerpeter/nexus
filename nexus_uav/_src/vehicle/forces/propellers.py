"""The propellers: the force element of each rotor, and the propeller model it runs.

Each rotor's rigid body applies the ``NexusPropellerAPI`` schema in the vehicle's Universal Scene
Description (USD) file, and the registry builds it as :class:`Propeller`, one force element per rotor. The element turns
its rotor's solver-integrated speed Ω and the hub's in-plane inflow into the airflow-aware thrust and H-force,
through :func:`propeller_force`, and the reaction drag torque, and adds them to the rotor body's wrench in
``state.body_f`` in one device stage, ``propeller``. The revolute joint transmits the lift to the airframe and
the motor's reaction torque yaws it; putting the drag on the base would double-count the motor reaction, the
runaway-yaw lesson from the original conformed-actuator validation. :class:`Propellers` runs the same kernel
over the rotors of many vehicles at once, as the RL task and the gradient rollout step them.

The propeller model, :func:`propeller_force`: given a motor speed ``Ω`` and the rotor hub's in-plane inflow
``(vx, vy)``, the airframe velocity at the rotor offset perpendicular to the thrust axis, it returns the
scalar axial thrust and the in-plane H-force, closed-form aerodynamic terms alongside the quasi-static
``kf·Ω²`` map:

* **axial thrust** ``= max(kf·Ω² − aero_h·|v_hor|², 0)``: the static thrust reduced by the forward-flight
  thrust loss. ``aero_h`` is that loss coefficient.
* **in-plane H-force** ``= −aero_hforce·|Ω|·v_hor``: rotor in-plane drag, opposing the horizontal inflow
  and scaling with rotor speed.

Every run, every differentiable or RL rollout, and the sampling Model Predictive Control (MPC) planner's own
model run this one propeller model. With ``aero_h = aero_hforce = 0`` it reduces **exactly** to ``kf·Ω²``,
so airflow is a capability
authored per-vehicle, off by default: ``f0 − 0·|v|² = f0`` and ``max(f0, 0) = f0`` since ``f0 ≥ 0``, and
the H-force is the zero vector, bit for bit the same as the quasi-static map.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, rotor_of

__all__ = ["RPM_PER_RADS", "Propeller", "Propellers", "propeller_force"]


@wp.func
def propeller_force(omega: float, vx: float, vy: float, kf: float, aero_h: float, aero_hforce: float) -> wp.vec3:
    """Body-frame propeller force from rotor speed ``omega`` plus in-plane inflow ``(vx, vy)``: returns
    ``(h_x, h_y, thrust_mag)``, the in-plane H-force x and y and the scalar axial thrust. ``kf = ct·
    RPM_PER_RADS²``, so ``kf·Ω²`` is the static thrust. Reduces to ``(0, 0, kf·Ω²)`` when the airflow
    coefficients are 0.
    """
    f0 = kf * omega * omega  # scalar static thrust [N], always ≥ 0
    vh2 = vx * vx + vy * vy  # in-plane inflow speed²
    thrust = wp.max(f0 - aero_h * vh2, 0.0)  # forward-flight thrust loss, a no-op at aero_h = 0
    drag = -aero_hforce * wp.abs(omega)  # H-force scales with rotor speed; opposes the in-plane inflow
    return wp.vec3(drag * vx, drag * vy, thrust)


@wp.kernel
def _propeller_wrench(
    kf: wp.array(dtype=float),  # (nr,) ct·RPM_PER_RADS² of each propeller: static thrust = kf·Ω²
    cd: wp.array(dtype=float),  # (nr,) reaction-torque-per-thrust, the yaw allocation κ
    aero_h: wp.array(dtype=float),  # (nr,) forward-flight thrust loss; 0 = quasi-static
    aero_hforce: wp.array(dtype=float),  # (nr,) in-plane rotor H-force, drag
    rotor_bases: wp.array(dtype=wp.int32),  # (nr,) each rotor's airframe body, its body-z thrust reference
    rotor_vel_dofs: wp.array(dtype=wp.int32),  # (nr,) Degrees Of Freedom (DOF) index of each rotor in joint_qd
    rotor_bodies: wp.array(dtype=wp.int32),  # (nr,) body index of each rotor
    joint_qd: wp.array(dtype=float),  # solver-integrated generalized velocities; rotor Ω lives here
    body_q: wp.array(dtype=wp.transform),  # per-body poses
    body_qd: wp.array(dtype=wp.spatial_vector),  # per-body world twists, the rotor inflow
    body_f: wp.array(dtype=wp.spatial_vector),  # world per-body wrench: force top / torque bottom
):
    """Per-rotor airflow-aware thrust + H-force + reaction drag from the SOLVER-INTEGRATED rotor
    speed Ω, added to the rotor body's wrench, single-counted through the joint; see the module docstring.
    """
    i = wp.tid()
    omega = joint_qd[rotor_vel_dofs[i]]  # rad/s, signed; Ω² terms are sign-independent
    bi = rotor_bodies[i]
    q_rot = wp.transform_get_rotation(body_q[bi])
    base_z_world = wp.quat_rotate(wp.transform_get_rotation(body_q[rotor_bases[i]]), wp.vec3(0.0, 0.0, 1.0))
    rotor_z_world = wp.quat_rotate(q_rot, wp.vec3(0.0, 0.0, 1.0))
    # In-plane inflow at the rotor: its world linear velocity in the rotor frame, x and y components.
    v_rot = wp.quat_rotate_inv(q_rot, wp.spatial_top(body_qd[bi]))
    f_b = propeller_force(omega, v_rot[0], v_rot[1], kf[i], aero_h[i], aero_hforce[i])  # (h_x, h_y, thrust_mag)
    # rotor-z encodes the USD-authored cw/ccw spin direction; thrust always points "up" off the body.
    thrust_sign = -wp.sign(wp.dot(rotor_z_world, base_z_world))
    force_world = wp.quat_rotate(q_rot, wp.vec3(f_b[0], f_b[1], thrust_sign * f_b[2]))
    drag_world = -f_b[2] * cd[i] * rotor_z_world  # reaction drag torque opposes the spin, via rotor-z
    # Both on the rotor body: force in the top slot, w, torque in the bottom slot, v. Added, never assigned:
    # another force element on the body keeps its wrench.
    wp.atomic_add(body_f, bi, wp.spatial_vector(w=force_world, v=drag_world))


class Propellers:
    """The propellers of rotors the caller has found, each with its own values and its own airframe, as the RL
    task steps the rotors of many vehicles in one model.

    Args:
        rotor_vel_dofs: ``(nr,)`` each rotor's velocity degree of freedom in ``state.joint_qd``.
        rotor_bodies: ``(nr,)`` each rotor's body.
        rotor_bases: ``(nr,)`` each rotor's airframe body, its thrust reference.
        ct: ``(nr,)`` each propeller's thrust coefficient [N/rpm²].
        cd: ``(nr,)`` each propeller's reaction-torque-per-thrust.
        aero_h: ``(nr,)`` each propeller's forward-flight thrust-loss coefficient; 0 = quasi-static.
        aero_hforce: ``(nr,)`` each propeller's in-plane rotor H-force coefficient.
    """

    def __init__(self, *, rotor_vel_dofs: wp.array, rotor_bodies: wp.array, rotor_bases: wp.array, ct, cd, aero_h, aero_hforce):  # fmt: skip
        device = rotor_vel_dofs.device
        self.nr = rotor_vel_dofs.shape[0]
        self._rotor_dofs = rotor_vel_dofs
        self._rotor_bodies = rotor_bodies
        self._rotor_bases = rotor_bases
        # thrust = kf·Ω², with Ω in rad/s: in double, then single
        kf = np.asarray(ct, dtype=np.float64) * RPM_PER_RADS * RPM_PER_RADS
        self._kf = wp.array(kf.astype(np.float32), dtype=float, device=device)
        self._cd = wp.array(np.asarray(cd, dtype=np.float32), dtype=float, device=device)
        self._aero_h = wp.array(np.asarray(aero_h, dtype=np.float32), dtype=float, device=device)
        self._aero_hforce = wp.array(np.asarray(aero_hforce, dtype=np.float32), dtype=float, device=device)

    def add(self, state) -> None:
        """Add each rotor's thrust, H-force and reaction drag, from its speed in ``state``, to its rotor body's
        wrench in ``state.body_f``.
        """
        wp.launch(
            _propeller_wrench,
            dim=self.nr,
            inputs=(
                self._kf,
                self._cd,
                self._aero_h,
                self._aero_hforce,
                self._rotor_bases,
                self._rotor_dofs,
                self._rotor_bodies,
                state.joint_qd,
                state.body_q,
                state.body_qd,
            ),
            outputs=(state.body_f,),
        )


class Propeller(Component):
    """The propeller of one rotor, which its rigid body declares with ``NexusPropellerAPI``: its thrust, H-force
    and reaction drag, from its rotor's solver-integrated speed, added to the rotor body's wrench.

    Args:
        model: The finalized articulated model, whose rotor joints really turn.
        path: The rotor's rigid body, the prim that applies the schema.
        ct: Thrust coefficient [N/rpm²], ``nexus:ct``, the vehicle USD authority.
        cd: Reaction-torque-per-thrust, ``nexus:cd``, the yaw allocation κ.
        aero_h: Forward-flight thrust-loss coefficient, ``nexus:aeroH``; 0 = quasi-static.
        aero_hforce: In-plane rotor H-force coefficient, ``nexus:aeroHforce``.
    """

    def __init__(self, model, path: str, *, ct: float, cd: float, aero_h: float = 0.0, aero_hforce: float = 0.0):
        dof, _coord, body, base = rotor_of(model, path)
        self._propellers = Propellers(
            rotor_vel_dofs=wp.array([dof], dtype=wp.int32),
            rotor_bodies=wp.array([body], dtype=wp.int32),
            rotor_bases=wp.array([base], dtype=wp.int32),
            ct=[ct],
            cd=[cd],
            aero_h=[aero_h],
            aero_hforce=[aero_hforce],
        )

    def stages(self) -> list[Stage]:
        """One device stage, ``propeller``, over the live state."""
        return [Stage("propeller", "device", lambda tick: self._propellers.add(tick.state))]
