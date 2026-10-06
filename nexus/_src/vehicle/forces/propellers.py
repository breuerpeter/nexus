"""The propellers: the force element of the shipped rotor chain, and the propeller model it runs.

Each rotor's rigid body applies the ``NexusPropellerAPI`` schema, :class:`Propeller`, in the vehicle's
Universal Scene Description (USD) file, which the build reads.
The force element, :class:`Propellers`, turns each rotor's solver-integrated speed Ω and the hub's in-plane
inflow into the airflow-aware thrust and H-force, through :func:`propeller_force`, and the reaction drag
torque, and adds them to the rotor body's wrench in ``state.body_f`` in one device stage, ``propellers``.
The revolute joint transmits the lift to the airframe and the motor's reaction torque yaws it; putting the
drag on the base would double-count the motor reaction, the runaway-yaw lesson from the original
conformed-actuator validation.

The propeller model, :func:`propeller_force`: given a motor speed ``Ω`` and the rotor hub's in-plane inflow
``(vx, vy)``, the airframe velocity at the rotor offset perpendicular to the thrust axis, it returns the
scalar axial thrust and the in-plane H-force, closed-form aerodynamic terms alongside the quasi-static
``kf·Ω²`` map:

* **axial thrust** ``= max(kf·Ω² − aero_h·|v_hor|², 0)``: the static thrust reduced by the forward-flight
  thrust loss. ``aero_h`` is that loss coefficient.
* **in-plane H-force** ``= −aero_hforce·|Ω|·v_hor``: rotor in-plane drag, opposing the horizontal inflow
  and scaling with rotor speed.

The examples' single-body ``Rotors`` and every differentiable or RL rollout run this one propeller model
too. With ``aero_h = aero_hforce = 0`` it reduces **exactly** to ``kf·Ω²``, so airflow is a capability
authored per-vehicle, off by default: ``f0 − 0·|v|² = f0`` and ``max(f0, 0) = f0`` since ``f0 ≥ 0``, and
the H-force is the zero vector, bit for bit the same as the quasi-static map.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from nexus._src.core.interfaces import Stage
from nexus._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints

__all__ = ["RPM_PER_RADS", "Propeller", "Propellers", "propeller_force"]


@dataclass(frozen=True)
class Propeller:
    """One rotor's propeller, as the ``NexusPropellerAPI`` schema on the rotor's rigid body declares it.

    Args:
        ct: Thrust coefficient [N/rpm²]: the static thrust is ``ct·rpm²``.
        cd: Reaction torque per unit of thrust [m], the yaw allocation κ.
        aero_h: Forward-flight thrust-loss coefficient [kg/m]; 0 is the quasi-static ``kf·Ω²``.
        aero_hforce: In-plane rotor H-force coefficient [kg/rad].
    """

    ct: float
    cd: float
    aero_h: float
    aero_hforce: float


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
    kf: float,  # ct·RPM_PER_RADS²: static thrust = kf·Ω²
    cd: float,  # reaction-torque-per-thrust, the yaw allocation κ
    aero_h: float,  # forward-flight thrust loss; 0 = quasi-static
    aero_hforce: float,  # in-plane rotor H-force, drag
    base_body: int,  # airframe body index, the body-z thrust reference
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
    base_z_world = wp.quat_rotate(wp.transform_get_rotation(body_q[base_body]), wp.vec3(0.0, 0.0, 1.0))
    rotor_z_world = wp.quat_rotate(q_rot, wp.vec3(0.0, 0.0, 1.0))
    # In-plane inflow at the rotor: its world linear velocity in the rotor frame, x and y components.
    v_rot = wp.quat_rotate_inv(q_rot, wp.spatial_top(body_qd[bi]))
    f_b = propeller_force(omega, v_rot[0], v_rot[1], kf, aero_h, aero_hforce)  # (h_x, h_y, thrust_mag)
    # rotor-z encodes the USD-authored cw/ccw spin direction; thrust always points "up" off the body.
    thrust_sign = -wp.sign(wp.dot(rotor_z_world, base_z_world))
    force_world = wp.quat_rotate(q_rot, wp.vec3(f_b[0], f_b[1], thrust_sign * f_b[2]))
    drag_world = -f_b[2] * cd * rotor_z_world  # reaction drag torque opposes the spin, via rotor-z
    # Both on the rotor body: force in the top slot, w, torque in the bottom slot, v. Added, never assigned:
    # another force element on the body keeps its wrench.
    wp.atomic_add(body_f, bi, wp.spatial_vector(w=force_world, v=drag_world))


class Propellers:
    """The propellers' force element: each rotor's thrust, H-force and reaction drag, from its
    solver-integrated speed, added to its rotor body's wrench.

    Args:
        model: The finalized articulated model, whose rotor joints really turn.
        joints: The joint path of each rotor the vehicle USD declares, the vehicle builder's
            ``rotor_joints()``.
        ct: Thrust coefficient [N/rpm²], the propeller schema's ``nexus:ct``, the vehicle USD authority.
        cd: Reaction-torque-per-thrust, ``nexus:cd``, the yaw allocation κ.
        aero_h: Forward-flight thrust-loss coefficient, ``nexus:aeroH``; 0 = quasi-static.
        aero_hforce: In-plane rotor H-force coefficient, ``nexus:aeroHforce``.
    """

    def __init__(self, *, model, joints, ct: float, cd: float, aero_h: float = 0.0, aero_hforce: float = 0.0):
        self.kf = float(ct) * RPM_PER_RADS * RPM_PER_RADS  # thrust = kf·Ω², with Ω in rad/s
        self.cd = float(cd)
        self.aero_h = float(aero_h)
        self.aero_hforce = float(aero_hforce)
        vel_dofs, _pos_coords, bodies, base_body = find_rotor_joints(model, joints)
        self.nr = len(vel_dofs)
        self.base = int(base_body)
        self._rotor_dofs = wp.array(vel_dofs, dtype=wp.int32)
        self._rotor_bodies = wp.array(bodies, dtype=wp.int32)

    def _add(self, tick) -> None:
        """The device stage: each rotor's wrench from the live state into ``state.body_f``."""
        state = tick.state
        wp.launch(
            _propeller_wrench,
            dim=self.nr,
            inputs=(
                self.kf,
                self.cd,
                self.aero_h,
                self.aero_hforce,
                self.base,
                self._rotor_dofs,
                self._rotor_bodies,
                state.joint_qd,
                state.body_q,
                state.body_qd,
            ),
            outputs=(state.body_f,),
        )

    def stages(self) -> list[Stage]:
        """One device stage, ``propellers``, over the live state."""
        return [Stage("propellers", "device", self._add)]
