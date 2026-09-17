"""The **PropellerModel**: a single propeller's airflow-aware force, in the rotor's local body frame.

Given a motor speed ``Ω`` and the rotor hub's in-plane inflow ``(vx, vy)``, the airframe velocity at the
rotor offset perpendicular to the thrust axis, it returns the scalar axial thrust and the in-plane
H-force: closed-form aerodynamic terms alongside the quasi-static ``kf·Ω²`` map:

* **axial thrust** ``= max(kf·Ω² − aero_h·|v_hor|², 0)``: the static thrust reduced by the forward-flight
  thrust loss. ``aero_h`` is that loss coefficient.
* **in-plane H-force** ``= −aero_hforce·|Ω|·v_hor``: rotor in-plane drag, opposing the horizontal inflow
  and scaling with rotor speed.

This is the one propeller model the unified :class:`~nexus._src.vehicle.actuators.rotors.Rotors` actuator and
every differentiable / RL rollout run, lifted out of the old PX4-only articulated path. With
``aero_h = aero_hforce = 0`` it reduces **exactly** to ``kf·Ω²``, so airflow is a capability authored
per-vehicle, off by default: ``f0 − 0·|v|² = f0`` and ``max(f0, 0) = f0`` since ``f0 ≥ 0``, and the
H-force is the zero vector, bit for bit the same as the quasi-static map.

The yaw reaction drag torque is *not* here: it's the ``κ·spin`` row of the control allocation ``B``,
``f → τz``, summed with the thrust into the base-body wrench by the coupling, so yaw authority comes
from the allocation, uniform with the single-body model.
"""

from __future__ import annotations

import warp as wp

from nexus._src.vehicle.actuators.layout import RPM_PER_RADS

__all__ = ["RPM_PER_RADS", "propeller_force"]


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
