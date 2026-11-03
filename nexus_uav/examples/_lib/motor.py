"""The **MotorModel** a planner keeps: a single motor's first-order rotor-speed lag.

The plant's motors are Newton's, on the rotor joints, which physics steps. A model-based controller that plans on
a lumped single body, the sampling Model Predictive Control (MPC) planner, keeps this lag as its own model of
them: each control step a motor's speed ``Ω`` relaxes toward the commanded speed by ``α = 1 − e^{−Δt/τ}`` and
saturates at ``[0, Ω_max]``; thrust is then the propeller's ``kf·Ω²``.

**``τ`` comes from the vehicle's motors and propellers**, :func:`motor_time_constant`, never from a constant of the
planner's own, and **``α`` derives from ``(τ, Δt)``**, the planner's own step.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from nexus_uav._src.scene.world import GRAVITY
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints, rotor_of


def motor_alpha(tau: float, dt: float) -> float:
    """The first-order lag coefficient ``α = 1 − e^{−Δt/τ}`` at this path's control step ``dt``."""
    return 1.0 - math.exp(-float(dt) / float(tau))


def motor_time_constant(model, rotors) -> float:
    """The rotors' speed time constant near hover [s], from the vehicle's motors and propellers.

    A rotor's motor is a first-order DC motor, whose torque falls with speed at ``kd = Ke²/R``, so near hover a
    rotor's speed follows a small change in its voltage as a first-order lag: its spin inertia over ``kd`` plus
    the slope of the propeller's drag torque, ``2·cd·kf·Ω``, at hover speed.

    Args:
        model: The finalized articulated model, for each rotor's spin inertia.
        rotors: The rotors the vehicle declares, the ``VehicleUsd``'s ``rotors()``, each with its own values.

    Returns:
        The time constant, averaged over the rotors.
    """
    found = [rotor_of(model, rotor.body) for rotor in rotors]  # each rotor's dof, coord, body and airframe
    mass = float(model.body_mass.numpy().sum())
    axes, inertia = model.joint_axis.numpy(), model.body_inertia.numpy()
    taus = []
    for rotor, (d, _coord, b, _base) in zip(rotors, found, strict=True):
        ke = RPM_PER_RADS / float(rotor.motor["kv"])
        kd = ke * ke / float(rotor.motor["resistance"])
        kf = float(rotor.propeller["ct"]) * RPM_PER_RADS**2
        hover = math.sqrt(mass * GRAVITY / len(found) / kf)
        drag_slope = 2.0 * float(rotor.propeller["cd"]) * kf * hover
        taus.append(float(axes[d] @ inertia[b] @ axes[d]) / (kd + drag_slope))
    return float(np.mean(taus))


def rotor_spin_inertia(model, joints) -> float:
    """The rotors' spin inertia [kg·m²], each rotor body's inertia about its joint's axis, averaged over the rotors.

    Args:
        model: The finalized articulated model.
        joints: The joint path of each rotor the vehicle declares.

    Returns:
        The mean spin inertia.
    """
    vel_dofs, _pos, bodies, _base = find_rotor_joints(model, joints)
    axes, inertia = model.joint_axis.numpy(), model.body_inertia.numpy()
    return float(np.mean([axes[d] @ inertia[b] @ axes[d] for d, b in zip(vel_dofs, bodies, strict=True)]))


@wp.func
def lag_step(omega_prev: float, omega_cmd: float, alpha: float, omega_max: float) -> float:
    """One first-order-lag step of a single rotor's motor speed: ``Ω ← clip(Ω + α·(Ω_cmd − Ω), 0, Ω_max)``.
    ``omega_cmd`` should arrive pre-clamped to ``[0, Ω_max]``; the saturation surfaces the κ-limited yaw.
    """
    om = omega_prev + alpha * (omega_cmd - omega_prev)
    return wp.clamp(om, 0.0, omega_max)
