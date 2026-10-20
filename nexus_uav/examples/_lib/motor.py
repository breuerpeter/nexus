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

    A Newton rotor motor is a speed servo with ``kd`` alone whose feedforward cancels the propeller's steady drag,
    so near hover a rotor's speed follows a small change in its target as a first-order lag: its spin inertia
    over the servo's ``kd`` plus the slope of the propeller's drag torque, ``2·cd·kf·Ω``, at hover speed.

    Args:
        model: The finalized articulated model, whose rotor joints carry Newton's motors.
        rotors: The rotors the vehicle declares, the ``VehicleUsd``'s ``rotors()``, each with its own values.

    Returns:
        The time constant, averaged over the rotors.

    Raises:
        ValueError: A rotor joint carries no motor with a ``kd`` gain. The message names its degree of freedom.
    """
    found = [rotor_of(model, rotor.body) for rotor in rotors]  # each rotor's dof, coord, body and airframe
    vel_dofs, bodies = [dof for dof, _coord, _body, _base in found], [body for _dof, _coord, body, _base in found]
    kd = {}
    for motor in model.actuators or []:
        gains = getattr(motor.controller, "kd", None)
        if gains is not None:
            kd.update(zip(motor.indices.numpy().tolist(), gains.numpy().tolist(), strict=True))
    if missing := [dof for dof in vel_dofs if dof not in kd]:
        raise ValueError(f"no Newton motor with a kd gain drives the rotor degrees of freedom {missing}")
    mass = float(model.body_mass.numpy().sum())
    axes, inertia = model.joint_axis.numpy(), model.body_inertia.numpy()
    taus = []
    for rotor, d, b in zip(rotors, vel_dofs, bodies, strict=True):
        kf = float(rotor.propeller["ct"]) * RPM_PER_RADS**2
        hover = math.sqrt(mass * GRAVITY / len(vel_dofs) / kf)
        drag_slope = 2.0 * float(rotor.propeller["cd"]) * kf * hover
        taus.append(float(axes[d] @ inertia[b] @ axes[d]) / (kd[d] + drag_slope))
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
