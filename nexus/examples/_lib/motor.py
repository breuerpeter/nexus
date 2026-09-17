"""The **MotorModel**: a single motor's first-order rotor-speed lag.

A motor's speed ``Ω`` is a real state: each control step it relaxes toward the commanded speed by
``α = 1 − e^{−Δt/τ}`` and saturates at ``[0, Ω_max]``; thrust is then the propeller's ``kf·Ω²``. This is
the one motor model the unified :class:`~nexus._src.vehicle.actuators.rotors.Rotors` actuator runs everywhere:
RL train + deploy, Proportional Integral Derivative (PID), sampling Model Predictive Control (MPC),
design-opt, and the PX4 path. The richer DC-motor Ordinary Differential Equation (ODE), servo torque
compared to rotor inertia + aero load, is on the roadmap; this lag is its zero-inertia special case.

**``α`` derives from ``(τ, Δt)``, and is never passed in precomputed.** ``τ = 0.033 s`` is the
physical invariant; ``Δt`` is each path's own control step, so training, ``Δt ≈ 0.02 s`` → ``α ≈ 0.454``,
and deploy, ``Δt = 0.004 s`` → ``α ≈ 0.114``, correctly get different ``α`` from the same ``τ``. Baking a
shared ``α`` would silently change the training dynamics.
"""

from __future__ import annotations

import math

import warp as wp


def motor_alpha(tau: float, dt: float) -> float:
    """The first-order lag coefficient ``α = 1 − e^{−Δt/τ}`` at this path's control step ``dt``."""
    return 1.0 - math.exp(-float(dt) / float(tau))


@wp.func
def lag_step(omega_prev: float, omega_cmd: float, alpha: float, omega_max: float) -> float:
    """One first-order-lag step of a single rotor's motor speed: ``Ω ← clip(Ω + α·(Ω_cmd − Ω), 0, Ω_max)``.
    ``omega_cmd`` should arrive pre-clamped to ``[0, Ω_max]``; the saturation surfaces the κ-limited yaw.
    """
    om = omega_prev + alpha * (omega_cmd - omega_prev)
    return wp.clamp(om, 0.0, omega_max)
