"""The differentiable Proportional Integral Derivative (PID) control law: the single source of truth, one
``obs[12] -> action[4]`` map used by **both** the in-process eager :class:`PidController`, the determinism
authority and built-in simple controller, and the differentiable design-optimization rollout,
``nexus/examples/design_opt/design_optimization.py``.

It's a hand-coded "policy": it consumes the **same 12-D body-frame observation** the trained
policy does, see ``examples/_lib/observation.py``, and emits a **4-D direct-moment action**
``[thrust, m_x, m_y, m_z]`` the single-body :class:`~nexus._src.vehicle.actuators.Rotors`
consumes in moment-input mode, allocation + motor lag with no rate loop, so the PID and a trained policy
are interchangeable at the controller seam, and the PID's gains are the differentiable *design parameters*
the rollout optimizes for the least overshoot + time-to-waypoint.

Observation layout, body frame, from ``observation_from_state``:
    obs[0:3] = lin_vel_b   obs[3:6] = ang_vel_b   obs[6:9] = grav_dir_b   obs[9:12] = goal_rel_b

Control law, a cascaded position->attitude PD; `hover` = the action[0] that holds a hover, which is
``2/thrust_to_weight - 1`` for the lumped actuator:
    action[0] = hover + kp_z*goal_z - kd_z*vz                          # altitude thrust
    roll_des  = clip(-kp_xy*goal_y + kd_xy*vy, ±tilt_limit)                  # +y via roll-left
    pitch_des = clip( kp_xy*goal_x - kd_xy*vx, ±tilt_limit)                  # +x via nose-down pitch
    action[1] = kp_att*(roll_des - (-grav_y)) - kd_att*wx             # roll  moment
    action[2] = kp_att*(pitch_des -  grav_x ) - kd_att*wy             # pitch moment
    action[3] = -kd_yaw*wz                                            # yaw-rate damp

The attitude-feedback signs come from the small-angle gravity map: roll about +x gives
``grav_b[1] = -sin φ``; pitch about +y gives ``grav_b[0] = +sin θ``.

**Forward Right Down (FRD) bodies.** All vehicle Universal Scene Description (USD) files use FRD
authoring, body +z down, and start rotors-up via a fixed 180°-about-X flip; see ``docs/conventions.md``.
The preceding law uses the upright convention, so ``_pid`` applies that flip as an exact similarity
transform: negate the y,z components of the body-frame obs vectors going in, run the law, and negate the
y,z components of the output moment going out. The collective, ``action[0]``, is a scalar; the actuator's
FRD thrust sign handles its world direction. The Warp kernel and the NumPy mirror must stay the same;
``tests`` asserts it.
"""

from __future__ import annotations

import numpy as np
import warp as wp

# Gain vector layout: the differentiable design parameters.
GAIN_NAMES = ("kp_z", "kd_z", "kp_xy", "kd_xy", "kp_att", "kd_att", "kd_yaw")
NUM_GAINS = len(GAIN_NAMES)
TILT_LIMIT = 0.3  # max commanded lean angle [rad], small-angle regime


def hover_action(thrust_to_weight: float) -> float:
    """action[0] that holds a hover under the lumped actuator: thrust = weight."""
    return 2.0 / float(thrust_to_weight) - 1.0


@wp.func
def _pid(lin: wp.vec3, ang: wp.vec3, grav: wp.vec3, goal: wp.vec3, g: wp.array(dtype=float), hover: float):
    # FRD → upright convention: negate y,z of the body-frame obs, the fixed 180°-about-X flip.
    lin = wp.vec3(lin[0], -lin[1], -lin[2])
    ang = wp.vec3(ang[0], -ang[1], -ang[2])
    grav = wp.vec3(grav[0], -grav[1], -grav[2])
    goal = wp.vec3(goal[0], -goal[1], -goal[2])
    a0 = hover + g[0] * goal[2] - g[1] * lin[2]
    roll_des = wp.clamp(-g[2] * goal[1] + g[3] * lin[1], -TILT_LIMIT, TILT_LIMIT)
    pitch_des = wp.clamp(g[2] * goal[0] - g[3] * lin[0], -TILT_LIMIT, TILT_LIMIT)
    roll_now = -grav[1]
    pitch_now = grav[0]
    a1 = g[4] * (roll_des - roll_now) - g[5] * ang[0]
    a2 = g[4] * (pitch_des - pitch_now) - g[5] * ang[1]
    a3 = -g[6] * ang[2]
    return wp.vec4(a0, a1, -a2, -a3)  # moments mapped back to the FRD frame: negate y,z


@wp.kernel
def pid_law(
    obs: wp.array(dtype=float),  # (12,) single-env observation
    gains: wp.array(dtype=float),  # (NUM_GAINS,)
    hover: float,
    action: wp.array(dtype=float),  # (4,) [thrust, m_x, m_y, m_z]
):
    lin = wp.vec3(obs[0], obs[1], obs[2])
    ang = wp.vec3(obs[3], obs[4], obs[5])
    grav = wp.vec3(obs[6], obs[7], obs[8])
    goal = wp.vec3(obs[9], obs[10], obs[11])
    a = _pid(lin, ang, grav, goal, gains, hover)
    action[0] = a[0]
    action[1] = a[1]
    action[2] = a[2]
    action[3] = a[3]


def pid_action_np(obs: np.ndarray, gains: np.ndarray, hover: float) -> np.ndarray:
    """NumPy mirror of :func:`pid_law` for the host, eager, controller path. Must match the kernel."""
    obs = np.asarray(obs, dtype=np.float32).reshape(12)
    g = np.asarray(gains, dtype=np.float32)
    flip = np.array([1.0, -1.0, -1.0], dtype=np.float32)  # FRD → upright: negate y,z of the body-frame obs
    lin, ang, grav, goal = obs[0:3] * flip, obs[3:6] * flip, obs[6:9] * flip, obs[9:12] * flip
    a0 = hover + g[0] * goal[2] - g[1] * lin[2]
    roll_des = np.clip(-g[2] * goal[1] + g[3] * lin[1], -TILT_LIMIT, TILT_LIMIT)
    pitch_des = np.clip(g[2] * goal[0] - g[3] * lin[0], -TILT_LIMIT, TILT_LIMIT)
    roll_now, pitch_now = -grav[1], grav[0]
    a1 = g[4] * (roll_des - roll_now) - g[5] * ang[0]
    a2 = g[4] * (pitch_des - pitch_now) - g[5] * ang[1]
    a3 = -g[6] * ang[2]
    return np.array([a0, a1, -a2, -a3], dtype=np.float32)  # moments mapped back to the FRD frame


# A reasonable, stable starting gain set, the design-opt baseline: a heavy multirotor flies to a
# waypoint from hover with mild overshoot under these, and the optimizer tightens overshoot + time.
# The validation of kp_z/kd_z ran on the legacy declared action scale, T/W 1.9; the scale is now the
# plant's USD-derived T/W, 2.221 for the astro-max, so both scale by 1.9/2.221 = 0.8554: the
# closed altitude loop is exactly the validated one, re-expressed in the true action scale.
DEFAULT_GAINS = np.array([0.2566, 0.2139, 0.18, 0.35, 8.0, 2.5, 1.0], dtype=np.float32)
