"""The single source of truth for the goal-relative observation: *one* implementation of
observation from state, reused by the `nexus-rl` training task, the trained-policy deploy
controller, and the Proportional Integral Derivative (PID) and design-opt examples, so the observation
the policy trains on is **the same, bit for bit,** as the one it gets at deploy, the prerequisite for a
bit-for-bit equal Isaac-Lab↔Orchestrator rollout.

The Isaac-Lab `Isaac-Quadcopter-Direct-v0` observation, in order:

    obs[0:3]  = root_lin_vel_b        body-frame linear velocity   [m/s]
    obs[3:6]  = root_ang_vel_b        body-frame angular velocity  [rad/s]
    obs[6:9]  = projected_gravity_b   unit gravity direction in body frame  [-]
    obs[9:12] = desired_pos_b         goal position relative to the body, body frame  [m]

All four are ``R(q)^T · <world quantity>``, over the **native-Newton xyzw** quaternion order that the
state reads commit to. Two implementations of the same math live here on purpose, parity-tested
against each other in tests/examples:

* :func:`observation_from_state`, torch, batched: the training path, plus its numpy convenience
  :func:`build_observation`, the deploy path. The torch import is lazy so warp-only consumers, such
  as the PID example, never pull it.
* :func:`observation_kernel` / :class:`WarpObservationSensor`, the device-native Warp twin: what
  lets the obs join a ``wp.Tape`` / Compute Unified Device Architecture (CUDA) graph, the
  differentiable and captured strategies.
"""

from __future__ import annotations

import numpy as np
import warp as wp

OBS_DIM = 12  # kinematic observation; the PID determinism gate / WarpObservationSensor use this dim
ACTION_DIM = 4
# The trained-policy observation is the kinematic obs plus the **last action**, a Collective Thrust and
# Body Rate (CTBR) command, the load-bearing fix for the per-rotor task: the rotor speed lags ~33 ms
# behind the command and stays hidden from the obs, so the policy needs its own last action to infer the
# hidden motor state, by feeding the last action into the obs. Kept separate from
# OBS_DIM so the warp/PID path stays a clean 12-D.
POLICY_OBS_DIM = OBS_DIM + ACTION_DIM


def quat_xyzw_to_matrix(q):
    """Rotation matrix R, body to world, from an ``xyzw`` quaternion. ``R @ v_body = v_world``.

    Accepts shape ``(..., 4)`` and returns ``(..., 3, 3)``, batched or single, torch.
    """
    import torch

    q = torch.as_tensor(q, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def observation_from_state(
    pos_w,
    quat_xyzw,
    lin_vel_w,
    ang_vel_w,
    goal_w,
    gravity_dir_w=(0.0, 0.0, -1.0),
    prev_action=None,
):
    """Canonical observation builder, batched, torch. Shapes ``(..., 3)`` / ``(..., 4)`` ->
    ``(..., 12)``. This is the one function both training and deploy call, the single source.

    ``prev_action``, shape ``(..., 4)``: when given, the builder appends the last CTBR action ->
    ``(..., 16)``, the trained-policy observation, :data:`POLICY_OBS_DIM`. The append is the single
    source for both training, where the env passes its last applied action, and deploy, where the
    controller passes its last emitted action, so the augmented obs can't drift across the seam.
    """
    import torch

    R = quat_xyzw_to_matrix(quat_xyzw)
    Rt = R.transpose(-1, -2)  # world -> body

    def to_body(v):
        return (Rt @ torch.as_tensor(v, dtype=torch.float32, device=R.device).unsqueeze(-1)).squeeze(-1)

    grav = torch.as_tensor(gravity_dir_w, dtype=torch.float32, device=R.device).expand_as(
        torch.as_tensor(pos_w, dtype=torch.float32)
    )
    lin_b = to_body(lin_vel_w)
    ang_b = to_body(ang_vel_w)
    grav_b = to_body(grav)
    goal_b = to_body(
        torch.as_tensor(goal_w, dtype=torch.float32, device=R.device)
        - torch.as_tensor(pos_w, dtype=torch.float32, device=R.device)
    )
    obs = torch.cat([lin_b, ang_b, grav_b, goal_b], dim=-1)
    if prev_action is not None:
        obs = torch.cat([obs, torch.as_tensor(prev_action, dtype=torch.float32, device=R.device)], dim=-1)
    return obs


def build_observation(
    pos_w, quat_xyzw, lin_vel_w, ang_vel_w, goal_w, gravity_dir_w=(0.0, 0.0, -1.0), prev_action=None
) -> np.ndarray:
    """NumPy convenience over :func:`observation_from_state` for a single env: same math, for the
    deploy path / tests. Returns a length-12 float32 array, or length-16 with ``prev_action``.
    """
    import torch

    obs = observation_from_state(
        torch.as_tensor(np.asarray(pos_w), dtype=torch.float32),
        torch.as_tensor(np.asarray(quat_xyzw), dtype=torch.float32),
        torch.as_tensor(np.asarray(lin_vel_w), dtype=torch.float32),
        torch.as_tensor(np.asarray(ang_vel_w), dtype=torch.float32),
        torch.as_tensor(np.asarray(goal_w), dtype=torch.float32),
        gravity_dir_w,
        prev_action=None if prev_action is None else torch.as_tensor(np.asarray(prev_action), dtype=torch.float32),
    )
    return obs.cpu().numpy().astype(np.float32)


def quat_wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    """Reorder a ``wxyz`` quaternion, the MAVLink / Isaac Universal Scene Description (USD) scalar-first
    order, to the neutral ``xyzw``.
    """
    w, x, y, z = (float(c) for c in q)
    return (x, y, z, w)


@wp.kernel
def observation_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    goal: wp.array(dtype=wp.vec3),  # length-1 persistent goal buffer, read as goal[0], updated via .assign,
    body_index: int,  # so a captured graph picks up a new goal on the next replay: the capture contract
    obs: wp.array(dtype=float),  # (12,) single-env observation
):
    tf = body_q[body_index]
    pos = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    lin_w = wp.spatial_top(body_qd[body_index])  # Newton body_qd = [linear, angular]
    ang_w = wp.spatial_bottom(body_qd[body_index])
    lin_b = wp.quat_rotate_inv(q, lin_w)
    ang_b = wp.quat_rotate_inv(q, ang_w)
    grav_b = wp.quat_rotate_inv(q, wp.vec3(0.0, 0.0, -1.0))
    goal_b = wp.quat_rotate_inv(q, goal[0] - pos)
    obs[0] = lin_b[0]
    obs[1] = lin_b[1]
    obs[2] = lin_b[2]
    obs[3] = ang_b[0]
    obs[4] = ang_b[1]
    obs[5] = ang_b[2]
    obs[6] = grav_b[0]
    obs[7] = grav_b[1]
    obs[8] = grav_b[2]
    obs[9] = goal_b[0]
    obs[10] = goal_b[1]
    obs[11] = goal_b[2]


class WarpObservationSensor:
    """Device-native ground-truth observation sensor: writes the 12-D obs into a persistent
    ``(12,)`` Warp array from the live state's device arrays, so it records on a tape / joins a
    graph, the Warp twin of :func:`observation_from_state`.

    Args:
        goal_w: target world position [m], the waypoint / hover goal.
        body_index: articulation body the observation tracks; 0 is the base.
    """

    capturable = True

    def __init__(self, goal_w=(0.0, 0.0, 1.0), body_index: int = 0):
        # Persistent length-1 device goal buffer with a static address: the kernel reads goal[0], so a
        # controller's accept_setpoint can .assign() a new goal in place and the next captured replay
        # picks it up, with zero re-capture: the capture contract.
        self.goal = wp.array(np.asarray([goal_w], dtype=np.float32), dtype=wp.vec3)
        self.body_index = int(body_index)
        self._obs = wp.zeros(OBS_DIM, dtype=float)  # persistent buffer for the eager/capturable path

    def set_goal(self, pos) -> None:
        """Update the goal in place via ``.assign``: static address, capture-safe."""
        self.goal.assign(np.asarray([pos], dtype=np.float32))

    def sample_wp(self, state, out_obs: wp.array) -> wp.array:
        """Launch the obs kernel over ``state.body_q`` / ``state.body_qd`` into ``out_obs`` (12,)."""
        wp.launch(
            observation_kernel,
            dim=1,
            inputs=(state.body_q, state.body_qd, self.goal, self.body_index),
            outputs=(out_obs,),
        )
        return out_obs

    def sample(self, state, env, t, out) -> None:
        """``Sensor`` protocol: write the 12-D obs, a **Warp array**, into ``out.observation``, so an
        in-process controller's ``exchange`` consumes it on-device and the loop stays capturable /
        tape-able. Reuses a persistent buffer with a static address for capture.
        """
        out.observation = self.sample_wp(state, self._obs)


__all__ = [
    "ACTION_DIM",
    "OBS_DIM",
    "POLICY_OBS_DIM",
    "WarpObservationSensor",
    "build_observation",
    "observation_from_state",
    "observation_kernel",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_matrix",
]
