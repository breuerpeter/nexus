"""Observation-bridge frame math: pure NumPy, no torch/warp/Newton, on CPU and deterministic."""

import math

import numpy as np

from nexus.examples._lib.observation import (
    OBS_DIM,
    POLICY_OBS_DIM,
    build_observation,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_matrix,
)

IDENT = (0.0, 0.0, 0.0, 1.0)  # xyzw identity


def test_obs_dim():
    assert build_observation((0, 0, 0), IDENT, (0, 0, 0), (0, 0, 0), (0, 0, 0)).shape == (OBS_DIM,)


def test_prev_action_appended():
    """The policy obs = the 12-D kinematic obs with the last action appended, POLICY_OBS_DIM wide.
    This is the single-source rule: deploy concatenates exactly what training appends, so the
    augmented obs can't drift across the FR-7 seam.
    """
    prev = (0.1, -0.2, 0.3, 0.4)
    args = ((0, 0, 1), IDENT, (1, 2, 3), (0.1, 0.2, 0.3), (5, 0, 1))
    kin = build_observation(*args)
    full = build_observation(*args, prev_action=prev)
    assert full.shape == (POLICY_OBS_DIM,)
    np.testing.assert_allclose(full[:OBS_DIM], kin, atol=1e-6)  # kinematic part unchanged
    np.testing.assert_allclose(full[OBS_DIM:], prev, atol=1e-6)  # last action appended verbatim


def test_level_passthrough_gravity_and_goal():
    # Identity attitude -> body frame == world frame.
    o = build_observation(
        pos_w=(0, 0, 1), quat_xyzw=IDENT, lin_vel_w=(1, 2, 3), ang_vel_w=(0.1, 0.2, 0.3), goal_w=(5, 0, 1)
    )
    np.testing.assert_allclose(o[0:3], [1, 2, 3], atol=1e-6)  # lin_vel_b
    np.testing.assert_allclose(o[3:6], [0.1, 0.2, 0.3], atol=1e-6)  # ang_vel_b
    np.testing.assert_allclose(o[6:9], [0, 0, -1], atol=1e-6)  # gravity points down in body
    np.testing.assert_allclose(o[9:12], [5, 0, 0], atol=1e-6)  # goal 5 m along world +x -> body +x


def test_yaw_90_rotates_goal_into_body():
    s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
    q = (0, 0, s, c)  # +90 deg yaw about world z
    o = build_observation((0, 0, 0), q, (0, 0, 0), (0, 0, 0), goal_w=(1, 0, 0))
    # body +x now points to world +y, so a world +x goal is at body -y
    np.testing.assert_allclose(o[9:12], [0, -1, 0], atol=1e-6)


def test_gravity_unit_norm_under_roll():
    s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
    q = (s, 0, 0, c)  # 90 deg roll about body x
    o = build_observation((0, 0, 0), q, (0, 0, 0), (0, 0, 0), (0, 0, 0))
    np.testing.assert_allclose(np.linalg.norm(o[6:9]), 1.0, atol=1e-6)
    # roll about x leaves gravity's x-component zero, tilts it into y/z
    np.testing.assert_allclose(o[6], 0.0, atol=1e-6)


def test_rotation_matrix_orthonormal():
    R = quat_xyzw_to_matrix((0.1, 0.2, 0.3, 0.9))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_quat_wxyz_to_xyzw():
    assert quat_wxyz_to_xyzw((1, 2, 3, 4)) == (2, 3, 4, 1)
