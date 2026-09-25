"""PidController + the shared Proportional Integral Derivative (PID) law, the FR-7 built-in simple controller.

Covers the controller plumbing, on CPU with no Newton needed, and the load-bearing invariant that the
Warp ``pid_law`` kernel and the NumPy ``pid_action_np`` mirror compute the same action: they
are the single source of truth shared by the eager controller and the differentiable rollout.
"""

import numpy as np
import pytest

from nexus._src.core.schema import Controls, Measurement
from nexus.examples.controllers.pid import (
    DEFAULT_GAINS,
    NUM_GAINS,
    PidController,
    hover_action,
    pid_action_np,
)


class _FakeState:
    """State-like: body_q = [p, q_xyzw], xyzw with 180°-X = a level Forward Right Down (FRD) body, body_qd = [lin, ang]."""

    def __init__(self, pos=(0.0, 0.0, 1.0)):
        import warp as wp

        bq = np.array([[*pos, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.body_q = wp.array(bq, dtype=wp.transform)
        self.body_qd = wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.spatial_vector)


def test_act_shape_and_hover_bias():
    c = PidController(goal_w=(0, 0, 1), thrust_to_weight=1.9)
    # at the goal, level, at rest -> action is pure hover bias on thrust, zero moments
    obs = np.zeros(12, np.float32)
    obs[6:9] = [0, 0, 1]  # level gravity for an FRD body: gravity along +body-z, which points down
    ctrl = c.act(obs)
    assert isinstance(ctrl, Controls)
    a = np.asarray(ctrl.command)
    assert a.shape == (4,)
    assert a[0] == pytest.approx(hover_action(1.9), abs=1e-6)
    np.testing.assert_allclose(a[1:], 0.0, atol=1e-6)


def test_altitude_error_increases_thrust():
    c = PidController(goal_w=(0, 0, 2), thrust_to_weight=1.9)
    # FRD body, below the goal: level gravity is +body-z, and a goal 1 m up is along −body-z → thrust > hover
    obs = np.zeros(12, np.float32)
    obs[6:9] = [0, 0, 1]
    obs[9:12] = [0, 0, -1.0]  # goal 1 m up, −body-z for the FRD body
    a = np.asarray(c.act(obs).command)
    assert a[0] > hover_action(1.9)


def test_exchange_reads_observation_or_provider():
    c = PidController(goal_w=(0, 0, 1), thrust_to_weight=1.9)
    meas = Measurement()
    with pytest.raises(RuntimeError):
        c.exchange(meas, None, None)  # no observation, no provider
    c.bind_state_provider(lambda: ((0, 0, 1), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0)))
    assert np.asarray(c.exchange(meas, None, None).command).shape == (4,)
    # and via meas.observation, the WarpObservationSensor plane
    meas.observation = np.zeros(12, np.float32)
    meas.observation[6:9] = [0, 0, 1]  # FRD level gravity
    assert np.asarray(c.exchange(meas, None, None).command).shape == (4,)


def test_act_from_state_matches_obs():
    c = PidController(goal_w=(0, 0, 2), thrust_to_weight=1.9)
    a = np.asarray(c.act_from_state(_FakeState(pos=(0, 0, 1))).command)
    assert a.shape == (4,) and a[0] > hover_action(1.9)  # 1 m below goal -> climb


def test_default_gains_shape():
    assert DEFAULT_GAINS.shape == (NUM_GAINS,)


@pytest.mark.usefixtures("warp_cpu")
def test_law_warp_matches_numpy():
    """The Warp kernel and the NumPy mirror *must* agree: single source of truth."""
    wp = pytest.importorskip("warp")
    from nexus.examples.controllers.pid import pid_law

    rng = np.random.default_rng(0)
    hover = hover_action(1.9)
    for _ in range(20):
        obs = rng.standard_normal(12).astype(np.float32)
        gains = np.abs(rng.standard_normal(NUM_GAINS)).astype(np.float32)
        ref = pid_action_np(obs, gains, hover)
        obs_wp = wp.array(obs, dtype=float)
        g_wp = wp.array(gains, dtype=float)
        out = wp.zeros(4, dtype=float)
        wp.launch(pid_law, dim=1, inputs=(obs_wp, g_wp, hover), outputs=(out,))
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)
