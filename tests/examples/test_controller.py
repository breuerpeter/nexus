"""TrainedPolicyController plumbing: uses a small scripted stand-in policy, with no GPU/Newton."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nexus._src.core.schema import Controls  # noqa: E402
from nexus.examples.controllers.policy.controller import TrainedPolicyController  # noqa: E402


def _make_stub_policy(tmp_path):
    """A scripted obs[1,12] -> action[1,4] policy that doubles the first 4 obs, to exercise clamp."""

    class Stub(torch.nn.Module):
        def forward(self, x):
            return x[:, :4] * 2.0

    path = tmp_path / "stub_policy.pt"
    torch.jit.script(Stub()).save(str(path))
    return str(path)


class _FakeState:
    """State-like: body_q = [p, q_xyzw], an xyzw identity, body_qd = [lin, ang]."""

    def __init__(self):
        import warp as wp

        self.body_q = wp.array(np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)
        self.body_qd = wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.spatial_vector)


def test_act_shape_and_clamp(tmp_path):
    c = TrainedPolicyController(policy_path=_make_stub_policy(tmp_path), goal_w=(0, 0, 1))
    c.connect()
    obs = np.zeros(12, np.float32)
    obs[0] = 10.0  # stub doubles -> 20 -> must clamp to 1.0
    ctrl = c.act(obs)
    assert isinstance(ctrl, Controls)
    a = np.asarray(ctrl.command)
    assert a.shape == (4,)
    assert a.max() <= 1.0 and a.min() >= -1.0
    assert a[0] == pytest.approx(1.0)


def test_obs_from_state_goal_relative(tmp_path):
    c = TrainedPolicyController(policy_path=_make_stub_policy(tmp_path), goal_w=(0, 0, 2))
    o = c.obs_from_state(_FakeState())
    assert o.shape == (16,)  # complete obs: 12-D kinematic + 4-D prev_action, folded in one place
    # goal (0,0,2) - pos (0,0,1) = (0,0,1); level attitude -> body == world
    np.testing.assert_allclose(o[9:12], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(o[12:16], 0.0, atol=1e-6)  # prev_action zeros at construction


def test_exchange_requires_observation(tmp_path):
    c = TrainedPolicyController(policy_path=_make_stub_policy(tmp_path))
    c.connect()
    with pytest.raises(RuntimeError):
        c.exchange(object(), None, None)
    # once a state provider binds, exchange runs
    c.bind_state_provider(lambda: ((0, 0, 1), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0)))
    ctrl = c.exchange(object(), None, None)
    assert np.asarray(ctrl.command).shape == (4,)


def test_prev_action_folded_into_obs_by_the_obs_function(tmp_path):
    """The obs function folds the last Collective Thrust and Body Rates (CTBR) action into the obs at
    [12:16]. That's the single obs-construction site, obs_from_view / the sensor; act() does *not* reshape
    it in, and act() updates the shared prev_action buffer *in place*. An echo policy returns obs[12:16],
    proving the prev action landed where training places it: the deploy twin of
    observation_from_state(..., prev_action=self._actions).
    """

    class Echo(torch.nn.Module):
        def forward(self, x):
            return x[:, 12:16]

    path = tmp_path / "echo.pt"
    torch.jit.script(Echo()).save(str(path))
    c = TrainedPolicyController(policy_path=str(path), goal_w=(0, 0, 1))
    c.connect()
    c._prev_action[:] = [0.5, -0.5, 0.25, 0.1]  # in place, the shared buffer
    obs = c.obs_from_state(_FakeState())  # the obs function folds prev_action into [12:16]
    np.testing.assert_allclose(obs[12:16], [0.5, -0.5, 0.25, 0.1], atol=1e-6)
    out = np.asarray(c.act(obs).command)  # Echo returns obs[12:16] -> the prev action
    np.testing.assert_allclose(out, [0.5, -0.5, 0.25, 0.1], atol=1e-6)
    np.testing.assert_allclose(c._prev_action, [0.5, -0.5, 0.25, 0.1], atol=1e-6)  # updated in place


def test_controller_builds_observation_from_state():
    c = TrainedPolicyController(policy_path="unused.pt", goal_w=(0, 0, 2))
    obs = c.obs_from_state(_FakeState())
    assert obs.shape == (16,)  # complete policy obs: kinematic + prev_action
    # level attitude, pos (0,0,1), goal (0,0,2) -> goal_rel_b (0,0,1)
    np.testing.assert_allclose(obs[9:12], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(obs[12:16], 0.0, atol=1e-6)  # default prev_action zeros
