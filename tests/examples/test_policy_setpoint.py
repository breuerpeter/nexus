"""The policy controller's thin control surface: accept_setpoint mutates the goal buffer *in place*,
and the controller's own observation builder picks it up: the control-surface-api.md contract, where the
controller owns the goal + action-history buffers and folds them into the obs itself.
"""

import numpy as np
import pytest

from nexus._src.core.schema import PositionGoal, Waypoints
from nexus.examples.controllers.policy.controller import TrainedPolicyController

torch = pytest.importorskip("torch")  # the obs builder is the torch single source


class _FakeState:
    """A one-body level hover at (0, 0, 1): a numpy-backed stand-in for newton.State."""

    class _Arr:
        def __init__(self, v):
            self._v = np.asarray(v, dtype=np.float32)

        def numpy(self):
            return self._v

    def __init__(self):
        self.body_q = self._Arr([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
        self.body_qd = self._Arr([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def _controller():
    # No connect()/policy load needed: accept_setpoint only touches the goal buffer.
    return TrainedPolicyController(policy_path="unused.pt", goal_w=(0.0, 0.0, 1.0))


def test_accept_setpoint_mutates_goal_in_place():
    c = _controller()
    buf = c.goal_w
    c.accept_setpoint(PositionGoal(pos=(1.0, 2.0, 3.0)))
    assert c.goal_w is buf  # same array object, static address: capture-safe value-mutation
    np.testing.assert_allclose(c.goal_w, [1.0, 2.0, 3.0])


def test_accept_setpoint_reaches_the_next_observation():
    c = _controller()
    c.accept_setpoint(PositionGoal(pos=(0.0, 0.0, 2.0)))
    obs = c.obs_from_state(_FakeState())  # level hover at (0,0,1); goal (0,0,2) -> goal_rel_b (0,0,1)
    np.testing.assert_allclose(obs[9:12], [0.0, 0.0, 1.0], atol=1e-6)
    c._prev_action[:] = [0.1, 0.2, 0.3, 0.4]  # an act() updates this in place
    obs = c.obs_from_state(_FakeState())
    np.testing.assert_allclose(obs[12:16], [0.1, 0.2, 0.3, 0.4], atol=1e-6)  # folded into the next obs


def test_exchange_builds_obs_from_meas_state(tmp_path):
    import torch

    class Echo(torch.nn.Module):
        def forward(self, x):
            return x[:, 12:16]

    path = tmp_path / "echo.pt"
    torch.jit.script(Echo()).save(str(path))
    c = TrainedPolicyController(policy_path=str(path), goal_w=(0.0, 0.0, 2.0))
    c.connect()
    c._prev_action[:] = [0.5, -0.5, 0.25, 0.1]

    class _Meas:
        state = _FakeState()

    out = c.exchange(_Meas(), t=0.0)  # obs built from meas.state, the StateSensor passthrough
    np.testing.assert_allclose(np.asarray(out.command), [0.5, -0.5, 0.25, 0.1], atol=1e-6)


def test_accept_setpoint_rejects_unsupported_variant():
    c = _controller()
    with pytest.raises(TypeError):
        c.accept_setpoint(Waypoints(points=[(0.0, 0.0, 1.0)]))
