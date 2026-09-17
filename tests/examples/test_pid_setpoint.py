"""PidController.accept_setpoint: writes the goal *in place* into the host buffer and, when bound, the
shared persistent device goal buffer via .assign, which is capture-safe, so the WarpObservationSensor's
obs kernel picks up the new goal on the next captured replay, see control-surface-api.md.
"""

import numpy as np
import pytest

from nexus._src.core.schema import PositionGoal, Waypoints
from nexus.examples.controllers.pid import PidController


def test_accept_setpoint_mutates_host_goal_in_place():
    c = PidController(goal_w=(0.0, 0.0, 1.0), thrust_to_weight=1.9)
    buf = c.goal_w
    c.accept_setpoint(PositionGoal(pos=(1.0, 2.0, 3.0)))
    assert c.goal_w is buf  # same array object, static address
    np.testing.assert_allclose(c.goal_w, [1.0, 2.0, 3.0])


def test_accept_setpoint_rejects_unsupported_variant():
    with pytest.raises(TypeError):
        PidController(goal_w=(0.0, 0.0, 1.0), thrust_to_weight=1.9).accept_setpoint(Waypoints(points=[(0.0, 0.0, 1.0)]))


def test_accept_setpoint_assigns_the_bound_device_goal_buffer():
    wp = pytest.importorskip("warp")
    wp.set_device("cpu")
    from nexus.examples._lib.observation import WarpObservationSensor

    sensor = WarpObservationSensor(goal_w=(0.0, 0.0, 1.0))
    c = PidController(goal_w=(0.0, 0.0, 1.0), thrust_to_weight=1.9)
    c.bind_goal_buffer(sensor.goal)  # the build wires the sensor's persistent device buffer
    c.accept_setpoint(PositionGoal(pos=(4.0, 5.0, 6.0)))
    np.testing.assert_allclose(sensor.goal.numpy().reshape(-1), [4.0, 5.0, 6.0], atol=1e-6)
