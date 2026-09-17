"""FlatnessReference, the operator's ruckig + differential-flatness planner: plans a valid jerk-limited
trajectory through the waypoints and lifts it to the full flat state. CPU-only, with ruckig + numpy.
"""

import numpy as np
import pytest

pytest.importorskip("ruckig")

from nexus.examples._lib.reference import FlatnessReference


def _ref():
    r = FlatnessReference([(2.0, 0.5, 3.5), (1.0, 2.5, 3.0), (3.0, 2.0, 4.0)], mass=1.0)
    r.set_start((0.0, 0.0, 2.0))
    return r


def test_duration_is_positive_and_builds_lazily():
    r = _ref()
    assert r._segments is None  # not built until needed
    assert r.duration > 0.0
    assert r._segments is not None


def test_reference_path_starts_at_start_and_ends_at_goal():
    r = _ref()
    path = r.reference_path(200)
    assert path.shape == (200, 3)
    np.testing.assert_allclose(path[0], [0.0, 0.0, 2.0], atol=1e-3)  # starts at the start
    np.testing.assert_allclose(path[-1], [3.0, 2.0, 4.0], atol=1e-3)  # settles on the final goal


def test_reference_holds_at_goal_past_the_duration():
    r = _ref()
    p_end, *_ = r._reference_at(r.duration)
    p_past, v_past, _a = r._reference_at(r.duration + 5.0)
    np.testing.assert_allclose(p_past, p_end, atol=1e-6)  # holds the goal after the trajectory ends
    np.testing.assert_allclose(v_past, np.zeros(3), atol=1e-6)  # at rest


def test_flat_state_at_returns_full_state():
    r = _ref()
    pos, quat, vel, omega, thrust = r.flat_state_at(r.duration * 0.5)
    assert pos.shape == (3,) and quat.shape == (4,) and vel.shape == (3,) and omega.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)  # unit quaternion
    assert thrust > 0.0  # collective thrust feedforward: ≈ m·g hovering, more when accelerating up
