"""The device-native observation, ``WarpObservationSensor``, must reproduce the host torch reference,
``observation_from_state``: one 12-D obs, single source of truth across the eager/capturable/
differentiable paths. This was part of the design-opt test suite; the obs sensor is framework, the
rollouts that used it moved to ``examples/``.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import newton
import warp as wp

wp.set_device("cpu")


def _single_body():
    b = newton.ModelBuilder()
    body = b.add_body(label="body", mass=9.0)
    b.add_shape_box(body, hx=0.2, hy=0.2, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    b.add_joint_free(child=body)
    return b.finalize()


def test_warp_obs_matches_torch_reference():
    from nexus.examples._lib.observation import WarpObservationSensor, observation_from_state

    state = _single_body().state()
    bq = state.body_q.numpy()
    bq[0, 0:3] = [0.3, -0.2, 1.1]
    bq[0, 3:7] = [0.0, 0.0, 0.0, 1.0]  # level, an xyzw identity
    state.body_q.assign(bq)
    bqd = state.body_qd.numpy()
    bqd[0, :] = [0.5, -0.1, 0.2, 0.0, 0.0, 0.0]  # [lin, ang]
    state.body_qd.assign(bqd)

    goal = (1.0, 0.5, 1.5)
    obs_wp = wp.zeros(12, dtype=float)
    WarpObservationSensor(goal_w=goal).sample_wp(state, obs_wp)
    wp.synchronize()

    ref = observation_from_state(bq[0, 0:3], bq[0, 3:7], bqd[0, 0:3], bqd[0, 3:6], np.asarray(goal)).numpy()
    np.testing.assert_allclose(obs_wp.numpy(), ref, atol=1e-5)
