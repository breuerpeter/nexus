"""SamplingMPCController.accept_setpoint: writes the active target *in place* from a PositionGoal
via .assign, since the captured rollout reads the buffer's current contents, so no graph re-capture.
Uniform with policy/pid; the operator sequences the mission. CPU-only, with a small 2-rollout model.
"""

import numpy as np
import pytest

from nexus._src.core.schema import PositionGoal, Waypoints

pytest.importorskip("warp")
pytest.importorskip("newton")

pytestmark = pytest.mark.usefixtures("warp_cpu")


def _controller():
    import newton

    from nexus.examples.controllers.sampling_mpc import SamplingMPCController

    b = newton.ModelBuilder()
    for _ in range(2):  # 2 rollouts → 2 free single bodies
        body = b.add_body(mass=1.0)
        b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        b.add_joint_free(child=body)
    batch_model = b.finalize(requires_grad=True)
    offsets = np.array([[0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [-0.2, 0.2, 0.0]], dtype=np.float32)
    dirs = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)  # quad-X spin signs
    return SamplingMPCController(
        batch_model=batch_model,
        mass=1.0,
        rotor_offsets=offsets,
        turning_dirs=dirs,
        ct=1.0e-6,
        rpm_max=2000.0,
        goal_w=(0.0, 0.0, 1.0),
        dt=0.005,
        num_rollouts=2,
        motor_tau=0.033,
    )


def test_accept_setpoint_assigns_target_in_place():
    c = _controller()
    buf = c.target
    c.accept_setpoint(PositionGoal(pos=(1.0, 2.0, 3.0)))
    assert c.target is buf  # same persistent buffer, static address: .assign in place, capture-safe
    np.testing.assert_allclose(c.target.numpy().reshape(-1), [1.0, 2.0, 3.0], atol=1e-6)


def test_accept_setpoint_rejects_unsupported_variant():
    c = _controller()
    with pytest.raises(TypeError):
        c.accept_setpoint(Waypoints(points=[(0.0, 0.0, 1.0)]))
