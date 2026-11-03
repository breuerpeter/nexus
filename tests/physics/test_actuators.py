"""The model's Newton actuators add their joints' torque, and a reset zeroes the state of the actuators on the
degrees of freedom it names, and of no other.

Real Newton builds on the Warp CPU backend: two links on revolute joints off the world, each held off its zero
target and driven by a Newton motor with an integral term alone, whose torque shows the motor's state.
"""

import numpy as np
import pytest

pytest.importorskip("newton")

import newton
import warp as wp
from newton.actuators import Actuator, ControllerPID

from nexus_uav._src.physics.actuators import Actuators

pytestmark = pytest.mark.usefixtures("warp_cpu")

DT = 0.01  # [s]
ANGLE = 0.5  # each joint's angle off its zero target [rad]
KI = 2.0  # each motor's integral gain [N·m/(rad·s)]


def _two_joints():
    """Two links on revolute joints off the world, both 0.5 rad off their targets, each driven by an integral-only
    Newton motor: the model, its state, its control and the joints' degrees of freedom.
    """
    builder = newton.ModelBuilder()
    joints = []
    for x in (0.0, 1.0):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        joint = builder.add_joint_revolute(-1, link, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
        joints.append(joint)
    model = builder.finalize()
    dofs = model.joint_qd_start.numpy()[joints]
    model.actuators = [
        Actuator(
            wp.array(dofs, dtype=wp.uint32),
            ControllerPID(
                kp=wp.zeros(2, dtype=float),
                ki=wp.full(2, KI, dtype=float),
                kd=wp.zeros(2, dtype=float),
                integral_max=wp.full(2, 1e9, dtype=float),
            ),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
    ]
    state = model.state()
    q = state.joint_q.numpy()
    q[model.joint_q_start.numpy()[joints]] = ANGLE
    state.joint_q.assign(q)
    return model, state, model.control(), dofs


def test_a_reset_zeroes_the_state_of_the_motors_it_names_alone():
    """A reset zeroes the state of the motors it names, and of no other.

    Given two joints 0.5 rad off their targets, each under an integral-only motor of gain 2, when the motors step
    twice, the second joint's motor resets, and they step once more, each step over a zeroed joint torque as the
    physics clear stage leaves it, then the first joint's torque is three steps' integral, -0.03 N·m, and the
    second's one step's, -0.01 N·m.
    """
    model, state, control, dofs = _two_joints()
    motors = Actuators(model)
    mask = np.zeros(model.joint_dof_count, dtype=bool)
    mask[dofs[1]] = True

    for reset in (False, False, True):
        if reset:
            motors.reset(wp.array(mask, dtype=wp.bool))
        control.joint_f.zero_()
        motors.step(state, control, DT)

    torque = control.joint_f.numpy()[dofs]
    np.testing.assert_allclose(torque, [-3 * KI * ANGLE * DT, -KI * ANGLE * DT], rtol=1e-5)


def test_the_actuators_add_their_torque_to_what_the_joint_torque_holds():
    """The actuators add their torque to what the joint torque holds, so a rotor's motor and a servo on one
    model accumulate into it, and the physics clear stage alone zeroes it.

    Given two joints under integral-only motors and a joint torque of 5 N·m left on both, when the actuators
    step once, then each joint's torque is 5 N·m plus one step's integral, 4.99 N·m.
    """
    model, state, control, dofs = _two_joints()
    control.joint_f.fill_(5.0)

    Actuators(model).step(state, control, DT)

    np.testing.assert_allclose(control.joint_f.numpy()[dofs], [5.0 - KI * ANGLE * DT] * 2, rtol=1e-5)
