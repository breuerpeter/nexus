"""The joints' motors turn each joint's current command into the torque on its joint, and emit the current it draws,
for the joints of one vehicle, as a run builds them, or of many at once.
"""

import newton
import numpy as np
import pytest
import warp as wp

from nexus_uav._src.vehicle.actuators.joint_motor import JointMotor, JointMotors

pytestmark = pytest.mark.usefixtures("warp_cpu")

DOFS = [[6, 7], [14, 15]]  # each vehicle's two motor joints' velocity degrees of freedom, sixteen wide in all
JOINT = "/vehicle/arm_joint"


class _Control:
    """Newton's control inputs the motors write, sixteen degrees of freedom wide."""

    def __init__(self):
        self.joint_f = wp.zeros(16, dtype=float)


def _motors(dofs=DOFS) -> JointMotors:
    """Two motors per vehicle on the controls' fifth and sixth values: 0.05 N·m/A at 2 A and 0.1 N·m/A at 4 A."""
    return JointMotors(
        joint_dofs=wp.array(dofs, dtype=wp.int32),
        channels=[4, 5],
        torque_constant=[0.05, 0.1],
        current_limit=[2.0, 4.0],
    )


def _step(motors: JointMotors, commands, stale: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Step `motors` once over `commands`, each vehicle's fifth and sixth controls values, with `stale` in the
    current's buffer from the step before. Returns each motor's joint effort and its current, in order.
    """
    controls = np.zeros((len(commands), 6), dtype=np.float32)
    controls[:, 4:] = commands
    control, current = _Control(), wp.full((len(commands), 2), stale, dtype=float)
    motors.step(wp.array(controls, dtype=float), control, current)
    flat = [d for dofs in DOFS[: len(commands)] for d in dofs]
    return control.joint_f.numpy()[flat], current.numpy().reshape(-1)


def test_each_motor_makes_the_torque_of_its_current_command_and_emits_its_current():
    """A motor's torque on its joint is `Kt·u·currentLimit` for a command from −1 to 1 of either sign, and the
    current it emits is `u·currentLimit`, for the joints of many vehicles in one step.

    Given the joints of two vehicles, each with a motor of 0.05 N·m/A at 2 A and one of 0.1 N·m/A at 4 A, when the
    motors step once at 0.5 and −0.25, and −1 and 1, then the torques are 0.05, −0.1, −0.1 and 0.4 N·m and the
    currents 1, −1, −2 and 4 A.
    """
    torque, current = _step(_motors(), [[0.5, -0.25], [-1.0, 1.0]])

    np.testing.assert_allclose([*torque, *current], [0.05, -0.1, -0.1, 0.4, 1.0, -1.0, -2.0, 4.0], rtol=1e-6)


def test_a_command_beyond_one_makes_the_torque_and_current_of_one():
    """A command beyond ±1 makes the torque and the current of ±1.

    Given one vehicle's two motors, 0.05 N·m/A at 2 A and 0.1 N·m/A at 4 A, when they step at 3 and −3, then the
    torques are 0.1 and −0.4 N·m and the currents 2 and −4 A, those of 1 and −1.
    """
    torque, current = _step(_motors(DOFS[:1]), [[3.0, -3.0]])

    np.testing.assert_allclose([*torque, *current], [0.1, -0.4, 2.0, -4.0], rtol=1e-6)


def test_a_nan_command_makes_no_torque_and_no_current():
    """A NaN command makes no torque and no current.

    Given one vehicle's two motors, whose step before left 9 A in each current, when they step at a NaN command
    each, then each joint's effort and each current is 0.
    """
    torque, current = _step(_motors(DOFS[:1]), [[np.nan, np.nan]], stale=9.0)

    assert [*torque, *current] == [0.0, 0.0, 0.0, 0.0]


def _arm():
    """A model of one link on a revolute joint, labeled as the prim the tests name."""
    builder = newton.ModelBuilder()
    link = builder.add_link(mass=0.1, inertia=wp.mat33(np.eye(3) * 1e-4))
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z, label=JOINT)
    builder.add_articulation([joint])
    return builder.finalize()


@pytest.mark.parametrize(
    ("attribute", "values"),
    [
        ("nexus:torqueConstant", {"torque_constant": 0.0, "current_limit": 2.0, "channel": 0}),
        ("nexus:torqueConstant", {"torque_constant": -0.05, "current_limit": 2.0, "channel": 0}),
        ("nexus:currentLimit", {"torque_constant": 0.05, "current_limit": 0.0, "channel": 0}),
        ("nexus:currentLimit", {"torque_constant": 0.05, "current_limit": -2.0, "channel": 0}),
        ("nexus:channel", {"torque_constant": 0.05, "current_limit": 2.0, "channel": -1}),
    ],
)
def test_a_constant_at_or_below_zero_or_a_channel_below_zero_fails_and_names_the_prim_and_the_attribute(
    attribute, values
):
    """A constant at or below 0, or a channel below 0, fails the build, the message naming the prim and the
    attribute.

    Given a model of one link on a revolute joint, and in turn a torque constant of 0 and of −0.05 N·m/A, a current
    limit of 0 and of −2 A, and the channel −1, when the test makes the motor of that joint, then it raises, and the
    message names the joint's prim and the attribute.
    """
    with pytest.raises(ValueError) as e:
        JointMotor(_arm(), JOINT, **values)

    assert [s in str(e.value) for s in (JOINT, attribute)] == [True, True]
