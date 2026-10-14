"""The rotors' command element writes each rotor's speed target and drag feedforward into Newton's control inputs,
for the rotors of one vehicle, as a run builds it, or of many at once, as the RL task steps them.
"""

import numpy as np
import pytest
import warp as wp

from nexus_uav._src.vehicle.commands.rotors import RotorCommand
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS

pytestmark = pytest.mark.usefixtures("warp_cpu")


class _Control:
    """Newton's two control inputs the command element writes, ten degrees of freedom wide."""

    def __init__(self):
        self.joint_target_qd = wp.zeros(10, dtype=float)
        self.joint_act = wp.zeros(10, dtype=float)


def test_each_rotor_of_each_vehicle_takes_its_own_command():
    """Each rotor of each vehicle takes its own command.

    Given the rotors of two vehicles at degrees of freedom [[1, 2], [6, 7]], a no-load speed of 100 rad/s, a drag
    of 0.1 and a thrust map of 1e-4 N·s², when the command element writes the commands [[0.5, 2.0], [-1.0, 0.25]],
    then the speed targets at 1, 2, 6 and 7 are 50, 100, 0 and 25 rad/s, and the drag feedforwards 0.025, 0.1,
    0 and 0.00625 N·m.
    """
    control = _Control()
    command = RotorCommand.for_rotors(
        rotor_vel_dofs=wp.array([[1, 2], [6, 7]], dtype=wp.int32),
        control=control,
        ct=1e-4 / RPM_PER_RADS**2,
        cd=0.1,
        rpm_max=100.0 * RPM_PER_RADS,
    )

    command.write(wp.array([[0.5, 2.0], [-1.0, 0.25]], dtype=float))

    written = (control.joint_target_qd.numpy()[[1, 2, 6, 7]], control.joint_act.numpy()[[1, 2, 6, 7]])
    np.testing.assert_allclose(written, ([50.0, 100.0, 0.0, 25.0], [0.025, 0.1, 0.0, 0.00625]), rtol=1e-5)
