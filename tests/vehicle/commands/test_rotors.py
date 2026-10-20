"""The rotors' ESCs write each rotor's speed target and drag feedforward into Newton's control inputs, each from
its own channel and its own values, for the rotors of one vehicle, as a run builds them, or of many at once, as
the RL task steps them.
"""

import numpy as np
import pytest
import warp as wp

from nexus_uav._src.vehicle.commands.esc import Escs
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS

pytestmark = pytest.mark.usefixtures("warp_cpu")


class _Control:
    """Newton's two control inputs the ESCs write, ten degrees of freedom wide."""

    def __init__(self):
        self.joint_target_qd = wp.zeros(10, dtype=float)
        self.joint_act = wp.zeros(10, dtype=float)


def test_each_rotor_of_each_vehicle_takes_its_own_command_from_its_channel_with_its_own_values():
    """Each rotor of each vehicle takes its own command from its channel, with its own values.

    Given the rotors of two vehicles at degrees of freedom [[1, 2], [6, 7]], the first rotor on channel 1 with a
    no-load speed of 100 rad/s, a drag of 0.1 and a thrust map of 1e-4 N·s², the second on channel 0 with a
    no-load speed of 200 rad/s, a drag of 0.2 and a thrust map of 2e-4 N·s², when the ESCs write the commands
    [[0.5, 2.0], [-1.0, 0.25]], then the speed targets at 1, 2, 6 and 7 are 100, 100, 25 and 0 rad/s, and the
    drag feedforwards 0.1, 0.4, 0.00625 and 0 N·m.
    """
    control = _Control()
    escs = Escs(
        rotor_vel_dofs=wp.array([[1, 2], [6, 7]], dtype=wp.int32),
        channels=[1, 0],
        ct=[1e-4 / RPM_PER_RADS**2, 2e-4 / RPM_PER_RADS**2],
        cd=[0.1, 0.2],
        rpm_max=[100.0 * RPM_PER_RADS, 200.0 * RPM_PER_RADS],
    )

    escs.write(wp.array([[0.5, 2.0], [-1.0, 0.25]], dtype=float), control)

    written = (control.joint_target_qd.numpy()[[1, 2, 6, 7]], control.joint_act.numpy()[[1, 2, 6, 7]])
    np.testing.assert_allclose(written, ([100.0, 100.0, 25.0, 0.0], [0.1, 0.4, 0.00625, 0.0]), rtol=1e-5)
