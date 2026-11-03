"""The rotors' ESCs write each rotor's terminal voltage from its own channel and its own duty map, for the rotors
of one vehicle, as a run builds them, or of many at once, as the RL task steps them.
"""

import numpy as np
import pytest
import warp as wp

from nexus_uav._src.vehicle.commands.esc import Escs

pytestmark = pytest.mark.usefixtures("warp_cpu")


def test_each_rotor_of_each_vehicle_takes_its_own_command_from_its_channel_with_its_own_duty_map():
    """Each rotor of each vehicle takes its own command from its channel, with its own duty map.

    Given the rotors of two vehicles, the first rotor on channel 1 with a duty map from 0.1 to 1, the second on
    channel 0 with one from 0.2 to 0.8, the first vehicle's bus at 24 V and the second's at 12 V, when the ESCs
    write the commands [[0.5, 2.0], [-1.0, 0.25]], then the voltages are 24, 12, 3.9 and 2.4 V.
    """
    voltage = wp.zeros((2, 2), dtype=float)
    escs = Escs(channels=[1, 0], idle_duty=[0.1, 0.2], max_duty=[1.0, 0.8])

    escs.write(
        wp.array([[0.5, 2.0], [-1.0, 0.25]], dtype=float), wp.array([[24.0, 24.0], [12.0, 12.0]], dtype=float), voltage
    )

    np.testing.assert_allclose(voltage.numpy(), [[24.0, 12.0], [3.9, 2.4]], rtol=1e-5)
