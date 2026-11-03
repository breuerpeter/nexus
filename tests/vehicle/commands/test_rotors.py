"""The rotors' ESCs write each rotor's terminal voltage from its own channel and its own duty map, and the rotors'
motors turn that voltage into the speed target and torque on each rotor's joint, for the rotors of one vehicle,
as a run builds them, or of many at once, as the RL task steps them.
"""

import math

import numpy as np
import pytest
import warp as wp

from nexus_uav._src.vehicle.commands.esc import Escs
from nexus_uav._src.vehicle.commands.motor import Motors

pytestmark = pytest.mark.usefixtures("warp_cpu")

DOFS = [[1, 2], [6, 7]]  # each vehicle's rotors' velocity degrees of freedom, ten wide in all
DT = 0.004  # [s]


class _Control:
    """Newton's control inputs the ESCs and motors write, ten degrees of freedom wide."""

    def __init__(self):
        self.joint_target_q = wp.zeros(11, dtype=float)
        self.joint_target_qd = wp.zeros(10, dtype=float)
        self.joint_act = wp.zeros(10, dtype=float)
        self.joint_f = wp.zeros(10, dtype=float)


class _State:
    """Newton's state the motors read, every rotor at 100 rad/s."""

    def __init__(self):
        self.joint_q = wp.zeros(11, dtype=float)
        self.joint_qd = wp.full(10, 100.0, dtype=float)


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


def test_each_motor_makes_the_dc_motors_torque_from_its_voltage_and_emits_its_current():
    """Each motor makes the DC motor's torque from its rotor's voltage, capped at its current limit, and emits
    the current it draws.

    Given the rotors of two vehicles at 100 rad/s, each motor with kv 100 rpm/V, so Ke is 0.0955 V·s/rad, and
    1 Ω, the second rotor's current capped at 2 A, when the motors step at 24 and 6 V, then the torques are
    Kt·(V − Ke·Ω)/R, 1.38 and -0.339 N·m, the capped ones ±0.191 N·m, and the currents those over Kt.
    """
    ke = 60.0 / (2.0 * math.pi * 100.0)
    control, state, current = _Control(), _State(), wp.zeros((2, 2), dtype=float)
    motors = Motors(
        rotor_vel_dofs=wp.array(DOFS, dtype=wp.int32),
        kv=[100.0, 100.0],
        resistance=[1.0, 1.0],
        current_limit=[1e9, 2.0],
    )

    motors.step(wp.array([[24.0, 24.0], [6.0, 6.0]], dtype=float), state, control, DT, current)

    torque = [ke * (24.0 - ke * 100.0), ke * 2.0, ke * (6.0 - ke * 100.0), -ke * 2.0]
    flat = [d for dofs in DOFS for d in dofs]
    np.testing.assert_allclose(control.joint_f.numpy()[flat], torque, rtol=1e-5)
    np.testing.assert_allclose(current.numpy().reshape(-1), np.asarray(torque) / ke, rtol=1e-5)
