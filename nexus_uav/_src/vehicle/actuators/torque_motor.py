"""The torque motors: each joint's current command to the torque on its joint."""

from __future__ import annotations

import warp as wp

from nexus_uav._src.core.interfaces import Component


class TorqueMotors:
    """The motors of joints the caller has found, over every vehicle of the model at once.

    Args:
        joint_dofs: ``(vehicles, n)`` each joint's velocity degree of freedom in the control.
        channels: ``(n,)`` the entry of its vehicle's controls row each motor reads.
        torque_constant: ``(n,)`` each motor's torque constant [N·m/A].
        current_limit: ``(n,)`` each motor's current at a command of 1 [A].
    """

    def __init__(self, *, joint_dofs: wp.array, channels, torque_constant, current_limit): ...

    def step(self, controls: wp.array, control, current: wp.array) -> None:
        """Add each motor's torque to ``control.joint_f`` from ``controls``, one row per vehicle, and write each
        motor's current into ``current``, ``(vehicles, n)``.
        """


class TorqueMotor(Component):
    """The motor of one joint, which the joint's prim declares.

    Args:
        model: The finalized articulated model, in which the motor finds its joint from its prim.
        path: The joint's prim, the one that applies the schema.
        torque_constant: The torque constant [N·m/A], ``nexus:torqueConstant``.
        current_limit: The current at a command of 1 [A], ``nexus:currentLimit``.
        channel: The entry of the controls the motor reads, ``nexus:channel``.
    """

    def __init__(self, model, path: str, *, torque_constant: float = 0.0, current_limit: float = 0.0, channel: int = 0): ...  # fmt: skip


__all__ = ["TorqueMotor", "TorqueMotors"]
