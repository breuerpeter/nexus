"""The torque motors: each joint's current command to the torque on its joint, as a motor under a current drive.

A direct-drive brushless motor whose drive holds a commanded current ``I`` makes the torque ``Kt·I``, where ``Kt`` in
N·m/A is the motor's torque constant. The drive cancels the back electromotive force, so the torque doesn't fall
with speed as the rotor's voltage motor's does, and a negative command brakes. The drive is part of the motor: it
reads the controller's normalized command ``u`` on its channel and holds ``I = clamp(u, −1, 1)·currentLimit``, as
a real drive scales a signed command over its current limit. A NaN command is a controller's no output, and the
drive holds no current. One kernel does it all: it writes the current and adds the torque into
``control.joint_f``, which the physics ``clear`` stage zeroes at each substep's start. It builds no Newton
actuator.

A joint's prim declares its motor with the ``NexusTorqueMotorAPI`` schema, and the registry builds it as
:class:`TorqueMotor`, an actuator with one device stage, ``torque_motor``: it reads the ``controls`` a controller
writes, the vehicle's one controller's unless the schema's connection names another, and emits the ``current`` the
motor draws, for a battery to sum. :class:`TorqueMotors` runs the same kernel over the joints of many vehicles at
once.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.signals import Signal

from .motor import check_constants


@wp.kernel
def _drive(
    controls: wp.array2d(dtype=float),  # (vehicles, n) normalized [-1, 1] commands, NaN for none, a row per vehicle
    channels: wp.array(dtype=wp.int32),  # (nm,) the entry of its vehicle's row each motor reads
    kt: wp.array(dtype=float),  # (nm,) each motor's torque constant [N·m/A]
    current_limit: wp.array(dtype=float),  # (nm,) each motor's current at a command of 1 [A]
    joint_dofs: wp.array2d(dtype=wp.int32),  # (vehicles, nm) each joint's velocity degree of freedom
    joint_f: wp.array(dtype=float),  # the control's joint efforts, which the torque joins
    current: wp.array2d(dtype=float),  # (vehicles, nm) out: each motor's current [A]
):
    """Normalized command → the current the drive holds, that fraction of the limit → the torque, ``Kt`` times it.
    A NaN command, no output, holds no current.
    """
    v, i = wp.tid()
    c = controls[v, channels[i]]
    amps = 0.0
    if not wp.isnan(c):
        amps = wp.clamp(c, -1.0, 1.0) * current_limit[i]
    current[v, i] = amps
    wp.atomic_add(joint_f, joint_dofs[v, i], kt[i] * amps)


class TorqueMotors:
    """The motors of joints the caller has found, over every vehicle of the model at once: one row of commands per
    vehicle, each motor with its own constants.

    Args:
        joint_dofs: ``(vehicles, nm)`` each joint's velocity degree of freedom in the control.
        channels: ``(nm,)`` the entry of its vehicle's controls row each motor reads.
        torque_constant: ``(nm,)`` each motor's torque constant [N·m/A].
        current_limit: ``(nm,)`` each motor's current at a command of 1 [A].
    """

    def __init__(self, *, joint_dofs: wp.array, channels, torque_constant, current_limit):
        device = joint_dofs.device
        self._joint_dofs = joint_dofs
        self.vehicles, self.nm = joint_dofs.shape
        self._channels = wp.array(np.asarray(channels, dtype=np.int32), dtype=wp.int32, device=device)
        self._kt = wp.array(np.asarray(torque_constant, dtype=np.float32), dtype=float, device=device)
        self._current_limit = wp.array(np.asarray(current_limit, dtype=np.float32), dtype=float, device=device)

    def step(self, controls: wp.array, control, current: wp.array) -> None:
        """Add each motor's torque to ``control.joint_f`` from ``controls``, one row of normalized commands per
        vehicle, and write each motor's current into ``current``, ``(vehicles, nm)``.
        """
        wp.launch(
            _drive,
            dim=(self.vehicles, self.nm),
            inputs=(controls, self._channels, self._kt, self._current_limit, self._joint_dofs),
            outputs=(control.joint_f, current),
        )


def _joint_dof(model, path: str) -> int:
    """The velocity degree of freedom of the joint at the prim path `path` in a built model.

    Raises:
        ValueError: The model has no joint of that path, or the joint has more or fewer than one degree of
            freedom; the message names the prim.
    """
    labels = [str(label) for label in model.joint_label]
    if path not in labels:
        raise ValueError(f"{path}: the model has no joint of that path, and a torque motor sits on its joint's prim")
    j = labels.index(path)
    start = model.joint_qd_start.numpy()
    count = int(start[j + 1] - start[j])
    if count != 1:
        raise ValueError(f"{path}: the joint has {count} degrees of freedom, and a torque motor drives a joint of one")
    return int(start[j])


class TorqueMotor(Component):
    """The motor of one joint, which the joint's prim declares with ``NexusTorqueMotorAPI``: the controller's
    normalized command on its channel to the torque on the joint, and the current it draws.

    Args:
        model: The finalized articulated model, in which the motor finds its joint from its prim.
        path: The joint's prim, the one that applies the schema.
        torque_constant: The torque constant [N·m/A], ``nexus:torqueConstant``.
        current_limit: The current at a command of 1 [A], ``nexus:currentLimit``.
        channel: The entry of the controls the motor reads, ``nexus:channel``.

    Raises:
        ValueError: A constant isn't positive, ``channel`` is below 0, or the prim is no joint of the model with
            one degree of freedom; the message names the prim, and the attribute for a value.
    """

    def __init__(self, model, path: str, *, torque_constant: float = 0.0, current_limit: float = 0.0, channel: int = 0):  # fmt: skip
        check_constants(path, torque_constant=torque_constant, current_limit=current_limit)
        if int(channel) < 0:
            raise ValueError(
                f"{path}: nexus:channel is {channel}, and a torque motor reads an entry of the controls, 0 or above"
            )
        self.channel = int(channel)
        self._motors = TorqueMotors(
            joint_dofs=wp.array([[_joint_dof(model, path)]], dtype=wp.int32),
            channels=[self.channel],
            torque_constant=[torque_constant],
            current_limit=[current_limit],
        )
        # The controller's commands, of which this motor reads one: the builder checks that the controller writes
        # at least that many.
        self.controls = Signal("controls", wp.float32, shape=(1, self.channel + 1))
        self.current = Signal("current", wp.float32, shape=(1,))

    def _step(self, tick) -> None:
        """The stage's work: the controls' buffer, one row, and the current's, one value, as one vehicle's one motor."""
        self._motors.step(self.controls.buffer, tick.control, self.current.buffer.reshape((1, 1)))

    def stages(self) -> list[Stage]:
        """One device stage, ``torque_motor``, reading the controls and writing the current."""
        return [Stage("torque_motor", "device", self._step, reads=(self.controls,), writes=(self.current,))]


__all__ = ["TorqueMotor", "TorqueMotors"]
