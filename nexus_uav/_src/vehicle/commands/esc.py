"""The rotors' Electronic Speed Controllers (ESCs): the controller's command to each rotor motor's control inputs.

Each rotor's motor is a ``NewtonActuator`` prim the vehicle's Universal Scene Description (USD) file authors
on the rotor joint, ``NewtonPIDControlAPI`` with ``kd`` alone, so a velocity servo,
``effort = kd·(Ω_cmd − Ω) + feedforward``, under the ``NewtonDCMotorClampingAPI`` torque-speed envelope,
which physics steps before its solver. The ESC turns the controller's normalized command into what that
servo reads: the rotor-speed target, ``clamp(u, 0, 1)·Ω_max`` with ``Ω_max`` the motor's no-load speed, and
the drag feedforward ``cd·kf·Ω²``, the aero braking torque at the target speed, which keeps the kd-only servo
from drooping against the steady load. The rotor speed Ω is then a real, solver-integrated joint state with
physical lag and saturation, and the props render spinning at no extra cost.

Each rotor's rigid body declares its ESC with the ``NexusEscAPI`` schema, beside its propeller's, and the registry
builds it as :class:`Esc`, a command element with one device stage, ``esc``: one kernel writes the rotor's
target and feedforward straight into Newton's control arrays, with no buffer of its own, so a captured graph
replays it over the controls the controller writes, and a recorded gradient rollout keeps every step's write in
that step's control. :class:`Escs` runs the same kernel over the rotors of many vehicles at once, as the RL task
and the gradient rollout step them.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import warp as wp

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.signals import Signal
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, motor_speed, rotor_of


@wp.kernel
def _rotor_targets(
    cmd: wp.array2d(dtype=float),  # (vehicles, n) normalized [0, 1] commands, one row per vehicle
    rotor_vel_dofs: wp.array2d(dtype=wp.int32),  # (vehicles, nr) each rotor's velocity degree of freedom
    channels: wp.array(dtype=wp.int32),  # (nr,) the entry of its vehicle's row each rotor reads
    omega_max: wp.array(dtype=float),  # (nr,) each rotor's speed at full command [rad/s]
    cd: wp.array(dtype=float),  # (nr,) each propeller's reaction-torque-per-thrust
    kf: wp.array(dtype=float),  # (nr,) each propeller's thrust = kf·Ω²
    target_qd: wp.array(dtype=float),  # control.joint_target_qd
    joint_act: wp.array(dtype=float),  # control.joint_act, the servo's feedforward slot
):
    """Normalized command → all-positive Ω target; rotor-z encodes cw/ccw, so yaw balances through the
    reaction torques. The drag feedforward, τ = cd·thrust = cd·kf·Ω², keeps the kd-only servo from
    drooping against the steady load.
    """
    v, i = wp.tid()
    w = wp.clamp(cmd[v, channels[i]], 0.0, 1.0) * omega_max[i]
    d = rotor_vel_dofs[v, i]
    target_qd[d] = w
    joint_act[d] = cd[i] * kf[i] * w * w


class Escs:
    """The ESCs of rotors the caller has found: one row of velocity degrees of freedom per vehicle, as the RL
    task steps many vehicles in one model, each rotor with its own values.

    Args:
        rotor_vel_dofs: ``(vehicles, nr)`` each rotor's velocity degree of freedom in the control.
        channels: ``(nr,)`` the entry of its vehicle's command row each rotor reads.
        ct: ``(nr,)`` each propeller's thrust coefficient [N/rpm²], which the feedforward reads.
        cd: ``(nr,)`` each propeller's reaction-torque-per-thrust, the yaw allocation κ.
        rpm_max: ``(nr,)`` each rotor's speed at normalized command 1.0 [rpm], its motor's no-load speed.
    """

    def __init__(self, *, rotor_vel_dofs: wp.array, channels, ct, cd, rpm_max):
        device = rotor_vel_dofs.device
        self._rotor_dofs = rotor_vel_dofs
        self.vehicles, self.nr = rotor_vel_dofs.shape
        self._channels = wp.array(np.asarray(channels, dtype=np.int32), dtype=wp.int32, device=device)
        # thrust = kf·Ω², with Ω in rad/s, and the speed at full command in rad/s: each in double, then single
        kf = np.asarray(ct, dtype=np.float64) * RPM_PER_RADS * RPM_PER_RADS
        omega_max = np.asarray(rpm_max, dtype=np.float64) / RPM_PER_RADS
        self._kf = wp.array(kf.astype(np.float32), dtype=float, device=device)
        self._cd = wp.array(np.asarray(cd, dtype=np.float32), dtype=float, device=device)
        self._omega_max = wp.array(omega_max.astype(np.float32), dtype=float, device=device)

    def write(self, cmd: wp.array, control) -> None:
        """Write each rotor's speed target and drag feedforward from ``cmd``, one row of normalized commands
        per vehicle, into ``control``.
        """
        wp.launch(
            _rotor_targets,
            dim=(self.vehicles, self.nr),
            inputs=(cmd, self._rotor_dofs, self._channels, self._omega_max, self._cd, self._kf),
            outputs=(control.joint_target_qd, control.joint_act),
        )


class Esc(Component):
    """The ESC of one rotor, which its rigid body declares with ``NexusEscAPI``: the controller's normalized
    command on its channel to its rotor motor's speed target and drag feedforward in the model's ``newton.Control``.

    Args:
        model: The finalized articulated model, whose rotor joint the vehicle USD authors a motor on.
        path: The rotor's rigid body, the prim that applies the schema.
        channel: The entry of the controls the ESC reads, ``nexus:channel``.
        propeller: The propeller's values on the same body, ``NexusPropellerAPI``'s ``ct`` and ``cd`` among them,
            which the feedforward reads; the builder hands them over, since the class names the schema in
            ``siblings``.
    """

    siblings: ClassVar[dict[str, str]] = {"propeller": "NexusPropellerAPI"}

    def __init__(self, model, path: str, *, channel: int = 0, propeller: dict):
        dof, _coord, _body, _base = rotor_of(model, path)
        speed = motor_speed(model, dof, path)  # rad/s
        self.channel = int(channel)
        self._escs = Escs(
            rotor_vel_dofs=wp.array([[dof]], dtype=wp.int32),
            channels=[0],
            ct=[float(propeller["ct"])],
            cd=[float(propeller["cd"])],
            rpm_max=[speed * RPM_PER_RADS],
        )
        # The controller's commands, of which this ESC reads one: the builder checks that the controller writes
        # at least that many, as PX4 streams 16 HIL_ACTUATOR_CONTROLS channels.
        self.controls = Signal("controls", wp.float32, shape=(1, self.channel + 1))

    def stages(self) -> list[Stage]:
        """One device stage, ``esc``, reading the controls."""
        return [
            Stage(
                "esc",
                "device",
                lambda tick: self._escs.write(self.controls.buffer[:, self.channel : self.channel + 1], tick.control),
                reads=(self.controls,),
            )
        ]


__all__ = ["Esc", "Escs"]
