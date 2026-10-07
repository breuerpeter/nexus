"""The rotors' command stage: the controller's command to each rotor motor's control inputs.

Each rotor's motor is a ``NewtonActuator`` prim the vehicle's Universal Scene Description (USD) file authors
on the rotor joint, ``NewtonPIDControlAPI`` with ``kd`` alone, so a velocity servo,
``effort = kd·(Ω_cmd − Ω) + feedforward``, under the ``NewtonDCMotorClampingAPI`` torque-speed envelope,
which physics steps before its solver. This stage turns the controller's normalized command into what that
servo reads: the rotor-speed target, ``clamp(u, 0, 1)·Ω_max`` with ``Ω_max`` the motor's no-load speed, and
the drag feedforward ``cd·kf·Ω²``, the aero braking torque at the target speed, which keeps the kd-only servo
from drooping against the steady load. The rotor speed Ω is then a real, solver-integrated joint state with
physical lag and saturation, and the props render spinning at no extra cost.

One device stage, ``rotors``: the targets kernel into persistent per-rotor buffers, then their scatter into
the model-sized control arrays, so a captured graph replays it over the controls the controller writes.
"""

from __future__ import annotations

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import Controls
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints


@wp.kernel
def _rotor_targets(
    cmd: wp.array2d(dtype=float),  # (1, n) normalized [0, 1] commands, the first nr the rotors
    omega_max: float,
    cd: float,
    kf: float,
    omega_cmd: wp.array(dtype=float),  # (nr,) rotor-speed targets [rad/s]
    drag_ff: wp.array(dtype=float),  # (nr,) the aero braking torque at the target speed
):
    """Normalized command → all-positive Ω target; rotor-z encodes cw/ccw, so yaw balances through the
    reaction torques. The drag feedforward, τ = cd·thrust = cd·kf·Ω², keeps the kd-only servo from
    drooping against the steady load.
    """
    i = wp.tid()
    w = wp.clamp(cmd[0, i], 0.0, 1.0) * omega_max
    omega_cmd[i] = w
    drag_ff[i] = cd * kf * w * w


@wp.kernel
def _scatter_rotor_commands(
    rotor_vel_dofs: wp.array(dtype=wp.int32),
    omega_cmd: wp.array(dtype=float),  # (nr,) target rotor speed [rad/s]
    drag_ff: wp.array(dtype=float),  # (nr,) feedforward effort [N·m], cancels the steady aero drag
    out_target_qd: wp.array(dtype=float),  # control.joint_target_qd
    out_joint_act: wp.array(dtype=float),  # control.joint_act, the servo's feedforward slot
):
    """Scatter the per-rotor velocity target + drag feedforward into the model-sized control arrays,
    device-side, so the captured graph replays it from the persistent ``omega_cmd``/``drag_ff``
    buffers the targets kernel refreshes each tick.
    """
    i = wp.tid()
    d = rotor_vel_dofs[i]
    out_target_qd[d] = omega_cmd[i]
    out_joint_act[d] = drag_ff[i]


class RotorCommand:
    """The rotors' command stage: the controller's normalized per-rotor commands to each rotor motor's
    speed target and drag feedforward in the model's ``newton.Control``.

    Args:
        model: The finalized articulated model, whose rotor joints the vehicle USD authors a motor on.
        control: The model's persistent ``newton.Control``; physics owns it. The stage writes the rotors'
            ``joint_target_qd`` and ``joint_act``, and physics' actuator step reads them.
        joints: The joint path of each rotor the vehicle USD declares, the vehicle builder's
            ``rotor_joints()``. A joint that isn't one of them is no rotor.
        ct: Thrust coefficient [N/rpm²], the propeller schema's ``nexus:ct``, which the feedforward reads.
        cd: Reaction-torque-per-thrust, ``nexus:cd``, the yaw allocation κ.
        rpm_max: Rotor speed at normalized command 1.0 [rpm], the motor's no-load speed,
            ``newton:velocityLimit``.
    """

    def __init__(self, *, model, control, joints, ct: float, cd: float, rpm_max: float):
        self.control = control
        self.kf = float(ct) * RPM_PER_RADS * RPM_PER_RADS  # thrust = kf·Ω², with Ω in rad/s
        self.cd = float(cd)
        self.omega_max = float(rpm_max) / RPM_PER_RADS  # rotor speed [rad/s] at normalized command 1.0
        vel_dofs, _pos_coords, _bodies, _base = find_rotor_joints(model, joints)
        self.nr = len(vel_dofs)
        self._rotor_dofs = wp.array(vel_dofs, dtype=wp.int32)
        # Persistent small device buffers for the per-rotor command scatter.
        self._omega_cmd = wp.zeros(self.nr, dtype=float)
        self._drag_ff = wp.zeros(self.nr, dtype=float)
        # The controller's commands, of which the first nr are the rotor motors: the builder checks that the
        # controller writes at least that many, as PX4 streams 16 HIL_ACTUATOR_CONTROLS channels.
        self.controls = Signal("controls", Controls, shape=(1, self.nr))

    def _write(self, tick) -> None:
        """The device stage: the controller's ``(1, n)`` normalized commands → rotor-speed targets and the
        steady-state drag feedforward → scatter into the control arrays.
        """
        wp.launch(
            _rotor_targets,
            dim=self.nr,
            inputs=(self.controls.buffer, self.omega_max, self.cd, self.kf),
            outputs=(self._omega_cmd, self._drag_ff),
        )
        wp.launch(
            _scatter_rotor_commands,
            dim=self.nr,
            inputs=(self._rotor_dofs, self._omega_cmd, self._drag_ff),
            outputs=(self.control.joint_target_qd, self.control.joint_act),
        )

    def stages(self) -> list[Stage]:
        """One device stage, ``rotors``, reading the controls."""
        return [Stage("rotors", "device", self._write, reads=(self.controls,))]


__all__ = ["RotorCommand"]
