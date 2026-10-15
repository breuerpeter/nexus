"""The rotors' command element: the controller's command to each rotor motor's control inputs.

Each rotor's motor is a ``NewtonActuator`` prim the vehicle's Universal Scene Description (USD) file authors
on the rotor joint, ``NewtonPIDControlAPI`` with ``kd`` alone, so a velocity servo,
``effort = kd·(Ω_cmd − Ω) + feedforward``, under the ``NewtonDCMotorClampingAPI`` torque-speed envelope,
which physics steps before its solver. This stage turns the controller's normalized command into what that
servo reads: the rotor-speed target, ``clamp(u, 0, 1)·Ω_max`` with ``Ω_max`` the motor's no-load speed, and
the drag feedforward ``cd·kf·Ω²``, the aero braking torque at the target speed, which keeps the kd-only servo
from drooping against the steady load. The rotor speed Ω is then a real, solver-integrated joint state with
physical lag and saturation, and the props render spinning at no extra cost.

One device stage, ``rotors``: one kernel writes each rotor's target and feedforward straight into Newton's
control arrays, with no buffer of its own, so a captured graph replays it over the controls the controller
writes, and a recorded gradient rollout keeps every step's write in that step's control.
"""

from __future__ import annotations

import warp as wp

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.signals import Signal
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints


@wp.kernel
def _rotor_targets(
    cmd: wp.array2d(dtype=float),  # (vehicles, n) normalized [0, 1] commands, the first nr of each row the rotors
    rotor_vel_dofs: wp.array2d(dtype=wp.int32),  # (vehicles, nr) each rotor's velocity degree of freedom
    omega_max: float,
    cd: float,
    kf: float,
    target_qd: wp.array(dtype=float),  # control.joint_target_qd
    joint_act: wp.array(dtype=float),  # control.joint_act, the servo's feedforward slot
):
    """Normalized command → all-positive Ω target; rotor-z encodes cw/ccw, so yaw balances through the
    reaction torques. The drag feedforward, τ = cd·thrust = cd·kf·Ω², keeps the kd-only servo from
    drooping against the steady load.
    """
    v, i = wp.tid()
    w = wp.clamp(cmd[v, i], 0.0, 1.0) * omega_max
    d = rotor_vel_dofs[v, i]
    target_qd[d] = w
    joint_act[d] = cd * kf * w * w


class RotorCommand(Component):
    """The rotors' command element: the controller's normalized per-rotor commands to each rotor motor's
    speed target and drag feedforward in the model's ``newton.Control``.

    Args:
        model: The finalized articulated model, whose rotor joints the vehicle USD authors a motor on.
        control: The model's persistent ``newton.Control``; physics owns it. The stage writes the rotors'
            ``joint_target_qd`` and ``joint_act``, and physics' actuator step reads them.
        joints: The joint path of each rotor the vehicle USD declares, the ``VehicleUsd``'s
            ``rotor_joints()``. A joint that isn't one of them is no rotor.
        ct: Thrust coefficient [N/rpm²], the propeller schema's ``nexus:ct``, which the feedforward reads.
        cd: Reaction-torque-per-thrust, ``nexus:cd``, the yaw allocation κ.
        rpm_max: Rotor speed at normalized command 1.0 [rpm], the motor's no-load speed,
            ``newton:velocityLimit``.
    """

    def __init__(self, *, model, control, joints, ct: float, cd: float, rpm_max: float):
        vel_dofs, _pos_coords, _bodies, _base = find_rotor_joints(model, joints)
        self._setup(wp.array([vel_dofs], dtype=wp.int32), control, ct, cd, rpm_max)

    @classmethod
    def for_rotors(cls, *, rotor_vel_dofs: wp.array, control, ct: float, cd: float, rpm_max: float) -> RotorCommand:
        """The command element of rotors the caller has found: one row of velocity degrees of freedom per
        vehicle, as the RL task steps many vehicles in one model.

        Args:
            rotor_vel_dofs: ``(vehicles, nr)`` each rotor's velocity degree of freedom in ``control``.
            control: The ``newton.Control`` whose ``joint_target_qd`` and ``joint_act`` the command writes.
            ct: Thrust coefficient [N/rpm²].
            cd: Reaction-torque-per-thrust.
            rpm_max: Rotor speed at normalized command 1.0 [rpm].

        Returns:
            The command element; :meth:`write` takes one row of commands per vehicle.
        """
        command = cls.__new__(cls)
        command._setup(rotor_vel_dofs, control, ct, cd, rpm_max)
        return command

    def _setup(self, rotor_vel_dofs: wp.array, control, ct: float, cd: float, rpm_max: float) -> None:
        self.control = control
        self.kf = float(ct) * RPM_PER_RADS * RPM_PER_RADS  # thrust = kf·Ω², with Ω in rad/s
        self.cd = float(cd)
        self.omega_max = float(rpm_max) / RPM_PER_RADS  # rotor speed [rad/s] at normalized command 1.0
        self._rotor_dofs = rotor_vel_dofs
        self.vehicles, self.nr = rotor_vel_dofs.shape
        # The controller's commands, of which the first nr are the rotor motors: the builder checks that the
        # controller writes at least that many, as PX4 streams 16 HIL_ACTUATOR_CONTROLS channels.
        self.controls = Signal("controls", wp.float32, shape=(1, self.nr))

    def write(self, cmd: wp.array, control=None) -> None:
        """Write each rotor's speed target and drag feedforward from ``cmd``, one row of normalized commands
        per vehicle, into ``control``, or into the command element's own when the caller passes none.
        """
        control = self.control if control is None else control
        wp.launch(
            _rotor_targets,
            dim=(self.vehicles, self.nr),
            inputs=(cmd, self._rotor_dofs, self.omega_max, self.cd, self.kf),
            outputs=(control.joint_target_qd, control.joint_act),
        )

    def stages(self) -> list[Stage]:
        """One device stage, ``rotors``, reading the controls."""
        return [Stage("rotors", "device", lambda tick: self.write(self.controls.buffer), reads=(self.controls,))]


__all__ = ["RotorCommand"]
