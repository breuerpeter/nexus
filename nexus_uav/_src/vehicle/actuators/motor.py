"""The rotors' motors: each rotor's terminal voltage to the torque on its joint, as a first-order DC motor.

A brushless motor at voltage ``V`` and speed ``Ω`` makes the torque ``(Kt/R)·(V − Ke·Ω)``, where ``Ke`` in
V·s/rad and ``Kt`` in N·m/A are the one constant ``60 / (2π·kv)`` of the sheet's KV, and ``R`` is the winding
resistance line to line. Newton's actuator holds that law: a ``ControllerPD`` with ``kd = Ke²/R`` and a speed
target ``V/Ke`` makes ``kd·(V/Ke − Ω)``, the same torque, and a ``ClampingMaxEffort`` at ``Kt·currentLimit`` caps
it at the drive train's continuous current. The motor builds that actuator in code from its own attributes, so
the vehicle's file declares no ``NewtonActuator`` prim on a rotor's joint, and the physics ``clear`` stage zeroes
``joint_f`` so every actuator of the model accumulates into it. The ESC's duty bounds the voltage, so the
motor needs no saturation line. The rotor speed is then a solver-integrated joint state that droops under the
propeller's load as a real drive's does, with no feedforward holding it up.

Each rotor's rigid body declares its motor with the ``NexusMotorAPI`` schema, beside its ESC's and its propeller's,
and the registry builds it as :class:`Motor`, an actuator with one device stage, ``motor``, which runs after the
command elements each substep whatever order the body lists the schemas in: it reads the
``voltage`` its ESC writes, the one on its own prim unless the schema's connection names another, writes the
actuator's target and its torque, and emits the ``current`` the motor draws, the applied torque over ``Kt``, for a
battery to sum. :class:`Motors` runs the same actuator over the rotors of many vehicles at once, as the RL task
and the gradient rollout step them.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from newton.actuators import Actuator, ClampingMaxEffort, ControllerPD

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.signals import Signal
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, rotor_of

# The attributes a motor's class takes, each of which must be positive: the sheet's constants.
_ATTRIBUTES = {"kv": "nexus:kv", "resistance": "nexus:resistance", "current_limit": "nexus:currentLimit"}


def motor_constant(kv: float) -> float:
    """``Ke`` [V·s/rad], the back electromotive force constant of a motor of ``kv`` [rpm/V], which is ``Kt`` [N·m/A]."""
    return RPM_PER_RADS / float(kv)


def check_constants(path: str, **values: float) -> None:
    """Fail a motor constant that isn't positive.

    Args:
        path: The prim the error names.
        **values: ``kv``, ``resistance`` and ``current_limit``.

    Raises:
        ValueError: A value isn't positive; the message names the prim and the attribute.
    """
    for keyword, attribute in _ATTRIBUTES.items():
        if not values[keyword] > 0.0:
            raise ValueError(f"{path}: {attribute} is {values[keyword]:g}, and a motor's constant must be positive")


@wp.kernel
def _targets(
    voltage: wp.array2d(dtype=float),  # (vehicles, nr) each rotor's terminal voltage [V]
    ke: wp.array(dtype=float),  # (nr,) each motor's Ke [V·s/rad]
    rotor_vel_dofs: wp.array2d(dtype=wp.int32),  # (vehicles, nr) each rotor's velocity degree of freedom
    target_qd: wp.array(dtype=float),  # the control's velocity targets
    joint_act: wp.array(dtype=float),  # the control's feedforward slot
):
    """The actuator's speed target, ``V/Ke``, the speed the voltage would spin an unloaded motor at, and no feedforward."""
    v, i = wp.tid()
    d = rotor_vel_dofs[v, i]
    target_qd[d] = voltage[v, i] / ke[i]
    joint_act[d] = 0.0


@wp.kernel
def _current(
    joint_f: wp.array(
        dtype=float
    ),  # the control's joint efforts, the actuator's torque at the rotor's degree of freedom
    ke: wp.array(dtype=float),  # (nr,) each motor's Kt [N·m/A], the same number as Ke
    rotor_vel_dofs: wp.array2d(dtype=wp.int32),  # (vehicles, nr)
    current: wp.array2d(dtype=float),  # (vehicles, nr) out: each motor's current [A]
):
    """The current a motor draws: its applied torque over ``Kt``."""
    v, i = wp.tid()
    current[v, i] = joint_f[rotor_vel_dofs[v, i]] / ke[i]


class Motors:
    """The motors of rotors the caller has found: one Newton actuator over every rotor of every vehicle, as the RL
    task steps many vehicles in one model, each rotor with its own constants.

    Args:
        rotor_vel_dofs: ``(vehicles, nr)`` each rotor's velocity degree of freedom in the control.
        kv: ``(nr,)`` each motor's velocity constant [rpm/V].
        resistance: ``(nr,)`` each motor's winding resistance, line to line [Ω].
        current_limit: ``(nr,)`` each drive train's continuous current [A].
        requires_grad: Give the actuator's buffers gradient support, for a recorded rollout.
    """

    def __init__(self, *, rotor_vel_dofs: wp.array, kv, resistance, current_limit, requires_grad: bool = False):
        device = rotor_vel_dofs.device
        self._rotor_dofs = rotor_vel_dofs
        self.vehicles, self.nr = rotor_vel_dofs.shape
        ke = np.array([motor_constant(k) for k in kv], dtype=np.float64)  # each in double, then single
        kd = ke * ke / np.asarray(resistance, dtype=np.float64)
        effort = ke * np.asarray(current_limit, dtype=np.float64)
        self._ke = wp.array(ke.astype(np.float32), dtype=float, device=device)
        n = self.vehicles * self.nr
        tile = lambda a: wp.array(np.tile(a.astype(np.float32), self.vehicles), dtype=float, device=device)  # noqa: E731
        self.actuator = Actuator(
            wp.array(rotor_vel_dofs.numpy().reshape(-1).astype(np.uint32), dtype=wp.uint32, device=device),
            ControllerPD(kp=wp.zeros(n, dtype=float, device=device), kd=tile(kd)),
            clamping=[ClampingMaxEffort(max_effort=tile(effort))],
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
            requires_grad=requires_grad,
        )

    def step(self, voltage: wp.array, state, control, dt: float, current: wp.array) -> None:
        """Write each motor's target from ``voltage``, ``(vehicles, nr)``, step the actuator so its torque joins
        ``control.joint_f``, and write each motor's current into ``current``, ``(vehicles, nr)``.
        """
        wp.launch(
            _targets,
            dim=(self.vehicles, self.nr),
            inputs=(voltage, self._ke, self._rotor_dofs),
            outputs=(control.joint_target_qd, control.joint_act),
        )
        self.actuator.step(state, control, None, None, dt)
        wp.launch(
            _current,
            dim=(self.vehicles, self.nr),
            inputs=(control.joint_f, self._ke, self._rotor_dofs),
            outputs=(current,),
        )


class Motor(Component):
    """The motor of one rotor, which its rigid body declares with ``NexusMotorAPI``: its ESC's terminal voltage to
    the torque on the rotor's joint, and the current it draws.

    Args:
        model: The finalized articulated model, in which the motor finds its rotor's joint from its prim.
        path: The rotor's rigid body, the prim that applies the schema.
        kv: The velocity constant [rpm/V], ``nexus:kv``.
        resistance: The winding resistance, line to line [Ω], ``nexus:resistance``.
        current_limit: The continuous current [A], ``nexus:currentLimit``.

    Raises:
        ValueError: A constant isn't positive, or no revolute joint has the body as its child; the message names
            the prim.
    """

    def __init__(self, model, path: str, *, kv: float = 0.0, resistance: float = 0.0, current_limit: float = 0.0):
        check_constants(path, kv=kv, resistance=resistance, current_limit=current_limit)
        dof, _coord, _body, _base = rotor_of(model, path)
        self._motors = Motors(
            rotor_vel_dofs=wp.array([[dof]], dtype=wp.int32),
            kv=[kv],
            resistance=[resistance],
            current_limit=[current_limit],
        )
        self.voltage = Signal("voltage", wp.float32, shape=(1,))
        self.current = Signal("current", wp.float32, shape=(1,))

    def _step(self, tick) -> None:
        """The stage's work: the voltage's buffer and the current's, each one value, as one vehicle's one rotor."""
        voltage, current = self.voltage.buffer.reshape((1, 1)), self.current.buffer.reshape((1, 1))
        self._motors.step(voltage, tick.state, tick.control, tick.dt, current)

    def stages(self) -> list[Stage]:
        """One device stage, ``motor``, reading the voltage and writing the current."""
        return [Stage("motor", "device", self._step, reads=(self.voltage,), writes=(self.current,))]


__all__ = ["Motor", "Motors", "check_constants", "motor_constant"]
