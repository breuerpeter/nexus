"""The articulated plant as a differentiable rollout: the vehicle every run flies, recorded on a tape.

:class:`ArticulatedRollout` builds the vehicle from its Universal Scene Description (USD) file as physics does:
its rotor joints turn under the Newton motors the file authors, the core's ESCs write each motor's
speed target, and the core's propellers turn each rotor's speed into its force. So a gradient example tunes the
plant PX4 flies, not a stand-in. Three things make the rollout differentiable:

* ``SolverSemiImplicit``, the solver whose step has an adjoint, at :data:`SUBSTEPS` steps per control tick, with
  its joint springs at :data:`JOINT_KE` and :data:`JOINT_KD`: at its defaults the stiff springs go NaN on the light
  rotors at any step that a run can afford.
* One state, one control and one Newton motor per solver step. A tape replays each step backward from the arrays
  that step wrote, so no step can overwrite another's. A Newton motor writes its torque into scratch buffers it owns,
  and its Proportional Integral Derivative (PID) controller keeps one more, so :func:`per_step_motors` builds a
  fresh motor per step through Newton's public constructors, the way a rollout keeps a state per step.
* :class:`CapturedGradient`, which replays the whole forward and backward pass from one CUDA graph: tens of
  thousands of small launches cost seconds as a graph and minutes from Python.
"""

from __future__ import annotations

import newton
import newton.solvers
import numpy as np
import warp as wp
from newton.actuators import Actuator, ClampingDCMotor, ClampingMaxEffort, ControllerPID

from nexus_uav._src.physics.vehicle import VehicleUsd
from nexus_uav._src.scene.world import GRAVITY
from nexus_uav._src.vehicle.commands import Escs
from nexus_uav._src.vehicle.forces import Propellers
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, rotor_of, rotor_speeds

SUBSTEPS = 16  # solver steps per 4 ms control tick: 0.25 ms, the step SemiImplicit holds the rotor joints at
JOINT_KE = 1000.0  # SemiImplicit's joint spring stiffness; its default goes NaN on a rotor of 1.5e-4 kg·m²
JOINT_KD = 1.0  # its joint spring damping

# The motor parts a fresh copy can share: they read their settings and hold no buffer a step writes.
_SHARED_CLAMPS = (ClampingDCMotor, ClampingMaxEffort)


def per_step_motors(motor: Actuator, steps: int) -> list[Actuator]:
    """One copy of a Newton motor per solver step, each with its own scratch buffers.

    A Newton motor computes its torque into buffers it owns, and its PID controller keeps one more for the
    integral, so a rollout that stepped one motor every step would overwrite, step after step, what the tape
    needs to replay each step backward. Each copy shares the motor's settings and clamps, and gets a PID
    controller of its own.

    Args:
        motor: A motor of the model, as Newton built it from the vehicle's ``NewtonActuator`` prims.
        steps: How many solver steps the rollout records.

    Returns:
        The motor for each step, in step order.

    Raises:
        ValueError: The motor has a part a copy can't share or rebuild: a controller other than
            ``ControllerPID``, a delay, or a clamp other than the DC motor or effort limit. The message names it.
    """
    controller = motor.controller
    unknown = [
        type(part).__name__
        for part in (controller, motor.delay, *motor.clamping)
        if part is not None and not isinstance(part, (ControllerPID, *_SHARED_CLAMPS))
    ]
    if unknown:
        raise ValueError(
            f"a gradient rollout copies a motor's ControllerPID and shares its DC motor or effort clamp; "
            f"this motor has {', '.join(unknown)}"
        )
    return [
        Actuator(
            motor.indices,
            ControllerPID(
                kp=controller.kp,
                ki=controller.ki,
                kd=controller.kd,
                integral_max=controller.integral_max,
                const_effort=controller.const_effort,
            ),
            clamping=motor.clamping,
            pos_indices=motor.pos_indices,
            target_pos_indices=motor.target_pos_indices,
            effort_indices=motor.effort_indices,
            state_pos_attr=motor.state_pos_attr,
            state_vel_attr=motor.state_vel_attr,
            control_target_pos_attr=motor.control_target_pos_attr,
            control_target_vel_attr=motor.control_target_vel_attr,
            control_feedforward_attr=motor.control_feedforward_attr,
            control_output_attr=motor.control_output_attr,
            requires_grad=motor.requires_grad,
        )
        for _ in range(steps)
    ]


class ArticulatedRollout:
    """The vehicle's articulated plant over ``ticks`` control ticks, ready to record on a tape.

    Args:
        vehicle_usd: The vehicle, as the run resolved it.
        ticks: The control ticks the rollout records.
        dt: The control tick [s]; the solver steps at ``dt / SUBSTEPS``.
        start: Where the airframe starts [m], upright with its rotors spinning at hover speed.
        requires_grad: Record the gradient of the rollout; off for a forward-only flight.
    """

    def __init__(self, vehicle_usd, *, ticks: int, dt: float = 0.004, start=(0.0, 0.0, 2.0), requires_grad=True):
        builder = newton.ModelBuilder()
        VehicleUsd({**vehicle_usd.cfg, "spawn_pos": tuple(float(v) for v in start)}).build(builder)
        builder.gravity = -GRAVITY
        self.model = model = builder.finalize(requires_grad=requires_grad)
        self.ticks = int(ticks)
        self.steps = self.ticks * SUBSTEPS
        self.dt = float(dt) / SUBSTEPS
        self.solver = newton.solvers.SolverSemiImplicit(model, joint_attach_ke=JOINT_KE, joint_attach_kd=JOINT_KD)
        self.states = [model.state(requires_grad=requires_grad) for _ in range(self.steps + 1)]
        self.controls = [model.control(requires_grad=requires_grad) for _ in range(self.steps)]
        self.motors = [per_step_motors(motor, self.steps) for motor in model.actuators or []]
        self.motor_states = [[motor.state() for _ in range(self.steps + 1)] for motor in model.actuators or []]

        rotors = vehicle_usd.rotors()  # in channel order, so a command's entries are the rotors'
        found = [rotor_of(model, rotor.body) for rotor in rotors]  # each rotor's dof, coord, body and airframe
        self.rotor_dofs = [dof for dof, _coord, _body, _base in found]
        self.rotor_bodies = [body for _dof, _coord, body, _base in found]
        self.base = found[0][3]
        values = {key: [rotor.propeller[key] for rotor in rotors] for key in ("ct", "cd", "aero_h", "aero_hforce")}
        self.rpm_max = rotor_speeds(model, rotors)  # each rotor's speed at full command [rpm], its motor's
        self.command = Escs(
            rotor_vel_dofs=wp.array([self.rotor_dofs], dtype=wp.int32),
            channels=[rotor.channel for rotor in rotors],
            ct=values["ct"],
            cd=values["cd"],
            rpm_max=self.rpm_max,
        )
        self.propellers = Propellers(
            rotor_vel_dofs=wp.array(self.rotor_dofs, dtype=wp.int32),
            rotor_bodies=wp.array(self.rotor_bodies, dtype=wp.int32),
            rotor_bases=wp.array([self.base] * len(found), dtype=wp.int32),
            **values,
        )
        self.mass = float(model.body_mass.numpy().sum())
        kf = np.asarray(values["ct"], dtype=np.float64) * RPM_PER_RADS * RPM_PER_RADS
        self.hover_speed = np.sqrt(self.mass * GRAVITY / len(self.rotor_dofs) / kf)  # (nr,) rad/s, each rotor's

    def reset(self) -> None:
        """Put the vehicle back at its start, at rest with each rotor at its hover speed, and its motors' state at zero."""
        joint_q = self.model.joint_q.numpy()
        joint_qd = np.zeros_like(self.model.joint_qd.numpy())
        joint_qd[self.rotor_dofs] = self.hover_speed
        start = self.states[0]
        start.joint_q.assign(joint_q)
        start.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, start.joint_q, start.joint_qd, start)
        for states in self.motor_states:
            for state in states:
                if state is not None:
                    state.reset()

    def step(self, i: int, cmd: wp.array) -> None:
        """Solver step ``i`` under ``cmd``, one row of normalized rotor commands, in the order a run steps physics:
        the ESCs, the propellers, Newton's motors, then the solver. The joints' coordinates of the next state
        follow its bodies, since the solver integrates the bodies alone.
        """
        state, nxt, control = self.states[i], self.states[i + 1], self.controls[i]
        state.clear_forces()
        self.command.write(cmd, control)
        self.propellers.add(state)
        control.joint_f.zero_()
        for motors, states in zip(self.motors, self.motor_states, strict=True):
            motors[i].step(state, control, states[i], states[i + 1], self.dt)
        self.solver.step(state, nxt, control, None, self.dt)
        newton.eval_ik(self.model, nxt, nxt.joint_q, nxt.joint_qd)

    def tick(self, t: int, cmd: wp.array) -> None:
        """Control tick ``t``: its :data:`SUBSTEPS` solver steps under ``cmd``."""
        for j in range(SUBSTEPS):
            self.step(t * SUBSTEPS + j, cmd)

    def state_at(self, t: int):
        """The state at the start of control tick ``t``, the end of tick ``t - 1``."""
        return self.states[t * SUBSTEPS]

    def rotor_speeds(self, t: int) -> np.ndarray:
        """Each rotor's speed at the start of control tick ``t`` [rad/s]."""
        return np.abs(self.state_at(t).joint_qd.numpy()[self.rotor_dofs])


class CapturedGradient:
    """The loss and gradient of a recorded rollout, replayed from one CUDA graph.

    On a CUDA device the first call captures the forward pass, the backward pass, the copy-out of the loss and the
    gradient, and the zeroing of every gradient for the next call, all in one graph, since the solver allocates per
    step and that memory exists only while the graph runs. Warp loads each kernel as the capture first launches it.
    Every call, the first too, then replays the graph, so the graph reads ``params`` as the caller last wrote it. On
    the CPU every call records eagerly.

    Args:
        forward: Runs the rollout and accumulates the loss into ``loss``; it reads ``params``.
        params: The array whose gradient the call returns.
        loss: The one-element loss array ``forward`` accumulates into.
    """

    def __init__(self, forward, params: wp.array, loss: wp.array):
        self.forward, self.params, self.loss = forward, params, loss
        self._grad = wp.zeros_like(params, requires_grad=False)
        self._loss = wp.zeros(1, dtype=float)
        self._graph = None

    def _record(self) -> wp.Tape:
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(self.loss)
        wp.copy(self._grad, self.params.grad)
        wp.copy(self._loss, self.loss)
        tape.zero()
        return tape

    def __call__(self) -> tuple[float, wp.array]:
        """Run the rollout once more: the loss, and the gradient as a device array the next call overwrites."""
        if not self.params.device.is_cuda:
            self._tape = self._record()
        else:
            if self._graph is None:
                with wp.ScopedCapture() as capture:
                    self._tape = self._record()
                self._graph = capture.graph
            wp.capture_launch(self._graph)
        return float(self._loss.numpy()[0]), self._grad
