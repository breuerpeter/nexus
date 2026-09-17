"""``ArticulatedRotors``: rotor *motors* as first-class ``newton.actuators``, aero as nexus's kernel.

The core, PX4-path, actuator: each rotor's motor is a **Universal Scene Description (USD)-authored**
``newton.actuators`` composition: a ``NewtonActuator`` prim targeting the rotor revolute joint, with
``NewtonPIDControlAPI``, Proportional Integral Derivative (PID) control with ``kp=ki=0`` and ``kd``
only, so a velocity servo,
``effort = kd·(Ω_cmd − Ω) + feedforward``, clamped by ``NewtonDCMotorClampingAPI``, the
four-quadrant torque–speed envelope. ``ModelBuilder.add_usd`` parses the prims onto
``model.actuators`` automatically: the motor and its constants travel with the asset, and there is
NO in-code fallback; an unauthored USD fails loudly, as the sensor suite does. The rotor speed Ω is a
real, solver-integrated joint state with physical lag + saturation, and the props render spinning
at no extra cost: the joints actually turn; no cosmetic spin kernel.

The split mirrors what the Newton actuator framework can and can't do: it's joint-space only,
``joint_q/joint_qd → joint_f``, Single Input Single Output (SISO), so it models the *motor*; the
**aerodynamics** stay nexus's Warp ``body_f`` kernel, :func:`aero_from_omega`: each rotor's
solver-integrated Ω produces the airflow-aware thrust + in-plane H-force, through
:func:`~nexus._src.vehicle.actuators.propeller.propeller_force`, the same propeller model every path runs,
and the reaction drag torque, applied to the *rotor* body.
The revolute joint transmits the lift to the airframe and the motor's reaction torque yaws it;
putting the drag on the base would double-count the motor reaction, the runaway-yaw lesson from the
original conformed-actuator validation.

Fully **capturable**: the host seam, ``write_controls``, does one small H2D of the per-rotor Ω
targets + drag feedforward; the device region, ``forces_wp``, is all kernels: the target scatter,
``joint_f`` zero, the motor step, where ``ControllerPID`` is graphable, an in-place actuator-state
copy-back, the graph-safe twin of the double-buffer swap, as in physics' state swap, and the aero
kernel.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus._src.vehicle.actuators.layout import RPM_PER_RADS, find_rotor_joints
from nexus._src.vehicle.actuators.propeller import propeller_force


@wp.kernel
def _scatter_rotor_commands(
    rotor_vel_dofs: wp.array(dtype=wp.int32),
    omega_cmd: wp.array(dtype=float),  # (nr,) target rotor speed [rad/s]
    drag_ff: wp.array(dtype=float),  # (nr,) feedforward effort [N·m], cancels the steady aero drag
    out_target_qd: wp.array(dtype=float),  # control.joint_target_qd
    out_joint_act: wp.array(dtype=float),  # control.joint_act, the PID feedforward slot
):
    """Scatter the per-rotor velocity target + drag feedforward into the model-sized control arrays,
    device-side, so the captured graph replays it from the persistent ``omega_cmd``/``drag_ff``
    buffers ``write_controls`` refreshes at the host seam.
    """
    i = wp.tid()
    d = rotor_vel_dofs[i]
    out_target_qd[d] = omega_cmd[i]
    out_joint_act[d] = drag_ff[i]


@wp.kernel
def aero_from_omega(
    kf: float,  # ct·RPM_PER_RADS²: static thrust = kf·Ω²
    cd: float,  # reaction-torque-per-thrust, the yaw allocation κ
    aero_h: float,  # forward-flight thrust loss; 0 = quasi-static
    aero_hforce: float,  # in-plane rotor H-force, drag
    base_body: int,  # airframe body index, the body-z thrust reference
    rotor_vel_dofs: wp.array(dtype=wp.int32),  # (nr,) Degrees Of Freedom (DOF) index of each rotor in joint_qd
    rotor_bodies: wp.array(dtype=wp.int32),  # (nr,) body index of each rotor
    joint_qd: wp.array(dtype=float),  # solver-integrated generalized velocities; rotor Ω lives here
    body_q: wp.array(dtype=wp.transform),  # per-body poses
    body_qd: wp.array(dtype=wp.spatial_vector),  # per-body world twists, the rotor inflow
    out_body_f: wp.array(dtype=wp.spatial_vector),  # world per-body wrench: force top / torque bottom
):
    """Per-rotor airflow-aware thrust + H-force + reaction drag from the SOLVER-INTEGRATED rotor
    speed Ω, applied to the rotor body, single-counted through the joint; see the module docstring.
    """
    i = wp.tid()
    omega = joint_qd[rotor_vel_dofs[i]]  # rad/s, signed; Ω² terms are sign-independent
    bi = rotor_bodies[i]
    q_rot = wp.transform_get_rotation(body_q[bi])
    base_z_world = wp.quat_rotate(wp.transform_get_rotation(body_q[base_body]), wp.vec3(0.0, 0.0, 1.0))
    rotor_z_world = wp.quat_rotate(q_rot, wp.vec3(0.0, 0.0, 1.0))
    # In-plane inflow at the rotor: its world linear velocity in the rotor frame, x and y components.
    v_rot = wp.quat_rotate_inv(q_rot, wp.spatial_top(body_qd[bi]))
    f_b = propeller_force(omega, v_rot[0], v_rot[1], kf, aero_h, aero_hforce)  # (h_x, h_y, thrust_mag)
    # rotor-z encodes the USD-authored cw/ccw spin direction; thrust always points "up" off the body.
    thrust_sign = -wp.sign(wp.dot(rotor_z_world, base_z_world))
    force_world = wp.quat_rotate(q_rot, wp.vec3(f_b[0], f_b[1], thrust_sign * f_b[2]))
    drag_world = -f_b[2] * cd * rotor_z_world  # reaction drag torque opposes the spin, via rotor-z
    # Both on the rotor body: force in the top slot, w, torque in the bottom slot, v.
    out_body_f[bi] = wp.spatial_vector(w=force_world, v=drag_world)


def _state_arrays(act_state) -> list[wp.array]:
    """The Warp arrays inside a composed ``newton.actuators`` Actuator.State, the delay + controller
    sub-states, in a construction-stable order, so two states built by the same ``actuator.state()``
    pair up positionally for the in-place copy-back.
    """
    out: list[wp.array] = []
    if act_state is None:
        return out
    for sub in (getattr(act_state, "delay_state", None), getattr(act_state, "controller_state", None)):
        if sub is None:
            continue
        for v in vars(sub).values():
            if isinstance(v, wp.array):
                out.append(v)
    return out


class ArticulatedRotors:
    """Step the model's USD-authored rotor motors, ``newton.actuators``, + nexus's aero kernel.

    Args:
        model: The finalized articulated model. Must carry ``model.actuators``; the vehicle USD
            authors one ``NewtonActuator`` per rotor joint. No fallback.
        control: The model's persistent ``newton.Control``; physics owns it. The motors write
            ``control.joint_f``, and the solver consumes it with the aero ``body_f`` in one solve.
        ct: Thrust coefficient [N/rpm²], ``propeller:ct``, the vehicle USD authority.
        cd: Reaction-torque-per-thrust, ``propeller:cd``, the yaw allocation κ.
        rpm_max: Rotor speed at normalized command 1.0 [rpm], ``propeller:rpm_max``.
        dt: Control timestep [s].
        aero_h: Forward-flight thrust-loss coefficient, ``propeller:aero_h``; 0 = quasi-static.
        aero_hforce: In-plane rotor H-force coefficient, ``propeller:aero_hforce``.
    """

    requires_articulated = True  # rotor joints *are* the actuator; the pairing guard enforces it
    capturable = True  # the device region is all kernels; the host seam is one small H2D

    def __init__(self, *, model, control, ct: float, cd: float, rpm_max: float, dt: float,
                 aero_h: float = 0.0, aero_hforce: float = 0.0):  # fmt: skip
        if not getattr(model, "actuators", None):
            raise ValueError(
                "ArticulatedRotors requires rotor motors on model.actuators: author them as "
                "NewtonActuator prims (NewtonPIDControlAPI + NewtonDCMotorClampingAPI) on the vehicle "
                "USD's rotor joints (see scripts/assets/author_astro_variants.py); add_usd parses them "
                "automatically. There is no in-code fallback (the USD is the single authority)."
            )
        self.model = model
        self.control = control
        self.dt = float(dt)
        self.kf = float(ct) * RPM_PER_RADS * RPM_PER_RADS  # thrust = kf·Ω², with Ω in rad/s
        self.cd = float(cd)
        self.ct = float(ct)
        self.aero_h = float(aero_h)
        self.aero_hforce = float(aero_hforce)
        self.omega_max = float(rpm_max) / RPM_PER_RADS  # rotor speed [rad/s] at normalized command 1.0
        vel_dofs, _pos_coords, bodies, base_body = find_rotor_joints(model)
        self.nr = len(vel_dofs)
        self.base = int(base_body)
        self._rotor_dofs = wp.array(vel_dofs, dtype=wp.int32)
        self._rotor_bodies = wp.array(bodies, dtype=wp.int32)
        # Persistent small device buffers for the per-rotor command scatter, the one host→device seam.
        self._omega_cmd = wp.zeros(self.nr, dtype=float)
        self._drag_ff = wp.zeros(self.nr, dtype=float)
        # Double-buffered per-actuator state, because ControllerPID is stateful. Under CUDA-graph capture a
        # Python buffer swap would freeze at the capture-time binding, so forces_wp copies next→current
        # in place after each step, the graph-safe twin of physics' state swap; the states are small:
        # the PID integral per DOF.
        self._states = [(a.state(), a.state()) for a in model.actuators]
        self._state_copies = [
            list(zip(_state_arrays(cur), _state_arrays(nxt), strict=True)) for cur, nxt in self._states
        ]

    # -- host seam, one H2D per tick ---------------------------------------------------------
    def write_controls(self, controls) -> None:
        """Per-rotor normalized ``[0, 1]`` commands → rotor-speed targets + the steady-state drag
        feedforward, into the persistent device buffers the captured graph reads. All-positive Ω
        targets; rotor-z encodes cw/ccw, so yaw balances through the reaction torques.
        """
        m = np.asarray(controls.command, dtype=np.float32).reshape(-1)
        if m.shape[0] < self.nr:
            raise ValueError(f"ArticulatedRotors expects at least {self.nr} per-rotor commands, got {m.shape[0]}")
        m = np.clip(m[: self.nr], 0.0, 1.0)
        omega_cmd = m * self.omega_max
        # The aero braking torque at the target speed, τ = cd·thrust = cd·kf·Ω², fed forward so the
        # kd-only servo doesn't droop against the steady load.
        drag_ff = self.cd * self.kf * omega_cmd * omega_cmd
        self._omega_cmd.assign(omega_cmd.astype(np.float32))
        self._drag_ff.assign(drag_ff.astype(np.float32))

    # -- device region, in-graph -------------------------------------------------------------
    def forces_wp(self, state) -> None:
        """The captured device region: scatter targets → zero ``joint_f`` → step the motors into
        ``control.joint_f`` → in-place state copy-back → aero into ``state.body_f``. The solver then
        consumes ``joint_f`` + ``body_f`` in one solve.
        """
        wp.launch(
            _scatter_rotor_commands,
            dim=self.nr,
            inputs=(self._rotor_dofs, self._omega_cmd, self._drag_ff),
            outputs=(self.control.joint_target_qd, self.control.joint_act),
        )
        self.control.joint_f.zero_()
        for (actuator, (cur, nxt)), copies in zip(
            zip(self.model.actuators, self._states, strict=True), self._state_copies, strict=True
        ):
            actuator.step(state, self.control, cur, nxt, self.dt)
            for dst, src in copies:  # current ← next, in place: the graph-safe double-buffer
                wp.copy(dst, src)
        wp.launch(
            aero_from_omega,
            dim=self.nr,
            inputs=(
                self.kf,
                self.cd,
                self.aero_h,
                self.aero_hforce,
                self.base,
                self._rotor_dofs,
                self._rotor_bodies,
                state.joint_qd,
                state.body_q,
                state.body_qd,
            ),
            outputs=(state.body_f,),
        )

    def forces(self, controls, state, env=None) -> None:
        """Eager path, the ``Sensor``-protocol twin of the captured seam: one host write + the device region."""
        self.write_controls(controls)
        self.forces_wp(state)


__all__ = ["ArticulatedRotors", "aero_from_omega"]
