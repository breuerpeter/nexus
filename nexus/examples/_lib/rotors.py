"""``Rotors``: the **one** actuator: ``N`` MotorModels, from :mod:`~nexus.examples._lib.motor`, + ``N``
PropellerModels, from :mod:`~nexus._src.vehicle.actuators.propeller`, + the summed-wrench **coupling** from
:mod:`~nexus.examples._lib.coupling`. It replaces the old two-class split of ``RigidBodyRotors`` +
``ArticulatedRotors``: there is no body-coupling axis baked into the class. Every consumer, RL train +
deploy, Proportional Integral Derivative (PID), policy, sampling Model Predictive Control (MPC),
design-opt, and the PX4 flight path on both runtimes, runs this actuator at the highest fidelity that
runs everywhere, airflow propeller + motor lag, summing the per-rotor wrench into one base-body wrench,
``state.body_f[base]``.

    per-rotor command u ∈ [0, 1]  →  Ω_cmd = clamp(u, 0, 1)·Ω_max  →  motor lag (Ω state)
    →  propeller f = kf·Ω² − airflow  →  forward ``B`` (Σ per-rotor → base-body wrench)  →  rotate to world

One input convention, the per-rotor command, and no modes; the mixer, rate loop / ``B⁻¹``, lives in the
controller. Built from a :class:`~nexus.examples._lib.mixer.RotorMixer`, the shared airframe ``B`` +
thrust map + rotor offsets, so the controller's ``B⁻¹`` and the actuator's forward ``B`` can't drift.

No cosmetic prop spin here: an articulated model that should render spinning props runs the core
:class:`~nexus._src.vehicle.actuators.articulated.ArticulatedRotors`; its rotor joints really turn. A
collapsed single body has no rotor joints to spin.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus.examples._lib.coupling import rigid_body_wrench_world
from nexus.examples._lib.mixer import RotorMixer
from nexus.examples._lib.motor import motor_alpha

# Single-body wrench actuator: pairs with any model; the pairing guard is a no-op for it.
requires_articulated = False


class Rotors:
    requires_articulated = False  # single base-body summed wrench: pairs with any model

    def __init__(
        self,
        *,
        mixer: RotorMixer,
        dt: float,
        # required: pass the vehicle's authored motor:tau from its Universal Scene Description (USD)
        # asset; no in-code default
        motor_tau: float,
        thrust_sign: float = -1.0,
        substeps: float = 1.0,
        aero_h: float = 0.0,
        aero_hforce: float = 0.0,
    ):
        self.nr = int(mixer.nr)
        self.base = int(mixer.base)
        self.kf = float(mixer.kf)
        self.omega_max_motor = float(mixer.omega_max_motor)
        self.thrust_sign = float(thrust_sign)
        self.substeps = float(substeps)
        self.aero_h = float(aero_h)  # forward-flight thrust loss; 0 = quasi-static kf·Ω²
        self.aero_hforce = float(aero_hforce)  # in-plane rotor H-force, drag
        self.dt = float(dt)
        self.alpha = motor_alpha(motor_tau, dt)  # first-order motor lag α from (τ, dt)
        self._B = wp.array(np.asarray(mixer.B, dtype=np.float32), dtype=float)  # (4, nr) forward allocation
        self._offsets = wp.array(np.asarray(mixer.rotor_offsets, dtype=np.float32), dtype=wp.vec3)  # (nr,) base-frame
        self._omega_state = wp.zeros((1, self.nr), dtype=float)  # (1, nr) per-rotor motor-speed state, updated in place
        self._cmd = wp.zeros((1, self.nr), dtype=float)  # (1, nr) per-rotor command buffer, host → device
        # No per-tick host op anywhere, so every deploy path stays device-native ⇒ the tick's device
        # region captures into a Compute Unified Device Architecture (CUDA) graph.
        self.capturable = True

    def write_controls(self, controls) -> None:
        """Host seam of the captured host-boundary strategy, one H2D copy: the peer's per-rotor commands →
        the persistent ``(1, nr)`` device buffer the captured graph reads. The capture recorded
        ``forces_wp`` over ``self._cmd``, so each replay picks up the fresh commands. Same slice as the
        eager path: PX4 streams 16-channel HIL_ACTUATOR_CONTROLS, the first ``nr`` are the rotor motors.
        """
        a = np.asarray(controls.command, dtype=np.float32).reshape(-1)
        if a.shape[0] < self.nr:
            raise ValueError(f"Rotors expects at least {self.nr} per-rotor commands, got {a.shape[0]}")
        self._cmd.assign(a[: self.nr].reshape(1, self.nr))

    def forces_wp(self, state) -> None:
        """The captured host-boundary device region: the persistent command buffer, which
        :meth:`write_controls` fills per tick, → motor lag → propeller → forward ``B`` → base-body wrench,
        no host ops.
        """
        self._launch_wrench(self._cmd, state)

    def _launch_wrench(self, cmd, state) -> None:
        """The capturable device region: per-rotor cmd → motor lag → propeller → forward ``B`` → base-body
        wrench, no host hop. ``cmd`` is the persistent ``(1, nr)`` buffer on the eager/host-boundary paths,
        or the controller's device-native ``(1, nr)`` Warp Controls when captured in-process, read straight
        in, so the loop joins a CUDA graph.
        """
        wp.launch(
            rigid_body_wrench_world,
            dim=1,
            inputs=(
                cmd,
                state.body_q,
                state.body_qd,
                self.base,
                self._offsets,
                self._B,
                self.kf,
                self.omega_max_motor,
                self.alpha,
                self.substeps,
                self.thrust_sign,
                self.aero_h,
                self.aero_hforce,
                self.nr,
                self._omega_state,  # in place, read before write per rotor: forward deploy
                self._omega_state,
            ),
            outputs=(state.body_f,),
        )

    def forces(self, controls, state, env=None) -> None:
        m = controls.command
        if isinstance(m, np.ndarray):  # eager host path: copy the per-rotor command H2D
            a = np.asarray(m, dtype=np.float32).reshape(-1)
            if a.shape[0] < self.nr:
                raise ValueError(f"Rotors expects at least {self.nr} per-rotor commands, got {a.shape[0]}")
            # PX4 streams 16-channel HIL_ACTUATOR_CONTROLS; the per-rotor mixers emit exactly nr. Take the
            # first nr either way, the rotor motors, matching the old ArticulatedRotors slice; the kernel
            # clamps each to [0, 1].
            self._cmd.assign(a[: self.nr].reshape(1, self.nr))
            cmd = self._cmd
        else:  # captured device-native path: a (1, nr) Warp array, the PID/policy controller's mixer output
            cmd = m
        self._launch_wrench(cmd, state)


# Back-compat alias: the unified actuator is the former single-body rigid-body actuator, now used
# everywhere. Existing ``RigidBodyRotors`` import sites keep working.
RigidBodyRotors = Rotors
