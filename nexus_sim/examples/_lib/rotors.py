"""``Rotors``: the examples' single-body actuator, the old actuator seam: ``N`` MotorModels, from
:mod:`~nexus_sim.examples._lib.motor`, + ``N`` PropellerModels, from :mod:`~nexus_sim._src.vehicle.forces.propellers`,
+ the summed-wrench **coupling** from :mod:`~nexus_sim.examples._lib.coupling`. Every consumer of the
single-body plant, RL train + deploy, Proportional Integral Derivative (PID), policy, sampling Model
Predictive Control (MPC) and design-opt, runs this actuator at the highest fidelity that runs everywhere,
airflow propeller + motor lag, summing the per-rotor wrench into one base-body wrench,
``state.body_f[base]``. It stays on the old seam, the loop's ``actuator``, until the single-body plant's
motor lag finds a home: it needs the controller's command, so it's neither a command stage, which writes
Newton's control inputs, nor a force element, which reads the state alone.

    per-rotor command u ∈ [0, 1]  →  Ω_cmd = clamp(u, 0, 1)·Ω_max  →  motor lag (Ω state)
    →  propeller f = kf·Ω² − airflow  →  forward ``B`` (Σ per-rotor → base-body wrench)  →  rotate to world

One input convention, the per-rotor command, and no modes; the mixer, rate loop / ``B⁻¹``, lives in the
controller. Built from a :class:`~nexus_sim.examples._lib.mixer.RotorMixer`, the shared airframe ``B`` +
thrust map + rotor offsets, so the controller's ``B⁻¹`` and the actuator's forward ``B`` can't drift.

No cosmetic prop spin here: an articulated model that should render spinning props runs the core rotor
chain, the rotors' command stage, the Newton motors physics steps and the propellers' force element; its
rotor joints really turn. A collapsed single body has no rotor joints to spin.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim.examples._lib.coupling import rigid_body_wrench_world
from nexus_sim.examples._lib.mixer import RotorMixer
from nexus_sim.examples._lib.motor import motor_alpha


class Rotors:
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

    def forces_wp(self, cmd, state) -> None:
        """The device stage: the controller's ``(1, n)`` command buffer → motor lag → propeller →
        forward ``B`` → base-body wrench, no host hop. The first ``nr`` commands are the rotor motors:
        PX4 streams 16-channel HIL_ACTUATOR_CONTROLS, the per-rotor mixers emit exactly ``nr``; the
        kernel clamps each to [0, 1].
        """
        if cmd.shape[1] < self.nr:
            raise ValueError(f"Rotors expects at least {self.nr} per-rotor commands, got {cmd.shape[1]}")
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

    def stages(self) -> list[Stage]:
        """One device stage over :meth:`forces_wp`, reading the controller's command buffer."""
        return [Stage("forces", "device", lambda tick: self.forces_wp(tick.controls, tick.state))]
