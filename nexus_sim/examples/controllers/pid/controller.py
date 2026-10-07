"""PidController: the **built-in simple controller**, an in-process, fully deterministic
Proportional Integral Derivative (PID) flown *in place of external PX4*.

Two roles, one control law, ``law.pid_action_np`` and ``law.pid_law``:

1. **Determinism authority.** A deterministic in-process controller over the bit-exact
   Newton CPU physics is what closes the bit-reproducibility gap left open with real PX4,
   whose multi-threaded work-queue interleaving isn't the same bit for bit. Flown through the unchanged
   :class:`~nexus_sim._src.core.orchestrator.Orchestrator` it gives a CI determinism gate that does
   not wait on PX4.

2. **The design-optimization controller.** Its gains are the differentiable design parameters the
   design-optimization example tunes: the *same* law, in Warp.

It's a state-feedback controller, as :class:`TrainedPolicyController` is: it reads the 12-D observation a
:class:`PolicyObservationSensor` writes to ``meas.observation``. The control law emits a
direct-moment action ``[thrust, m_x, m_y, m_z]``; the controller's **moment mixer**, ``B⁻¹`` allocation
with no rate loop, :class:`~nexus_sim.examples._lib.mixer.MomentMixer`, then turns it into the ``nr``
per-rotor commands the single-body :class:`~nexus_sim.examples._lib.rotors.Rotors`
motor model consumes. So PID, policy, and PX4 all fly through the same orchestrator tick, all emitting
``Controls.command`` = per-rotor commands; only the host boundary differs.

The mixer is optional: the differentiable design-optimization rollout configures the controller with no
mixer and consumes the raw moment action, ``act_wp``, directly, composing the shared mixer + motor-model
kernels on its own tape, so the gains stay the differentiable leaf.
"""

from __future__ import annotations

import numpy as np

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import Controls, PositionGoal
from nexus_sim._src.core.signals import Signal

from .law import DEFAULT_GAINS, hover_action, pid_action_np


class PidController:
    """Deterministic in-process PID controller, built-in.

    Args:
        gains: the PID gain vector, see ``law.GAIN_NAMES``; defaults to ``law.DEFAULT_GAINS``.
        goal_w: target world position [m], the waypoint to fly to from hover.
        thrust_to_weight: the action scale, the plant's true T/W, derived from the Universal Scene
            Description (USD) via ``RotorMixer.thrust_to_weight``. Sets the hover thrust bias so
            ``action[0]`` holds altitude at the gravity-balancing thrust. Must match the paired actuator +
            the mixer.
        mixer: the airframe :class:`~nexus_sim.examples._lib.mixer.RotorMixer`, ``B⁻¹`` + thrust map.
            When given, the deploy path, the controller emits ``nr`` per-rotor commands; when ``None``, the
            differentiable design-opt rollout, it emits the raw moment action and the caller mixes.
        weight: vehicle weight ``m·g`` [N] for the moment mixer's collective scale; required iff ``mixer``.
        moment_scale: N·m per unit moment action; must match the design-opt and deploy convention.
        body_index: articulation body whose state drives the observation; 0 = base.
    """

    def __init__(
        self,
        *,
        gains=DEFAULT_GAINS,
        goal_w=(0.0, 0.0, 1.0),
        thrust_to_weight: float,  # required: the action scale; pass the plant's USD-derived T/W, mixer.thrust_to_weight
        mixer=None,
        weight: float = 0.0,
        moment_scale: float = 0.01,
        body_index: int = 0,
    ):
        self.gains = np.asarray(gains, dtype=np.float32)
        # Owned goal on the host: obs_from_state and the exchange fallback read it.
        self.goal_w = np.array(goal_w, dtype=np.float64)
        # The goal the loop flies to: the setpoint a guidance writes, `goal_w` until one does. The
        # observation sensor's kernel reads it, since the observation is still a sensor of its own.
        self.setpoint = Signal("setpoint", PositionGoal, shape=(1,), default=[goal_w])
        self.thrust_to_weight = float(thrust_to_weight)
        self.hover = hover_action(thrust_to_weight)
        self.body_index = int(body_index)
        # The airframe moment mixer, B⁻¹ → per-rotor command, built lazily on first device use from the
        # supplied RotorMixer. None ⇒ law-only; design-opt composes the mixer itself.
        self._rotor_mixer = mixer
        self.weight = float(weight)
        self.moment_scale = float(moment_scale)
        self._moment_mixer = None
        # The per-rotor commands the loop flies, which the moment mixer writes: one per rotor of the airframe.
        self.controls = Signal("controls", Controls, shape=(1, mixer.nr)) if mixer is not None else None
        # Device-native state, lazily allocated on first Warp use; keeps the host/eager-numpy path
        # warp-free to import. ``gains_wp`` is the differentiable leaf the optimizer descends,
        # and can replace; ``_action_wp`` is a persistent moment buffer, static address -> graph-safe.
        self.gains_wp = None
        self._action_wp = None

    # -- lifecycle; no I/O: an in-process controller has nothing to connect to --
    def connect(self) -> None:
        """Reserve the device buffers at the pre-run lifecycle seam; ``connect`` runs before any
        CUDA-graph capture. Creating them lazily in the first ``exchange`` put the allocations
        inside the captured graph, graph-owned memory that nothing must ever reference across replays;
        it read as stable until some other consumer, the Kit renderer, allocated between replays,
        then the mixer's buffers silently diverged from the graph's and the flight froze.
        """
        self._ensure_wp()

    def close(self) -> None:
        pass

    # -- inference --------------------------------------------------------------
    def act(self, obs: np.ndarray) -> Controls:
        """Run the PID law on a 12-D observation -> ``Controls``. With an airframe mixer the controls are
        the ``nr`` per-rotor commands, mixed via the device-native moment mixer; without one, the
        design-opt law-only path, they're the raw moment action ``[thrust, m_x, m_y, m_z]``.
        """
        moments = pid_action_np(obs, self.gains, self.hover)
        if self._rotor_mixer is None:
            return Controls(command=moments.astype(np.float32))  # law-only; the differentiable rollout mixes
        self._ensure_wp()
        self._action_wp.assign(moments.astype(np.float32))
        cmd = self._moment_mixer.cmd_wp(self._action_wp)
        return Controls(command=cmd.numpy().reshape(-1).astype(np.float32))

    def _ensure_wp(self):
        """Create the Warp gains leaf + persistent moment buffer + the moment mixer on first device use."""
        import warp as wp

        from nexus_sim.examples._lib import MomentMixer

        if self.gains_wp is None:
            self.gains_wp = wp.array(self.gains, dtype=float, requires_grad=True)
        if self._action_wp is None:
            self._action_wp = wp.zeros(4, dtype=float)
        if self._moment_mixer is None and self._rotor_mixer is not None:
            self._moment_mixer = MomentMixer(
                self._rotor_mixer,
                thrust_to_weight=self.thrust_to_weight,
                weight=self.weight,
                moment_scale=self.moment_scale,
            )

    def act_wp(self, obs_wp, out_action_wp):
        """Device-native step: launch the shared ``pid_law`` kernel on Warp arrays, with no host
        round-trip, so the controller records on a ``wp.Tape`` or joins a CUDA graph. Uses
        ``self.gains_wp``, the differentiable leaf; the optimizer reads its ``.grad`` and can
        swap the array. ``out_action_wp`` is caller-provided, so the eager loop passes a persistent
        buffer, graph-safe, and the differentiable rollout passes a per-step buffer, the tape history.
        """
        import warp as wp  # lazy: keep the host path, eager and determinism, import-light

        from .law import pid_law

        self._ensure_wp()
        wp.launch(pid_law, dim=1, inputs=(obs_wp, self.gains_wp, self.hover), outputs=(out_action_wp,))
        return out_action_wp

    def obs_from_state(self, state, goal_w=None) -> np.ndarray:
        from nexus_sim.examples._lib.observation import build_observation

        i = self.body_index
        goal = self.goal_w if goal_w is None else np.asarray(goal_w, dtype=np.float64)
        bq = state.body_q.numpy()[i]  # [px,py,pz, qx,qy,qz,qw], xyzw order, world
        bqd = state.body_qd.numpy()[i]  # [lin(0:3), ang(3:6)], world, at the Center Of Mass (COM)
        return build_observation(
            pos_w=bq[0:3],
            quat_xyzw=bq[3:7],
            lin_vel_w=bqd[0:3],
            ang_vel_w=bqd[3:6],
            goal_w=goal,
        )

    def act_from_state(self, state, goal_w=None) -> Controls:
        return self.act(self.obs_from_state(state, goal_w))

    # -- Controller Protocol conformance ----------------------------------------
    def exchange(self, meas, t, timeout=None) -> Controls | None:
        """Protocol step: flies through ``Orchestrator.run()`` exactly as PX4 and the policy do.

        Reads the 12-D observation a ``*ObservationSensor`` writes to ``meas.observation``; falls
        back to a bound state provider. Never returns ``None``, since an in-process controller doesn't
        time out, so a finite run sets ``Orchestrator(max_steps=...)``.

        **Device-native when the observation is a Warp array**, from ``WarpObservationSensor``: runs
        ``act_wp`` into the persistent action buffer and returns Warp ``Controls``, with no per-tick host
        hop, so the whole tick stays one graph and tape-able. A NumPy observation, from the torch
        ``PolicyObservationSensor`` or the provider fallback, takes the host ``act`` path.
        """
        obs = getattr(meas, "observation", None)
        if obs is None:
            provider = getattr(self, "_state_provider", None)
            if provider is None:
                raise RuntimeError(
                    "PidController.exchange() needs the 12-D observation. Add an observation sensor "
                    "(it fills meas.observation), or bind_state_provider(fn), or call act_from_state(state)."
                )
            from nexus_sim.examples._lib.observation import build_observation

            pos_w, quat_xyzw, lin_w, ang_w = provider()
            obs = build_observation(pos_w, quat_xyzw, lin_w, ang_w, self.goal_w)
        if isinstance(obs, np.ndarray):
            return self.act(obs)  # host path, numpy obs
        self._ensure_wp()  # device-native path, Warp obs: no host round-trip
        self.act_wp(obs, self._action_wp)  # law → moments, device-native
        if self._moment_mixer is None:
            return Controls(command=self._action_wp)  # law-only; no airframe mixer configured
        # Mix moments → per-rotor commands on-device, so the whole tick stays one graph.
        return Controls(command=self._moment_mixer.cmd_wp(self._action_wp))

    def stages(self) -> list[Stage]:
        """One device stage, ``act``: the law over the observation sensor's device buffer, then the
        moment mixer into the controls, the ``(1, nr)`` per-rotor commands the actuator reads. It reads the
        setpoint, the goal of that observation. The whole tick stays one graph.

        Raises:
            RuntimeError: The controller has no airframe mixer, so it has no per-rotor commands to fly.
        """
        if self._rotor_mixer is None:
            raise RuntimeError("PidController flies through the loop only with an airframe mixer")
        return [Stage("act", "device", self._act_stage, reads=(self.setpoint,), writes=(self.controls,))]

    def _act_stage(self, tick) -> None:
        self._ensure_wp()
        self.act_wp(tick.meas.observation, self._action_wp)  # law → moments, device-native
        self._moment_mixer.cmd_wp(self._action_wp, out=self.controls.buffer)  # moments → per-rotor commands

    def bind_state_provider(self, fn) -> None:
        """Bind a callable returning ``(pos_w, quat_xyzw, lin_vel_w, ang_vel_w)`` for :meth:`exchange`."""
        self._state_provider = fn
