"""TrainedPolicyController: the trained-policy Controller seam.

Loads a policy exported by `nexus-rl`, Isaac Lab on the Newton backend, and runs it
*in place of external PX4*, closing the loop from training on Isaac Lab on Newton to
deployment on core. A trained policy is a **state-feedback** controller: it consumes the vehicle's
kinematics as a 12-D observation, not the MAVLink Hardware in the Loop (HIL) sensor bundle. It
builds that observation itself from the estimate, the base body's pose and twist an estimator writes,
with no observation sensor, using the single train↔deploy obs source
``nexus_sim.examples._lib.observation``.

The exported policy maps ``obs[16] -> action[4]`` = ``[thrust, ωx_cmd, ωy_cmd, ωz_cmd]``, the
Isaac-Lab quadcopter **Collective Thrust and Body Rate (CTBR)** action of collective thrust plus
body-rate setpoints, not raw moments. The observation is the 12-D kinematic obs optionally plus the
policy's last action, the per-rotor convergence fix. Whether a policy expects the augmented 16-D
form or the legacy 12-D form is a property **of the policy**, its input width, so the controller
**infers it from the loaded policy**; there is no deploy obs flag, because deploy conforms to the
trained policy.

The controller is **airframe-aware**, as a real flight controller is: its CTBR **mixer**,
:class:`~nexus_sim.examples._lib.mixer.CtbrMixer`, an inner rate loop → ``B⁻¹`` allocation,
turns the policy's CTBR action into the ``nr`` per-rotor commands ``Controls.command`` carries, which
the single-body :class:`~nexus_sim.examples._lib.rotors.Rotors` motor model then realizes. The mixer
is the same ``ctbr_to_cmd_batched`` op the training env launches with ``dim = N``, the byte-shared
train↔deploy parity surface. The rate loop needs only the body-frame rate ``ω_b``, which is exactly
``obs[3:6]``, which equals ``Rᵀ·ω_w``, so the controller needs no extra state plumbing.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_sim._src.core.schema import Controls, PoseTwist
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.core.stages import peer_stages
from nexus_sim.examples._lib.observation import ACTION_DIM, build_observation


class TrainedPolicyController:
    """Run an exported TorchScript RL policy as a Controller.

    Args:
        policy_path: path to the exported TorchScript ``policy.pt``.
        goal_w: target world position [m], the hover/move-to-position goal.
        action_clip: clamp applied to policy outputs; Isaac Lab clamps actions to [-1, 1].
        device: torch device for inference; ``"cpu"`` keeps the host stage on the CPU.
        mixer: the airframe :class:`~nexus_sim.examples._lib.mixer.RotorMixer`, ``B⁻¹`` plus the thrust
            map. With it the controller emits ``nr`` per-rotor commands; ``None`` emits the raw CTBR action,
            the legacy form.
        ctbr_params: the :class:`~nexus_sim.examples._lib.mixer.CtbrParams` for the inner rate loop, required iff
            ``mixer``. Must match the training env's CTBR params diluted to the deploy step: T/W and rate gain.
        body_index: the articulation body :meth:`obs_from_state` reads; 0 = base. In the loop the
            observation comes from the estimate, the base body's pose and twist.
    """

    def __init__(
        self,
        *,
        policy_path: str,
        goal_w=(0.0, 0.0, 1.0),
        action_clip: float = 1.0,
        device: str = "cpu",
        mixer=None,
        ctbr_params=None,
        body_index: int = 0,
    ):
        self.policy_path = str(policy_path)
        # The goal the goal-relative observation reads, an owned, mutable buffer: `np.array`, not asarray.
        # In the loop each exchange copies it from the setpoint, which a guidance writes, `goal_w` until
        # one does.
        self.goal_w = np.array(goal_w, dtype=np.float64)
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,), default=[goal_w])
        # The vehicle's estimate, the base body's pose and twist an estimator writes: the observation's source.
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))
        self.action_clip = float(action_clip)
        self.device = device
        self.body_index = int(body_index)
        # The airframe CTBR mixer, rate loop + B⁻¹ → per-rotor command, built on connect from the supplied
        # RotorMixer + CtbrParams. None ⇒ emit the raw CTBR action, for a legacy/raw-action consumer.
        self._rotor_mixer = mixer
        self._ctbr_params = ctbr_params
        self._mixer = None
        # The trained policy's obs is the 12-D kinematic obs + its own last action, POLICY_OBS_DIM wide,
        # the per-rotor convergence fix: the rotor speed lags the command and no observation carries it.
        # The controller owns the action history, the single augmentation site: obs_from_state folds
        # self._prev_action into the kinematic obs and act() stores the new action, assembled in one
        # place, the obs function, exactly as in training. Updated in place each act().
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._policy = None
        self._torch = None

    # -- lifecycle ---------------------------------------------------------------
    def connect(self) -> None:
        import torch  # lazy: keep obs/actuator import-light and CPU-testable

        self._torch = torch
        self._policy = torch.jit.load(self.policy_path, map_location=self.device)
        self._policy.eval()
        self._prev_action[:] = 0.0  # fresh action history per rollout, in place: the sensor shares it
        if self._rotor_mixer is not None and self._mixer is None:
            from nexus_sim.examples._lib import CtbrMixer

            # substeps = 1: the standalone applies the wrench once per step, the deploy effective CTBR params.
            self._mixer = CtbrMixer(self._rotor_mixer, self._ctbr_params, substeps=1.0)

    def close(self) -> None:
        self._policy = None

    def stages(self):
        """The ``exchange`` host stage: inference runs on the host between replays, over the
        estimate and toward the setpoint the exchange reads.
        """
        return peer_stages(self, reads=(self.setpoint, self.estimate))

    # -- inference ---------------------------------------------------------------
    def act(self, obs: np.ndarray) -> Controls:
        """Run the policy on the **complete** observation -> ``Controls`` carrying ``action[4]``.

        The obs function, :meth:`obs_from_state`, builds the observation whole and folds in this
        controller's last action; the controller doesn't reshape or augment it. After inference the
        new action lands **in place** in ``_prev_action``, so the next observation carries it, the
        deploy twin of the training env's ``observation_from_state(..., prev_action=self._actions)``.
        """
        if self._policy is None:
            raise RuntimeError("TrainedPolicyController.connect() must be called before act().")
        net_obs = np.asarray(obs, dtype=np.float32).reshape(-1)  # the complete obs the obs function built
        omega_b = net_obs[3:6]  # body-frame angular velocity, = Rᵀ·ω_w: the rate loop's only state input
        with self._torch.no_grad():
            action = self._policy(self._torch.from_numpy(net_obs.reshape(1, -1)).to(self.device))
        action = np.clip(action.cpu().numpy().reshape(ACTION_DIM), -self.action_clip, self.action_clip)
        self._prev_action[:] = action.astype(np.float32)  # in place -> the shared sensor obs reads it next tick
        if self._mixer is None:
            return Controls(command=action.astype(np.float32))  # raw CTBR action, for a legacy / raw-action consumer
        # The airframe CTBR mixer: action + body rate → nr per-rotor commands, the deploy half of the
        # byte-shared train↔deploy mixer. Host path, because the policy runs in a host stage on the CPU, so read back numpy.
        cmd = self._mixer.cmd(action, omega_b)
        return Controls(command=cmd.numpy().reshape(-1).astype(np.float32))

    def obs_from_state(self, state, goal_w=None) -> np.ndarray:
        """Build the complete observation from the live ground-truth ``newton.State``: the kinematic obs
        with this controller's last action folded in, the single train↔deploy obs function.
        """
        i = self.body_index
        goal = self.goal_w if goal_w is None else np.asarray(goal_w, dtype=np.float64)
        bq = state.body_q.numpy()[i]  # [px,py,pz, qx,qy,qz,qw], xyzw order, world frame
        bqd = state.body_qd.numpy()[i]  # [lin(0:3), ang(3:6)], world frame, at the Center of Mass (COM)
        return build_observation(
            pos_w=bq[0:3],
            quat_xyzw=bq[3:7],
            lin_vel_w=bqd[0:3],
            ang_vel_w=bqd[3:6],
            goal_w=goal,
            prev_action=self._prev_action,
        )

    def act_from_state(self, state, goal_w=None) -> Controls:
        """Primary entry: ground-truth state -> observation -> policy -> Controls."""
        return self.act(self.obs_from_state(state, goal_w))

    # -- Controller Protocol conformance ----------------------------------------
    def exchange(self, t, timeout=None) -> Controls | None:
        """Protocol-conforming step: flies through ``Orchestrator.run()`` exactly as PX4 does.

        Builds the complete observation from the estimate, the base body's pose and twist an estimator
        writes, through :func:`~nexus_sim.examples._lib.observation.build_observation`, the single
        train↔deploy obs source, last action folded in. In the loop, the goal is the setpoint's: the
        exchange copies it to the host first.
        """
        if self.setpoint.buffer is not None:  # wired by the loop: fly to the goal the guidance writes
            self.goal_w[:] = self.setpoint.read()[0]
        est = self.estimate.read()[0]  # world frame
        obs = build_observation(
            pos_w=est["position"],
            quat_xyzw=est["orientation"],
            lin_vel_w=est["linear_velocity"],
            ang_vel_w=est["angular_velocity"],
            goal_w=self.goal_w,
            prev_action=self._prev_action,
        )
        return self.act(obs)
