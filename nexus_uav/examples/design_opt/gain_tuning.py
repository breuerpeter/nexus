"""Controller gain tuning by differentiable simulation: back-propagate through a closed-loop
hover→waypoint flight to **auto-tune the Proportional Integral Derivative (PID) gains**.

Self-contained example, as NVIDIA's ``example_diffsim_*`` are: it defines its own differentiable rollout
inline and imports only reusable framework bits, namely the shared PID law, the observation and mixer kernels,
and the articulated rollout. :class:`AstroMaxWaypointRollout` flies the articulated astro-max, the plant every
run flies: Newton's motors spin its rotor joints, and its propellers turn each rotor's speed into its force, under
``SolverSemiImplicit``. The PID law's collective and moments reach the rotors through the **moment mixer**, with
**no Collective Thrust and Body Rate (CTBR) rate loop**. Gradient descent on the PID gains makes it fly a waypoint
and settle, with an **exact full-horizon gradient**, checked against finite differences. The rate-loop-free
front-end is what keeps the long-horizon Backpropagation Through Time (BPTT) bounded; the CTBR variant's inner
rate loop pushes the closed-loop spectral radius >1 and explodes it.

The companion ``mass_recovery.py`` is the other differentiable-design gate, physical-parameter system-ID.

Run:

    uv run --extra examples -m nexus_uav.examples gain_tuning            # asserts gradient-correct + tuned
    uv run --extra examples -m nexus_uav.examples gain_tuning --log      # also write the optimized-flight .rrd

Runs on Compute Unified Device Architecture (CUDA) when present, where each gradient replays from one captured
graph, else on the CPU backend, much slower.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

import nexus_uav as nx
from nexus_uav.examples._lib import (
    RunClock,
    build_rotor_mixer_from_model,  # airframe B / B⁻¹ + thrust map from the model's rotor joints
    dump_run,
    moment_to_cmd_batched,  # the moment mixer: collective + moments → B⁻¹ → per-rotor command, no rate loop
    pack_vec4,  # pack the length-4 moment action into the wp.vec4 the mixer reads
)
from nexus_uav.examples._lib.articulated import ArticulatedRollout, CapturedGradient
from nexus_uav.examples._lib.observation import WarpObservationSensor
from nexus_uav.examples.controllers.pid import NUM_GAINS, PidController

GRAVITY = 9.81


@wp.kernel
def waypoint_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body: int,
    goal: wp.vec3,
    vel_weight: float,
    time_weight: float,
    cost: wp.array(dtype=float),
):
    """Per-step cost: distance-to-waypoint + velocity, which penalizes overshoot, ramped in time so
    later steps weigh more, which penalizes slow convergence, the time-to-reach.
    """
    p = wp.transform_get_translation(body_q[body])
    v = wp.spatial_top(body_qd[body])
    d = p - goal
    wp.atomic_add(cost, 0, time_weight * (wp.dot(d, d) + vel_weight * wp.dot(v, v)))


class AstroMaxWaypointRollout:
    """Differentiable hover→waypoint rollout on the **articulated astro-max**, the plant every run flies.

    Each control tick, the shared PID law turns the observation into a collective and three moments, the
    **moment mixer** :func:`moment_to_cmd_batched` turns those into a normalized command per rotor through
    ``B^-1``, with NO rate loop, and :class:`ArticulatedRollout` flies the commands: Newton's motors spin the
    rotor joints toward their speed targets, and the propellers turn each rotor's speed into its force. So the
    optimizer tunes the gains on the plant they deploy on.

    Why the *moment* mixer, not the CTBR one: the CTBR inner rate loop, the ``omega_des - omega_b``
    feedback, pushes the closed-loop BPTT spectral radius >1, so its full-horizon gradient explodes, with a
    measured cosine of ~0.15 at 400 steps. The moment-input front-end skips the rate loop, feeding the
    controller's collective + moments straight into the allocation and the motors, which keeps a clean
    full-horizon gradient. The moment interface also *matches* the original control, since the old
    ``pid_to_rotor_rpms`` took ``[collective, roll, pitch, yaw]`` moments, so the shared ``_pid`` law + its
    gains transfer.
    """

    def __init__(
        self,
        *,
        n_steps: int = 250,
        dt: float = 0.004,
        goal=(1.5, 1.0, 3.0),
        start=(0.0, 0.0, 2.0),
        thrust_to_weight: float | None = None,  # None -> the plant's true T/W from the USD thrust map
        moment_scale: float = 0.06,
        vel_weight: float = 0.2,
        ramp_time_weight: bool = True,
    ):
        from nexus_uav._src.build.launch import resolve_vehicle_usd
        from nexus_uav._src.config import LaunchConfig

        vehicle_usd, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"))
        self.rollout = ArticulatedRollout(vehicle_usd, ticks=n_steps, dt=dt, start=start)
        model = self.rollout.model
        self.N = int(n_steps)
        self.dt = float(dt)
        self.goal_np = np.asarray(goal, dtype=np.float32)
        self.goal = wp.vec3(*[float(x) for x in goal])
        self.vel_weight = float(vel_weight)
        self.ramp_time_weight = bool(ramp_time_weight)
        self.weight = self.rollout.mass * GRAVITY
        self.moment_scale = float(moment_scale)

        # Airframe mixer: control allocation B, per-rotor thrust -> [T, tx, ty, tz], + its inverse B^-1, from the
        # model's rotor joints at rest, the same builder the deploy flight uses.
        rest = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, rest)
        mixer = build_rotor_mixer_from_model(model, vehicle_usd.rotors(), rest.body_q.numpy())
        self.nr = mixer.nr
        # The action scale = the plant's true T/W from the USD thrust map, no declared constant:
        # the same derivation the deploy build uses, so tuned gains deploy on the same scale.
        self.t2w = (
            float(thrust_to_weight) if thrust_to_weight is not None else mixer.thrust_to_weight(self.rollout.mass)
        )
        self.B_inv_wp = wp.array(mixer.B_inv.astype(np.float32), dtype=float)  # mixer B^-1, the allocation
        self.kf = wp.array(mixer.kf.astype(np.float32), dtype=float)  # each rotor's thrust map
        self.omega_max_motor = wp.array(mixer.omega_max_motor.astype(np.float32), dtype=float)

        # Reused loop components + full tape history, one array per tick.
        self.obs_sensor = WarpObservationSensor(goal_w=goal, body_index=self.rollout.base)
        self.controller = PidController(goal_w=goal, thrust_to_weight=self.t2w)
        self.obs = [wp.zeros(12, dtype=float, requires_grad=True) for _ in range(self.N)]
        self.acts = [wp.zeros(4, dtype=float, requires_grad=True) for _ in range(self.N)]  # [collective, m_x, m_y, m_z]
        self.act4 = [wp.zeros(1, dtype=wp.vec4, requires_grad=True) for _ in range(self.N)]  # packed moment action
        self.cmd = [wp.zeros((1, self.nr), dtype=float, requires_grad=True) for _ in range(self.N)]  # per-rotor command
        self.rollout.reset()

    @property
    def model(self):
        """The articulated model the rollout flies."""
        return self.rollout.model

    @property
    def states(self) -> list:
        """Every solver step's state, the start first."""
        return self.rollout.states

    def _rollout_step(self, t: int, cost: wp.array | None, record: list | None = None) -> None:
        """One control tick: obs -> shared PID law, collective + moments -> moment mixer, B⁻¹ → per-rotor
        command with NO rate loop -> the articulated plant's solver steps -> optional cost.
        """
        state = self.rollout.state_at(t)
        self.obs_sensor.sample_wp(state, self.obs[t])
        self.controller.act_wp(self.obs[t], self.acts[t])  # obs -> [collective, roll, pitch, yaw] moments
        wp.launch(pack_vec4, dim=1, inputs=(self.acts[t],), outputs=(self.act4[t],))
        # Moment mixer: collective + moments -> B⁻¹ -> per-rotor command u, the differentiable allocation.
        wp.launch(
            moment_to_cmd_batched,
            dim=1,
            inputs=(
                self.act4[t],
                self.t2w,
                self.weight,
                self.moment_scale,
                self.B_inv_wp,
                self.kf,
                self.omega_max_motor,
                self.nr,
            ),
            outputs=(self.cmd[t],),
        )
        self.rollout.tick(t, self.cmd[t])
        nxt = self.rollout.state_at(t + 1)
        if cost is not None:
            tw = (float(t + 1) / float(self.N)) if self.ramp_time_weight else 1.0
            wp.launch(
                waypoint_cost,
                dim=1,
                inputs=(nxt.body_q, nxt.body_qd, self.rollout.base, self.goal, self.vel_weight, tw),
                outputs=(cost,),
            )
        if record is not None:
            record.append(nxt.body_q.numpy()[self.rollout.base, :3].copy())

    def forward(self, gains: wp.array, cost: wp.array) -> None:
        """Full-horizon closed-loop rollout accumulating the scalar waypoint cost. Run inside a tape for
        the exact full-horizon gradient w.r.t. ``gains``, the controller's differentiable leaf.
        """
        self.controller.gains_wp = gains  # the differentiable design parameter, the optimizer's leaf
        cost.zero_()
        for t in range(self.N):
            self._rollout_step(t, cost)

    # -- helpers ------------------------------------------------------------------
    def _gains_array(self, gains) -> wp.array:
        return wp.array(np.asarray(gains, dtype=np.float32), dtype=float, requires_grad=True)

    def finite_difference_check(self, gains, eps: float = 1e-3) -> dict:
        """Check ``tape.backward`` against central finite differences over the full horizon. One captured graph
        serves every rollout of the check: each writes its gains into the array the graph reads.
        """
        g0 = np.asarray(gains, dtype=np.float32)
        g = self._gains_array(g0)
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        gradient = CapturedGradient(lambda: self.forward(g, cost), g, cost)

        def at(values):
            g.assign(values)
            self.rollout.reset()
            return gradient()

        analytic = at(g0)[1].numpy().astype(np.float64)
        fd = np.zeros(NUM_GAINS, dtype=np.float64)
        for i in range(NUM_GAINS):
            step = np.zeros_like(g0)
            step[i] = eps
            fd[i] = (at(g0 + step)[0] - at(g0 - step)[0]) / (2.0 * eps)
        cos = float(analytic @ fd / (np.linalg.norm(analytic) * np.linalg.norm(fd) + 1e-30))
        floor = 0.05 * np.max(np.abs(fd))
        sig = np.abs(fd) >= floor
        rel = np.abs(analytic[sig] - fd[sig]) / (np.abs(fd[sig]) + 1e-12)
        return {
            "analytic": analytic.tolist(),
            "finite_diff": fd.tolist(),
            "cosine_similarity": cos,
            "max_rel_err_significant": float(rel.max()) if rel.size else 0.0,
            "eps": eps,
        }

    def trajectory(self, gains) -> np.ndarray:
        """Forward-only, full horizon: the ``(N+1, 3)`` base-body position trajectory, start prepended, for
        the flight-quality metrics.
        """
        self.controller.gains_wp = self._gains_array(gains)
        rec: list = []
        self.rollout.reset()
        rec.append(self.rollout.state_at(0).body_q.numpy()[self.rollout.base, :3].copy())
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        cost.zero_()
        for t in range(self.N):
            self._rollout_step(t, cost, record=rec)
        wp.synchronize()
        return np.array(rec)

    def optimize(self, gains0, *, iters: int = 40, lr: float = 0.02, lo=None, hi=None):
        """Adam over the PID gains via the exact full-horizon gradient, projected onto physical, non-negative
        bounds each step, tracking the lowest-loss iterate. The gradient replays from one captured graph on a
        CUDA device; returns ``(best_gains, loss_history)``.
        """
        g = self._gains_array(gains0)
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        gradient = CapturedGradient(lambda: self.forward(g, cost), g, cost)
        opt = wp.optim.Adam([g], lr=lr)
        lo = None if lo is None else np.asarray(lo, dtype=np.float32)
        hi = None if hi is None else np.asarray(hi, dtype=np.float32)
        best_loss = float("inf")
        best_gains = np.asarray(gains0, dtype=np.float32).copy()
        history: list[float] = []
        for _ in range(iters):
            self.rollout.reset()
            total, grad = gradient()
            history.append(total)
            if total < best_loss:
                best_loss = total
                best_gains = g.numpy().copy()
            opt.step([grad])
            if lo is not None or hi is not None:
                g.assign(np.clip(g.numpy(), lo, hi))
        return best_gains, history


def overshoot_and_settle(traj: np.ndarray, goal, dt: float, tol: float = 0.05):
    """Trajectory quality metrics: peak overshoot past the goal, final distance, settle time [s]."""
    goal = np.asarray(goal, dtype=np.float64)
    d = np.linalg.norm(traj - goal, axis=1)
    # overshoot: how far past the goal it travels along the start->goal direction
    final_dist = float(d[-1])
    peak_overshoot = float(max(0.0, (np.linalg.norm(traj - traj[0], axis=1)).max() - np.linalg.norm(goal - traj[0])))
    settled = np.where(d <= tol)[0]
    settle_time = float(settled[0] * dt) if settled.size else float("inf")
    return {"final_dist": final_dist, "peak_overshoot": peak_overshoot, "settle_time_s": settle_time}


def main():
    # The run's time, net of the kernels Warp compiles into a cold cache, is what CI's time gate reads.
    with RunClock() as clock:
        # Runs on CUDA when present, where each gradient replays from one captured graph, else on the CPU.
        if wp.is_cuda_available():
            wp.set_device("cuda:0")
        GOAL = (3.0, 2.0, 4.0)  # a far waypoint, ~4 m: the 6 s horizon reaches it by ~3 s and must hold for ~3 s
        MS = 0.12  # moment scale: the per-rotor allocation B's attitude authority
        # The moment-input BPTT through the articulated plant is exact over the whole horizon, checked against
        # finite differences. The closed loop is contractive, so BPTT needs no truncation or windowing; the CTBR
        # variant's inner rate loop would re-explode it. Crucially the horizon is long enough, 6 s, to include a
        # genuine hold phase after the drone reaches the far goal. The ramped time-weight then penalizes residual
        # velocity in the late steps, so the optimizer tunes a controller that SETTLES, comes to rest, not one
        # that merely reaches by the horizon's end and then overshoots. A short horizon that ends at arrival was
        # the regression: the tuned gains reached but didn't hold, so the deployed flight flew through the goal.
        STABLE_REF = np.array([0.3, 1.0, 0.6, 1.2, 8.0, 3.0, 1.0], dtype=np.float32)
        am_grad = AstroMaxWaypointRollout(n_steps=200, goal=GOAL, vel_weight=0.2, moment_scale=MS)
        fd2 = am_grad.finite_difference_check(STABLE_REF)
        nx.logger.info(f"astro-max full-horizon gradient: cosine={fd2['cosine_similarity']:.4f}")
        assert fd2["cosine_similarity"] > 0.99, "astro-max full-horizon gradient not finite-diff-correct"

        # Deliberately mistuned baseline: weak altitude P, g0, + low horizontal-position P, g2, ⇒ it crawls,
        # falls short of the waypoint and never settles: a clear "before" against the tuned controller.
        BASE = np.array([0.07, 0.7, 0.08, 0.35, 8.0, 2.5, 1.0], dtype=np.float32)
        LO = [0.01, 0.01, 0.01, 0.01, 1.0, 0.2, 0.1]
        HI = [1.5, 3.0, 1.0, 2.0, 20.0, 8.0, 4.0]
        am = AstroMaxWaypointRollout(n_steps=1500, goal=GOAL, vel_weight=0.2, moment_scale=MS)  # 6 s → reach + hold
        # 30 iterations keep the run under 5 min on CI's A10G. The tuned flight's hold error isn't monotonic in
        # them: on the RTX 5080 it holds within 0.066 m at 30, 0.527 m at 50 and 0.151 m at 80.
        gopt, hist = am.optimize(BASE, iters=30, lr=0.04, lo=LO, hi=HI)
        nx.logger.info(f"gain opt: loss {hist[0]:.0f} -> min {min(hist):.0f}  gains={np.round(gopt, 3)}")

        # The real gate: fly both gain sets free-running on the articulated plant the tuning ran on, the PID
        # example's run, with moment_scale=MS matching the tuning. Gate on the FLIGHT, not the fixed-horizon
        # rollout metric: that metric looked "settled" while the drone was still moving through the goal at the
        # horizon's end, the trap
        # this example fell into. The tuned controller must reach the far goal and hold it, with no flying through;
        # the mistuned baseline must not. --log/--view records each flight; the deploy IS the standard Loop run.
        def fly(gains, name: str):
            # The deploy flight IS the standard loop run, on the articulated plant the tuning ran on: the
            # example-owned PID assembly.
            from nexus_uav._src.build.launch import resolve_scenario
            from nexus_uav._src.config import LaunchConfig
            from nexus_uav._src.guidance import MissionGuidance
            from nexus_uav.examples.controllers.pid.assembly import build_pid_loop

            launch = LaunchConfig().set_vehicle("astro_max_base").set_scene("empty")
            launch.runtime.device = "cuda"  # prefer CUDA; falls back to CPU without one
            vehicle_usd_2, _resolved, cfg = resolve_scenario(launch)
            orch = build_pid_loop(
                cfg,
                vehicle_usd=vehicle_usd_2,
                gains=list(gains),
                moment_scale=MS,
                max_steps=2750,
                rerun=True,
            )
            guidance = MissionGuidance(reached_m=0.15, final_hold_s=3.0)
            with nx.Sim._from_loop(orch, guidance=guidance) as sim:
                guidance.set_mission([GOAL])
                sim.run()
                q = np.asarray(sim[f"vehicle/body/{sim.base_body}"].history()["position"], dtype=float)
                rotors = [path for path in sim if path.startswith("vehicle/joints/") and "rotor" in path]
                speeds = [abs(float(sim[path].history()[-1]["qd"][0])) for path in rotors]
                rrd = sim.artifacts().get("rrd")
            d = np.linalg.norm(q - np.array(GOAL, dtype=float), axis=1)
            reached = bool(d.min() < 0.2)  # got to the goal
            hold = float(d[int(d.argmin()) :].max()) if reached else float("inf")  # worst dist after first reaching it
            nx.logger.info(
                f"{name} flight: final {d[-1]:.3f} m, closest {d.min():.3f} m, hold-error {hold:.3f} m"
                + (f"  ({rrd})" if rrd else "")
            )
            return d[-1], reached, hold, sim, guidance, speeds

        base_final, base_reached, _, _, _, _ = fly(BASE, "baseline")
        _opt_final, opt_reached, opt_hold, opt_sim, opt_guidance, speeds = fly(gopt, "optimized")
        # Evaluation artifacts first, before any gate can raise, because a failed run must still leave its
        # trajectory for diagnosis: the tuned deploy flight + the tuning-quality metrics. The CI harness
        # gates the gradient cosine, the hold error, the run's wall time and the rotors' speed at the end of the
        # tuned flight, and monitors position Absolute Pose Error (APE) + the deploy Real-Time Factor (RTF).
        stats = {
            "gradient_cosine": round(fd2["cosine_similarity"], 4),
            "opt_hold_m": round(opt_hold, 4),
            "opt_final_m": round(float(_opt_final), 4),
            "base_final_m": round(float(base_final), 4),
            "rotor_speed_min_rad_per_sec": round(min(speeds), 1) if speeds else 0.0,
            **clock.stats(),
        }
        dump_run(opt_sim, "gain_tuning", stats=stats, waypoints=[GOAL], arrival_times=opt_guidance.arrival_times)
        # The win is HOLDING the waypoint: the tuned controller reaches and stays, hold-error small with no
        # flying through, where the mistuned baseline never gets there.
        assert opt_reached and opt_hold < 0.3, (
            f"tuned controller flew through / didn't hold (hold-error {opt_hold:.3f} m)"
        )
        assert not base_reached, f"the baseline was supposed to be mistuned (it reached; final {base_final:.3f} m)"
        nx.logger.info("OK: gradient correct; tuned PID reaches + HOLDS the waypoint (mistuned baseline does not)")


if __name__ == "__main__":
    main()
