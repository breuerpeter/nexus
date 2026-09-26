"""Controller gain tuning by differentiable simulation: back-propagate through a closed-loop
hover→waypoint flight to **auto-tune the Proportional Integral Derivative (PID) gains**.

Self-contained example, as NVIDIA's ``example_diffsim_*`` are: it defines its own differentiable rollout
inline and imports only reusable framework bits, namely the shared PID law, the obs/actuator kernels, and
the solvers. :class:`AstroMaxWaypointRollout` is a single free body with astro-max's mass/inertia + a
quad-X rotor layout, actuated by the shared **moment-input RigidBodyRotors** kernel, namely control
allocation ``B`` + per-rotor motor lag + saturation with **no Collective Thrust and Body Rate (CTBR) rate
loop**, under ``SolverSemiImplicit``. Gradient descent on the PID gains makes it fly a waypoint and settle,
with an **exact full-horizon gradient**, cosine ≈ 1.0 compared to finite-diff at a contractive operating
point. The rate-loop-free front-end is what keeps the long-horizon Backpropagation Through Time (BPTT)
bounded; the CTBR variant's inner rate loop pushes the closed-loop spectral radius >1 and explodes it.

The companion ``mass_recovery.py`` is the other differentiable-design gate, physical-parameter system-ID.

Run:

    uv run --extra examples -m nexus.examples gain_tuning            # asserts gradient-correct + tuned
    uv run --extra examples -m nexus.examples gain_tuning --log      # also write the optimized-flight .rrd

Runs on Compute Unified Device Architecture (CUDA) when present, else the CPU backend; the single-body
RigidBodyRotors rollout is light either way.
"""

from __future__ import annotations

import newton
import newton.solvers
import numpy as np
import warp as wp

import nexus as na
from nexus.examples._lib import (
    build_rotor_mixer_from_layout,  # airframe B / B⁻¹ + thrust map from the rotor layout
    dump_run,
    moment_to_cmd_batched,  # the moment mixer: collective + moments → B⁻¹ → per-rotor command, no rate loop
    motor_alpha,
    pack_vec4,  # pack the length-4 moment action into the wp.vec4 the mixer reads
    rigid_body_wrench_world,  # the single-body motor model: per-rotor cmd → lag → forward-B → base wrench
)
from nexus.examples._lib.observation import WarpObservationSensor
from nexus.examples._lib.single_body import collapse_to_single_body  # the one single-body seam
from nexus.examples.controllers.pid import NUM_GAINS, PidController

GRAVITY = 9.81


@wp.kernel
def waypoint_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    goal: wp.vec3,
    vel_weight: float,
    time_weight: float,
    cost: wp.array(dtype=float),
):
    """Per-step cost: distance-to-waypoint + velocity, which penalizes overshoot, ramped in time so
    later steps weigh more, which penalizes slow convergence, the time-to-reach.
    """
    p = wp.transform_get_translation(body_q[0])
    v = wp.spatial_top(body_qd[0])
    d = p - goal
    wp.atomic_add(cost, 0, time_weight * (wp.dot(d, d) + vel_weight * wp.dot(v, v)))


class AstroMaxWaypointRollout:
    """Differentiable hover->waypoint rollout on the **single-body astro-max + the shared RigidBodyRotors model**.

    Where the old rollout flew the multi-body astro-max Universal Scene Description (USD) with a hand-rolled
    per-rotor thrust kernel under ``SolverFeatherstone``, whose articulation forced *truncated* BPTT, this
    flies the **single free body** astro-max proxy from :func:`build_single_body_drone`, real mass/inertia +
    a quad-X rotor layout, with the shared, fully differentiable seam: the **moment mixer**
    :func:`moment_to_cmd_batched`, collective + moments -> ``B^-1`` -> per-rotor command with NO rate loop,
    feeding the single-body motor model :func:`rigid_body_wrench_world`, per-rotor command -> first-order
    motor lag/saturation -> forward ``B`` -> base wrench, under the gradient-capable ``SolverSemiImplicit``.
    So **BPTT is exact over the full horizon**, cosine ~ 1.0 compared to finite-diff, with no windowing.

    Why the *moment* mixer, not the CTBR one: the CTBR inner rate loop, the ``omega_des - omega_b``
    feedback, pushes the closed-loop BPTT spectral radius >1, so its full-horizon gradient explodes, with a
    measured cosine of ~0.15 at 400 steps. The moment-input front-end skips the rate loop, feeding the
    controller's collective + moments straight into the stable, first-order allocation + motor-lag
    back-end, which keeps a clean full-horizon gradient while retaining the per-rotor allocation ``B``,
    motor lag ``tau``, and rotor-speed saturation fidelity. The moment interface also *matches* the original
    control, since the old ``pid_to_rotor_rpms`` took ``[collective, roll, pitch, yaw]`` moments, so the
    shared ``_pid`` law + its gains transfer; the single body spawns upright, so the z-up sign convention
    holds, with no Forward Right Down (FRD) un-flip, and thrust is +body-z.
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
        from nexus._src.build.launch import resolve_to_vehicle_builder
        from nexus._src.config import LaunchConfig

        vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base"))
        # Single rigid body collapsed from the astro-max USD, the same seam the sampling
        # Model Predictive Control (MPC) example uses: correct lumped mass/inertia + the real rotor layout,
        # FRD with thrust along −body-z, spawned rotors-up.
        sb = collapse_to_single_body(vb, requires_grad=True)
        self.model, self.total_mass, rotor_offsets, turning_dirs = sb.model, sb.mass, sb.offsets, sb.dirs
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.N = int(n_steps)
        self.dt = float(dt)
        self.start = np.asarray(start, dtype=np.float32)
        self.goal_np = np.asarray(goal, dtype=np.float32)
        self.goal = wp.vec3(*[float(x) for x in goal])
        self.vel_weight = float(vel_weight)
        self.ramp_time_weight = bool(ramp_time_weight)
        self.weight = float(self.total_mass) * GRAVITY
        self.thrust_sign = -1.0  # FRD-authored USD: thrust along −body-z, rotors spawned up
        self.moment_scale = float(moment_scale)

        # Airframe mixer: control allocation B, per-rotor thrust -> [T, tx, ty, tz], + its inverse B^-1, from
        # the quad-X rotor geometry, the same builder the deploy paths use. The moment mixer, B^-1 with no
        # rate loop, and the forward B, the RigidBodyRotors motor model, carry the real arm + kappa yaw
        # authority.
        m = vb.actuator_params()  # aero/thrust map from the vehicle USD, the freefly:actuator:* attributes
        mixer = build_rotor_mixer_from_layout(
            rotor_offsets, turning_dirs, {"ct": m["ct"], "cd": m["cd"], "rpm_max": m["rpm_max"]}
        )
        self.nr = mixer.nr
        # The action scale = the plant's true T/W from the USD thrust map, no declared constant:
        # the same derivation the deploy build uses, so tuned gains deploy on the same scale.
        self.t2w = float(thrust_to_weight) if thrust_to_weight is not None else mixer.thrust_to_weight(self.total_mass)
        self.B_wp = wp.array(mixer.B.astype(np.float32), dtype=float)  # forward B, the motor model
        self.B_inv_wp = wp.array(mixer.B_inv.astype(np.float32), dtype=float)  # mixer B^-1, the allocation
        self._offsets_wp = wp.array(mixer.rotor_offsets.astype(np.float32), dtype=wp.vec3)  # (nr,) base-frame offsets

        # Motor map: thrust = kf*O^2; first-order lag tau; saturation at omega_max_motor; m is the USD map read earlier.
        self.kf = mixer.kf
        self.omega_max_motor = mixer.omega_max_motor
        self.alpha = motor_alpha(m["tau"], dt)

        # Reused orchestrator components, the same as the lumped rollout, + full tape history.
        self.obs_sensor = WarpObservationSensor(goal_w=goal)
        self.controller = PidController(goal_w=goal, thrust_to_weight=self.t2w)
        self.states = [self.model.state(requires_grad=True) for _ in range(self.N + 1)]
        self.obs = [wp.zeros(12, dtype=float, requires_grad=True) for _ in range(self.N)]
        self.acts = [wp.zeros(4, dtype=float, requires_grad=True) for _ in range(self.N)]  # [collective, m_x, m_y, m_z]
        self.act4 = [wp.zeros(1, dtype=wp.vec4, requires_grad=True) for _ in range(self.N)]  # packed moment action
        self.cmd = [wp.zeros((1, self.nr), dtype=float, requires_grad=True) for _ in range(self.N)]  # per-rotor command
        # per-step motor-speed states O[t] -> O[t+1]; distinct buffers => clean full-horizon BPTT recurrence
        self.omega = [wp.zeros((1, self.nr), dtype=float, requires_grad=True) for _ in range(self.N + 1)]
        self._init_start_state()

    def _init_start_state(self) -> None:
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        bq = self.states[0].body_q.numpy()
        bq[0, 0:3] = self.start
        bq[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # 180° about X, xyzw: FRD body spawned rotors-up, thrust −body-z → world +z
        self.states[0].body_q.assign(bq)
        bqd = self.states[0].body_qd.numpy()
        bqd[0, :] = 0.0  # start at hover, zero velocity
        self.states[0].body_qd.assign(bqd)

    def _rollout_step(self, t: int, cost: wp.array | None, record: list | None = None) -> None:
        """One control+physics tick: obs -> shared PID law, collective + moments -> moment mixer, B⁻¹ →
        per-rotor command with NO rate loop -> single-body motor model -> SemiImplicit step -> optional cost.
        The motor-speed state chains omega[t] -> omega[t+1]; distinct buffers ⇒ clean full-horizon BPTT.
        """
        self.states[t].clear_forces()
        self.obs_sensor.sample_wp(self.states[t], self.obs[t])
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
        # Single-body motor model: per-rotor command -> motor lag -> kf·Ω² -> forward B -> base-body wrench.
        wp.launch(
            rigid_body_wrench_world,
            dim=1,
            inputs=(
                self.cmd[t],
                self.states[t].body_q,
                self.states[t].body_qd,  # base twist: airflow inflow; airflow off here
                0,
                self._offsets_wp,
                self.B_wp,
                self.kf,
                self.omega_max_motor,
                self.alpha,
                1.0,
                self.thrust_sign,
                0.0,  # aero_h: airflow off, single-body proxy
                0.0,  # aero_hforce
                self.nr,
                self.omega[t],
                self.omega[t + 1],
            ),
            outputs=(self.states[t].body_f,),
        )
        self.solver.step(self.states[t], self.states[t + 1], None, None, self.dt)
        if cost is not None:
            tw = (float(t + 1) / float(self.N)) if self.ramp_time_weight else 1.0
            wp.launch(
                waypoint_cost,
                dim=1,
                inputs=(self.states[t + 1].body_q, self.states[t + 1].body_qd, self.goal, self.vel_weight, tw),
                outputs=(cost,),
            )
        if record is not None:
            record.append(self.states[t + 1].body_q.numpy()[0, :3].copy())

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

    def loss(self, gains) -> float:
        g = self._gains_array(gains)
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        self._init_start_state()
        self.forward(g, cost)
        wp.synchronize()
        return float(cost.numpy()[0])

    def gradient(self, gains):
        """Return ``(loss, grad)`` via one taped full-horizon forward+backward."""
        g = self._gains_array(gains)
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        self._init_start_state()
        tape = wp.Tape()
        with tape:
            self.forward(g, cost)
        tape.backward(cost)
        wp.synchronize()
        loss = float(cost.numpy()[0])
        grad = g.grad.numpy().copy()
        tape.zero()
        return loss, grad

    def finite_difference_check(self, gains, eps: float = 1e-3) -> dict:
        """Check ``tape.backward`` against central finite differences over the full horizon: the exact
        gradient the moment-input single-body RigidBodyRotors model affords, the win over truncated BPTT.
        """
        g0 = np.asarray(gains, dtype=np.float32)
        _, analytic = self.gradient(g0)
        fd = np.zeros(NUM_GAINS, dtype=np.float64)
        for i in range(NUM_GAINS):
            gp = g0.copy()
            gp[i] += eps
            gm = g0.copy()
            gm[i] -= eps
            fd[i] = (self.loss(gp) - self.loss(gm)) / (2.0 * eps)
        analytic = analytic.astype(np.float64)
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
        the flight-quality metrics. The flown ``.rrd`` comes from a standard ``na.Sim`` builtin run on the
        same collapsed single-body plant with ``solver="semi_implicit"``, so the deploy plant ≡ this tuning
        plant and the optimized gains actually settle; see :meth:`main`.
        """
        self.controller.gains_wp = self._gains_array(gains)
        rec: list = []
        self._init_start_state()
        rec.append(self.states[0].body_q.numpy()[0, :3].copy())
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        cost.zero_()
        for t in range(self.N):
            self._rollout_step(t, cost, record=rec)
        wp.synchronize()
        return np.array(rec)

    def optimize(self, gains0, *, iters: int = 40, lr: float = 0.02, lo=None, hi=None):
        """Adam over the PID gains via the exact full-horizon gradient, projected onto physical, non-negative
        bounds each step, tracking the lowest-loss iterate. Eager, with a fresh tape per iter; returns
        ``(best_gains, loss_history)``.
        """
        g = self._gains_array(gains0)
        self.controller.gains_wp = g
        cost = wp.zeros(1, dtype=float, requires_grad=True)
        opt = wp.optim.Adam([g], lr=lr)
        lo = None if lo is None else np.asarray(lo, dtype=np.float32)
        hi = None if hi is None else np.asarray(hi, dtype=np.float32)
        best_loss = float("inf")
        best_gains = np.asarray(gains0, dtype=np.float32).copy()
        history: list[float] = []
        for _ in range(iters):
            self._init_start_state()
            cost.zero_()
            tape = wp.Tape()
            with tape:
                self.forward(g, cost)
            tape.backward(cost)
            wp.synchronize()
            total = float(cost.numpy()[0])
            history.append(total)
            if total < best_loss:
                best_loss = total
                best_gains = g.numpy().copy()
            opt.step([g.grad])
            tape.zero()
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
    # Runs on CUDA when present, else CPU; single body + SemiImplicit is light enough either way.
    if wp.is_cuda_available():
        wp.set_device("cuda:0")
    GOAL = (3.0, 2.0, 4.0)  # a far waypoint, ~4 m: the 6 s horizon reaches it by ~3 s and must hold for ~3 s
    MS = 0.12  # moment scale: the per-rotor allocation B's attitude authority
    # The moment-input single-body BPTT is exact over the whole horizon: cosine 1.0 to ≥1500 steps, verified
    # against finite-diff. The closed loop is contractive, so BPTT needs no truncation or windowing; the CTBR
    # variant's inner rate loop would re-explode it. Crucially the horizon is long enough, 6 s, to include a
    # genuine hold phase after the drone reaches the far goal. The ramped time-weight then penalizes residual
    # velocity in the late steps, so the optimizer tunes a controller that SETTLES, comes to rest, not one
    # that merely reaches by the horizon's end and then overshoots. A short horizon that ends at arrival was
    # the regression: the tuned gains reached but didn't hold, so the deployed flight flew through the goal.
    STABLE_REF = np.array([0.3, 1.0, 0.6, 1.2, 8.0, 3.0, 1.0], dtype=np.float32)
    am_grad = AstroMaxWaypointRollout(n_steps=200, goal=GOAL, vel_weight=0.2, moment_scale=MS)
    fd2 = am_grad.finite_difference_check(STABLE_REF)
    na.logger.info(f"astro-max full-horizon gradient: cosine={fd2['cosine_similarity']:.4f}")
    assert fd2["cosine_similarity"] > 0.99, "astro-max full-horizon gradient not finite-diff-correct"

    # Deliberately mistuned baseline: weak altitude P, g0, + low horizontal-position P, g2, ⇒ it crawls,
    # falls short of the waypoint and never settles: a clear "before" against the tuned controller.
    BASE = np.array([0.07, 0.7, 0.08, 0.35, 8.0, 2.5, 1.0], dtype=np.float32)
    LO = [0.01, 0.01, 0.01, 0.01, 1.0, 0.2, 0.1]
    HI = [1.5, 3.0, 1.0, 2.0, 20.0, 8.0, 4.0]
    am = AstroMaxWaypointRollout(n_steps=1500, goal=GOAL, vel_weight=0.2, moment_scale=MS)  # 6 s → reach + hold
    gopt, hist = am.optimize(BASE, iters=80, lr=0.04, lo=LO, hi=HI)
    na.logger.info(f"gain opt: loss {hist[0]:.0f} -> min {min(hist):.0f}  gains={np.round(gopt, 3)}")

    # The real gate: fly both gain sets free-running on the collapsed single-body plant the tuning ran on:
    # the na.Sim builtin with solver="semi_implicit" → build_pid_orchestrator's collapsed diffsim regime, and
    # moment_scale=MS matching the tuning. Gate on the FLIGHT, not the fixed-horizon rollout metric: that
    # metric looked "settled" while the drone was still moving through the goal at the horizon's end, the trap
    # this example fell into. The tuned controller must reach the far goal and hold it, with no flying through;
    # the mistuned baseline must not. --log/--view records each flight; the deploy IS the standard Orchestrator run.
    def fly(gains, name: str):
        # The deploy flight IS the standard orchestrator run, on the same collapsed single-body plant
        # the tuning ran on: the example-owned PID assembly with solver=semi_implicit.
        from nexus._src.build.launch import resolve_scenario
        from nexus._src.config import LaunchConfig
        from nexus._src.rendering import rtx_renderer
        from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

        launch = LaunchConfig().set_vehicle("astro_max_base")
        launch.runtime.device = "cuda"  # prefer CUDA; falls back to CPU without one
        launch.runtime.solver = "semi_implicit"  # the collapsed diffsim plant: deploy ≡ tuning plant
        vb2, _resolved, cfg = resolve_scenario(launch)
        orch = build_pid_orchestrator(
            cfg,
            vehicle_builder=vb2,
            gains=list(gains),
            moment_scale=MS,
            max_steps=2750,
            rerun=True,
            renderer_factory=rtx_renderer(vb2, cfg),  # the Kit peer, when the vehicle authors RTX sensors
        )
        with na.Sim.from_orchestrator(orch, reached_m=0.15, final_hold_s=3.0) as sim:
            sim.operator.set_mission([GOAL])
            sim.run()
            q = np.array([s.position for s in sim.physics[sim.base_body].history()])
            rrd = sim.artifacts().get("rrd")
        d = np.linalg.norm(q - np.array(GOAL, dtype=float), axis=1)
        reached = bool(d.min() < 0.2)  # got to the goal
        hold = float(d[int(d.argmin()) :].max()) if reached else float("inf")  # worst dist after first reaching it
        na.logger.info(
            f"{name} flight: final {d[-1]:.3f} m, closest {d.min():.3f} m, hold-error {hold:.3f} m"
            + (f"  ({rrd})" if rrd else "")
        )
        return d[-1], reached, hold, sim

    base_final, base_reached, _, _ = fly(BASE, "baseline")
    _opt_final, opt_reached, opt_hold, opt_sim = fly(gopt, "optimized")
    # Evaluation artifacts first, before any gate can raise, because a failed run must still leave its
    # trajectory for diagnosis: the tuned deploy flight + the tuning-quality metrics. The CI harness
    # gates the gradient cosine and hold error, and monitors position Absolute Pose Error (APE) + the
    # deploy Real-Time Factor (RTF).
    stats = {
        "gradient_cosine": round(fd2["cosine_similarity"], 4),
        "opt_hold_m": round(opt_hold, 4),
        "opt_final_m": round(float(_opt_final), 4),
        "base_final_m": round(float(base_final), 4),
    }
    dump_run(opt_sim, "gain_tuning", stats=stats, waypoints=[GOAL], arrival_times=opt_sim.operator.arrival_times)
    # The win is HOLDING the waypoint: the tuned controller reaches and stays, hold-error small with no
    # flying through, where the mistuned baseline never gets there.
    assert opt_reached and opt_hold < 0.3, f"tuned controller flew through / didn't hold (hold-error {opt_hold:.3f} m)"
    assert not base_reached, f"the baseline was supposed to be mistuned (it reached; final {base_final:.3f} m)"
    na.logger.info("OK: gradient correct; tuned PID reaches + HOLDS the waypoint (mistuned baseline does not)")


if __name__ == "__main__":
    main()
