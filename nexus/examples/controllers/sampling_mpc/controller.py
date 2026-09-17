"""``SamplingMPCController``: receding-horizon sampling + gradient diffsim Model Predictive
Control (MPC) on the Controller seam.

A first-class in-process Controller, alongside :class:`~nexus.examples.controllers.pid.PidController`,
:class:`~nexus._src.vehicle.controllers.px4.Px4MavlinkController`, and the trained-policy controller. Each
tick it samples ``num_rollouts`` noisy plans around the nominal, refines them all in parallel by
back-propagating an obstacle-aware cost through a batched differentiable rollout, a forward/backward
captured as a Compute Unified Device Architecture (CUDA) graph, keeps the lowest-cost one, and applies
its first control. Mirrors NVIDIA's ``example_diffsim_drone``: control-point trajectories interpolated
over the horizon, the optimisation replayed from a captured graph, the sampling done outside the graph.

The controller is task-parameterized: the caller passes the **batched rollout model**, ``num_rollouts``
differentiable drones + the cost-only obstacle pillars, the obstacle shape indices, the rotor geometry,
and the initial ``goal_w``; the operator advances the active target via ``accept_setpoint(PositionGoal)``,
uniform with policy/pid. The per-rotor rollout dynamics is the shared single-body motor
model :func:`~nexus.examples._lib.rigid_body_wrench_world`, so planner ≡ the real-sim
:class:`~nexus.examples._lib.rotors.RigidBodyRotors` actuator: both consume per-rotor commands and
run the forward-``B`` allocation. The cost-weight defaults below tune a gentle arrive-and-stop with
obstacle avoidance. The vehicle's Universal Scene Description (USD) asset uses the Forward Right
Down (FRD) convention: thrust points along −body-z.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.optim
from newton.geometry import sdf_capsule  # warp-callable Signed Distance Field (SDF) for the collision-cost kernel

from nexus._src.core.schema import PositionGoal
from nexus.examples._lib import build_rotor_mixer_from_layout, rigid_body_wrench_world

GRAVITY = 9.81

# Cost weights, following the NVIDIA example's balance. The task stays gentle, a short hop to a nearby
# target, reachable within the planning horizon, so the controller plans a smooth arrive-and-stop rather
# than perpetual max-acceleration: the obstacle detour is then the only real maneuver. A modest upright
# term and a nose-first heading term mop up the residual tilt and turn the nose to face the direction of
# travel. The yaw lesson: a first-order method can't null a yaw *rate*, so penalize the *angle*, here the
# angle between the nose and the velocity, so the drone flies nose-first through the slalom.
POS_WEIGHT = 400.0  # horizontal xy position tracking, strong enough to station-keep at the target
# against drift: the velocity damping alone settles velocity but lets position wander during the hold
ALTITUDE_WEIGHT = 6000.0  # vertical z held hard: rolling to strafe tilts the thrust off vertical and the drone
# sags, then overshoots recovering; a strong altitude term makes it plan the extra thrust to hold height
UPRIGHT_WEIGHT = 600.0  # > the position weight so the optimizer never trades uprightness for a faster reach,
# which keeps tilt gentle
# Nose-first heading, a SECONDARY term. The cost scales with speed², up to ~9 at cruise, so a big weight,
# 200 made it ~1800, co-equal with the ~1600 position cost, makes the planner sacrifice the sharp waypoint
# reversals to keep the nose aligned: it slows/lingers at a turn instead of powering to the next waypoint.
# 60 keeps the nose tracking the velocity, ~14° median, while staying well below the position term.
HEAD_WEIGHT = 60.0
VEL_WEIGHT = 22.0  # damps linear+angular velocity → a crisp arrive-and-stop; the heavier USD-correct
# inertia is more sluggish in attitude, so the old value of 15 under-damped it into tilt overshoot/oscillation
CONTROL_WEIGHT = 3.0
COLLISION_WEIGHT = 8.0e3  # the SDF barrier; large so proximity dominates the pull toward the target


@wp.kernel
def sample_gaussian(
    nominal: wp.array3d(dtype=float),  # (1, n_points, control_dim) current lowest-cost plan
    noise_scale: float,
    n_points: int,
    control_dim: int,
    control_limits: wp.array2d(dtype=float),
    seed: wp.array(dtype=int),
    rollouts: wp.array3d(dtype=float),  # (num_rollouts, n_points, control_dim) out
):
    w, pt, c = wp.tid()
    uid = (w * n_points + pt) * control_dim + c
    r = wp.rand_init(seed[0], uid)
    mean = nominal[0, pt, c]
    lo, hi = control_limits[c, 0], control_limits[c, 1]
    s = mean + noise_scale * wp.randn(r)
    for _ in range(10):  # rejection-resample to stay in bounds, NVIDIA's trick
        if s < lo or s > hi:
            s = mean + noise_scale * wp.randn(r)
        else:
            break
    rollouts[w, pt, c] = wp.clamp(s, lo, hi)


@wp.kernel
def increment_seed(seed: wp.array(dtype=int)):
    seed[0] += 1


@wp.kernel
def replicate_state(
    src_q: wp.array(dtype=wp.transform),
    src_qd: wp.array(dtype=wp.spatial_vector),
    dst_q: wp.array(dtype=wp.transform),
    dst_qd: wp.array(dtype=wp.spatial_vector),
):
    w = wp.tid()
    dst_q[w] = src_q[0]
    dst_qd[w] = src_qd[0]


@wp.kernel
def interpolate_control(
    rollouts: wp.array3d(dtype=float),  # (num_rollouts, n_points, control_dim)
    point_coord: float,  # continuous control-point coordinate at this rollout step
    control_dim: int,
    controls: wp.array2d(dtype=float),  # (num_rollouts, control_dim) out: the per-rotor command the motor model reads
):
    w, c = wp.tid()
    ti = int(point_coord)
    frac = point_coord - wp.floor(point_coord)
    left = rollouts[w, ti, c]
    right = rollouts[w, ti + 1, c]
    controls[w, c] = left * (1.0 - frac) + right * frac


@wp.kernel
def drone_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target: wp.array(dtype=wp.vec3),
    controls: wp.array2d(dtype=float),  # (num_rollouts, n_rotors) per-rotor command
    n_rotors: int,
    horizon: int,
    cost: wp.array(dtype=float),
):
    w = wp.tid()
    tf = body_q[w]
    pos = wp.transform_get_translation(tf)
    d = pos - target[0]
    pos_cost = d[0] * d[0] + d[1] * d[1]  # horizontal only; altitude gets its own, harder weight
    alt_cost = d[2] * d[2]
    up = wp.transform_vector(tf, wp.vec3(0.0, 0.0, -1.0))  # thrust/up axis in world: −body-z on the FRD-authored USD
    upright_cost = 1.0 - up[2]  # 0 level, grows with tilt → keeps it upright instead of flipping
    fwd = wp.transform_vector(tf, wp.vec3(1.0, 0.0, 0.0))  # body-forward axis in world
    vel_w = wp.spatial_top(body_qd[w])  # world linear velocity
    # Nose-first: penalize the horizontal velocity not explained by forward motion along the nose:
    # heading_cost = |v_hor|² − relu(fwd·v_hor)². This is zero only when the nose points along the velocity
    # in the forward direction. A sin²/cross² term is also zero flying tail-first, and with real yaw
    # authority the planner settles into 180°, measured 167°; the relu keeps backward motion fully penalized
    # so it flies nose-forward. Scales with speed² ⇒ vanishes at hover, heading free; no division ⇒ smooth
    # Backpropagation Through Time (BPTT).
    vx = vel_w[0]
    vy = vel_w[1]
    fdot = wp.max(fwd[0] * vx + fwd[1] * vy, 0.0)  # forward speed: velocity along the nose, backward clamped to 0
    heading_cost = (vx * vx + vy * vy) - fdot * fdot
    vel_cost = wp.length_sq(body_qd[w])  # damp linear + angular velocity → settle at the target
    control = wp.vec4(controls[w, 0], controls[w, 1], controls[w, 2], controls[w, 3])  # quad: 4 rotors
    control_cost = wp.dot(control, control)
    # Uniform per-step weighting, no terminal discount: attitude/heading/altitude carry their full
    # weight at every step, including the first, which is the control actually executed, so the drone
    # stays upright and on-heading throughout and settles level at the target, instead of drifting off once
    # the position error, the only term a terminal discount keeps alive, vanishes.
    inv_h = 1.0 / wp.float(horizon)
    wp.atomic_add(
        cost,
        w,
        (
            POS_WEIGHT * pos_cost
            + ALTITUDE_WEIGHT * alt_cost
            + UPRIGHT_WEIGHT * upright_cost
            + HEAD_WEIGHT * heading_cost
            + VEL_WEIGHT * vel_cost
            + CONTROL_WEIGHT * control_cost
        )
        * inv_h,
    )


@wp.kernel
def collision_cost(
    body_q: wp.array(dtype=wp.transform),
    obs_indices: wp.array(dtype=int),  # shape indices of the obstacle pillars
    shape_transform: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    margin: float,
    horizon: int,
    cost: wp.array(dtype=float),
):
    w, oid = wp.tid()  # one thread per rollout-obstacle pair: the drone avoids every pillar at once
    si = obs_indices[oid]
    px = wp.transform_get_translation(body_q[w])
    x_local = wp.transform_point(wp.transform_inverse(shape_transform[si]), px)
    scale = shape_scale[si]
    d = sdf_capsule(x_local, scale[0], scale[1], 2)  # 2 = Axis.Z, the capsule's long axis
    d = wp.max(d, 0.0)
    if d < margin:
        wp.atomic_add(cost, w, COLLISION_WEIGHT * (margin - d) / wp.float(horizon))


@wp.kernel
def enforce_limits(control_limits: wp.array2d(dtype=float), rollouts: wp.array3d(dtype=float)):
    w, pt, c = wp.tid()
    lo, hi = control_limits[c, 0], control_limits[c, 1]
    rollouts[w, pt, c] = wp.clamp(rollouts[w, pt, c], lo, hi)


@wp.kernel
def pick_best(rollouts: wp.array3d(dtype=float), best_id: int, nominal: wp.array3d(dtype=float)):
    pt, c = wp.tid()
    nominal[0, pt, c] = rollouts[best_id, pt, c]


class SamplingMPCController:
    """Receding-horizon sampling MPC. Each tick: sample ``num_rollouts`` noisy plans around the nominal,
    refine them all in parallel by back-propagating an obstacle-aware cost through a batched
    differentiable rollout, keep the lowest-cost one, apply its first control. Mirrors NVIDIA's
    ``example_diffsim_drone``: control-point trajectories interpolated over the horizon, a CUDA-graph-
    captured forward/backward replayed for the optimisation steps, sampling done outside the graph.
    """

    def __init__(
        self,
        *,
        batch_model,
        mass,
        rotor_offsets,
        turning_dirs,
        ct,
        rpm_max,
        goal_w=(0.0, 0.0, 1.0),  # initial target; the operator advances it via accept_setpoint(PositionGoal)
        dt,
        num_rollouts=16,
        control_points=5,  # trajectory knots; interpolated over the horizon: low-dim → effective sampling
        point_step=16,  # rollout steps between knots → horizon = control_points * point_step, 80 steps ≈ 0.8 s:
        # long enough to plan a gentle arrive-and-stop over the hop; a short horizon forces aggressive flips
        plan_dt=0.01,  # planning rollout dt, deliberately fine: an agile quad needs it, a coarse dt mispredicts turns
        optim_steps=12,
        replan_every=4,
        lr=0.03,
        noise_scale=0.10,
        collision_margin=0.6,
        reaction_k=0.05,
        motor_tau,  # required: the plant's first-order motor lag [s]; pass the vehicle USD's motor:tau
    ):
        import newton.solvers

        self.num_rollouts = int(num_rollouts)
        self.n_points = int(control_points) + 1  # +1 knot for the final interpolation segment
        self.point_step = int(point_step)
        self.horizon = int(control_points) * self.point_step
        self.control_dim = len(turning_dirs)
        self.n_rotors = len(turning_dirs)
        self.dt = float(dt)
        self.plan_dt = float(plan_dt)
        self.optim_steps = int(optim_steps)
        self.replan_every = int(replan_every)
        self.noise_scale = float(noise_scale)
        self.collision_margin = float(collision_margin)
        # The obstacles are simply the scene: its static capsule shapes, discovered from the model.
        # The scene USD authored them; nothing threads in anything scene-specific. The avoidance SDF
        # supports capsules; the cost ignores other static scene shapes.
        sb = batch_model.shape_body.numpy()
        st = batch_model.shape_type.numpy()
        obs_indices = [i for i in range(len(sb)) if sb[i] == -1 and st[i] == int(newton.GeoType.CAPSULE)]
        self.obs_indices = wp.array(np.asarray(obs_indices, dtype=np.int32), dtype=int)
        self.num_obstacles = len(obs_indices)
        # The ACTIVE target: a persistent (1,) vec3 buffer the rollout reads; accept_setpoint advances it
        # in place via .assign without invalidating the captured CUDA graph, because the rollout reads its
        # current contents. The operator owns mission sequencing, advance on arrival, uniform with policy/pid.
        self.target = wp.array(np.asarray([goal_w], dtype=np.float32), dtype=wp.vec3)

        self.model = batch_model
        self.solver = newton.solvers.SolverSemiImplicit(batch_model)
        # Airframe mixer from the rotor layout: the same forward B the real-sim RigidBodyRotors actuator
        # uses, so planner ≡ real. κ = reaction_k is the yaw allocation, passed as the layout's "cd" key.
        mixer = build_rotor_mixer_from_layout(rotor_offsets, turning_dirs, {"ct": ct, "cd": reaction_k, "rpm_max": rpm_max})  # fmt: skip
        self.B_wp = wp.array(mixer.B.astype(np.float32), dtype=float)  # (4, nr) forward allocation
        self._offsets_wp = wp.array(mixer.rotor_offsets.astype(np.float32), dtype=wp.vec3)  # (nr,) base-frame offsets
        self.thrust_sign = -1.0  # FRD-authored USD: thrust along −body-z, rotors spawned up
        self.hover = float(np.sqrt(mass * GRAVITY / self.n_rotors / ct) / rpm_max)
        # Per-rotor motor lag: the sampled per-rotor command sets a rotor-speed target; Ω relaxes toward it
        # with time constant tau, and thrust = kf·Ω² re-mixes through forward B. Stable filter ⇒ the BPTT
        # stays bounded. alpha uses the plan step, plan_dt. Ω starts each plan from the hover speed.
        self.kf = mixer.kf
        self.omega_max_motor = mixer.omega_max_motor
        self.motor_alpha = float(1.0 - np.exp(-self.plan_dt / float(motor_tau)))
        self.omega_hover = float(self.hover * self.omega_max_motor)

        lo_hi = np.array([[0.05, 1.0]] * self.control_dim, dtype=np.float32)
        self.control_limits = wp.array(lo_hi, dtype=float)

        # Plans: the single nominal, and the sampled rollouts, which get optimised.
        self.nominal = wp.zeros((1, self.n_points, self.control_dim), dtype=float)
        self.nominal.assign(np.full((1, self.n_points, self.control_dim), self.hover, dtype=np.float32))
        self.rollouts = wp.zeros((self.num_rollouts, self.n_points, self.control_dim), dtype=float, requires_grad=True)
        self.seed = wp.zeros(1, dtype=int)

        # Per-rollout-step states + control buffers for the taped rollout.
        self.S = [batch_model.state(requires_grad=True) for _ in range(self.horizon + 1)]
        self.step_controls = [
            wp.zeros((self.num_rollouts, self.control_dim), dtype=float, requires_grad=True)
            for _ in range(self.horizon)
        ]
        # Per-step per-rotor motor-speed state Ω[h] -> Ω[h+1]; distinct buffers ⇒ clean BPTT through the lag.
        self.omega = [
            wp.zeros((self.num_rollouts, self.n_rotors), dtype=float, requires_grad=True)
            for _ in range(self.horizon + 1)
        ]
        self.costs = wp.zeros(self.num_rollouts, dtype=float, requires_grad=True)
        self.lr = float(lr)
        self.opt = warp.optim.Adam([self.rollouts.flatten()], lr=self.lr)  # per-param normalised: stable over the
        # deep, long-horizon backprop where Stochastic Gradient Descent (SGD) explodes, and follows the weak
        # heading/attitude gradients
        self.tape = None
        self._graph = None

        self._k = 0
        self._logger = None  # the orchestrator hands over the Logger, None when off; gates the horizon viz

    def connect(self) -> None:
        pass

    def set_logger(self, logger) -> None:
        """The orchestrator hands over the Logger, ``None`` when off: the gate for the horizon viz."""
        self._logger = logger

    def close(self) -> None:  # lifecycle teardown: nothing to release; the horizon logs from _plan()
        pass

    def _seed_states(self, state) -> None:
        wp.launch(
            replicate_state,
            dim=self.num_rollouts,
            inputs=(state.body_q, state.body_qd),
            outputs=(self.S[0].body_q, self.S[0].body_qd),
        )

    def _rollout(self) -> None:
        self.costs.zero_()
        self.omega[0].fill_(self.omega_hover)  # each plan starts from the hover rotor speed, a capturable reset
        for h in range(self.horizon):
            self.S[h].clear_forces()
            wp.launch(
                interpolate_control,
                dim=(self.num_rollouts, self.control_dim),
                inputs=(self.rollouts, float(h) / float(self.point_step), self.control_dim),
                outputs=(self.step_controls[h],),
            )
            wp.launch(
                rigid_body_wrench_world,
                dim=self.num_rollouts,  # one thread per rollout, looping the rotors; differentiable for BPTT
                inputs=(
                    self.step_controls[h],  # (num_rollouts, n_rotors) per-rotor command
                    self.S[h].body_q,
                    self.S[h].body_qd,  # base twist, for the airflow inflow; airflow off here
                    0,  # base = 0: rollout w drives body w, bodies 0..N-1
                    self._offsets_wp,  # (nr,) base-frame rotor offsets
                    self.B_wp,
                    self.kf,
                    self.omega_max_motor,
                    self.motor_alpha,
                    1.0,  # substeps = 1: the planner integrates the wrench once per plan step
                    self.thrust_sign,
                    0.0,  # aero_h: airflow off; single-body proxy, no inflow model
                    0.0,  # aero_hforce
                    self.n_rotors,
                    self.omega[h],  # Ω[h] in
                    self.omega[h + 1],  # Ω[h+1] out; distinct ⇒ clean BPTT
                ),
                outputs=(self.S[h].body_f,),
            )
            self.solver.step(self.S[h], self.S[h + 1], None, None, self.plan_dt)
            wp.launch(
                drone_cost,
                dim=self.num_rollouts,
                inputs=(
                    self.S[h + 1].body_q,
                    self.S[h + 1].body_qd,
                    self.target,
                    self.step_controls[h],
                    self.n_rotors,
                    self.horizon,
                ),
                outputs=(self.costs,),
            )
            wp.launch(
                collision_cost,
                dim=(self.num_rollouts, self.num_obstacles),
                inputs=(
                    self.S[h + 1].body_q,
                    self.obs_indices,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    self.collision_margin,
                    self.horizon,
                ),
                outputs=(self.costs,),
            )

    def _forward_backward(self) -> None:
        self.tape = wp.Tape()
        with self.tape:
            self._rollout()
        self.costs.grad.fill_(1.0)
        self.tape.backward()

    def _plan(self, state) -> None:
        self._seed_states(state)
        # Sample noisy plans around the nominal, outside the captured graph; keep the last slot = nominal.
        wp.launch(
            sample_gaussian,
            dim=(self.num_rollouts - 1, self.n_points, self.control_dim),
            inputs=(self.nominal, self.noise_scale, self.n_points, self.control_dim, self.control_limits, self.seed),
            outputs=(self.rollouts,),
        )
        wp.launch(increment_seed, dim=1, inputs=(), outputs=(self.seed,))
        self.rollouts[-1].assign(self.nominal[0])

        if self._graph is None and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._forward_backward()
            self._graph = capture.graph

        for _ in range(self.optim_steps):
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                self._forward_backward()
            self.opt.step([self.rollouts.grad.flatten()])
            wp.launch(enforce_limits, dim=self.rollouts.shape, inputs=(self.control_limits,), outputs=(self.rollouts,))
            self.tape.zero()

        best = int(np.argmin(self.costs.numpy()))
        wp.launch(
            pick_best, dim=(self.n_points, self.control_dim), inputs=(self.rollouts, best), outputs=(self.nominal,)
        )
        # Component-owned horizon viz, event-driven, per replan: emit the lowest-cost rollout's predicted
        # path to controller/mpc_horizon. Gated on self._logger so the code skips the horizon+1 GPU→CPU
        # reads below entirely when not recording; they're pure viz cost. The orchestrator set the timeline
        # at tick start, so scrubbing shows the plan logged most recently before the cursor, the active one.
        if self._logger is not None:
            pred = np.array([self.S[h].body_q.numpy()[best, :3] for h in range(self.horizon + 1)], dtype=np.float32)
            self._logger.log_strip("controller/mpc_horizon", pred, color=(255, 140, 0))

    def exchange(self, meas, t, timeout=None):
        from nexus._src.core import Controls

        state = meas.state
        if not hasattr(self, "_step"):
            self._step = 0
        if self._k == 0:
            self._plan(state)
        # Apply the first control of the lowest-cost plan: interpolate the nominal at t=0, the first knot.
        u0 = self.nominal.numpy()[0, 0].astype(np.float32)
        self._k = (self._k + 1) % self.replan_every
        self._step += 1
        return Controls(command=u0)

    # -- control surface: the thin setpoint seam ---------
    def accept_setpoint(self, sp) -> None:
        """Write the active target **in place** from a :class:`PositionGoal` via ``.assign``: the captured
        rollout reads the buffer's current contents, so no graph re-capture. Uniform with policy/pid: the
        operator sequences a mission, advance on arrival, and the planner always optimizes toward the
        current target. Raises on a non-``PositionGoal`` variant.
        """
        if not isinstance(sp, PositionGoal):
            raise TypeError(f"SamplingMPCController accepts a PositionGoal setpoint, got {type(sp).__name__}")
        self.target.assign(np.asarray([sp.pos], dtype=np.float32))
