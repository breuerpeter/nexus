---
description: "A real-time Nonlinear Model Predictive Control (NMPC) controller on acados, with the High-Performance Interior Point Method (HPIPM) solver and Sequential Quadratic Programming (SQP) Real-Time Iteration (RTI), flying a curving seven-waypoint course nose-first along a min-snap reference."
---

# Nonlinear model predictive control with acados

`nexus/examples/controllers/acados_nmpc/waypoint_tracking.py` flies the astro-max through a
**curving seven-waypoint course** with a **real-time nonlinear Model Predictive Control (MPC)**
controller.

- **Analytic NMPC, not differentiable simulation:** **acados** solves the optimal-control problem
  directly, with HPIPM and SQP real-time iteration. The model has 13 states, position, quaternion, velocity, and body
  rate, with the four single-rotor thrusts as inputs and an `N=20` horizon, re-solved every
  control tick.
- **A min-snap reference, tracked as the full flat state:** the NMPC is a pure *tracking*
  controller. The planner lives with the **operator**, in `sim.operator.set_mission(WAYPOINTS)`.
  It fits a **min-snap polynomial** through the waypoints, in `examples/_lib/min_snap.py`.
  Differential flatness then lifts it
  to a full state reference of attitude and body rate plus thrust feed-forward. The
  polynomial order adapts to the waypoint count, as `n_waypoints + 8`, so the min-snap cost always
  keeps 4 free coefficients beyond the waypoint and rest-endpoint constraints. The ruckig
  `FlatnessReference` remains as the jerk-limited fallback planner.
- **Flown nose-first:** min-snap smoothness is what makes nose-first flight track cleanly. The
  velocity direction turns gently, so the velocity-aligned yaw reference and its ω_z feed-forward
  stay bounded by construction. The course exercises it: a climbing left-hand arc that hooks back
  past the start, sweeping ~250° of heading over the ~13 s flight. The yaw rate stays at most 1.1
  rad per second. It curves throughout, with no straight segments.
- **Gated on tracking, not just arrival:** besides reaching the goal, the example asserts the
  evo-style position Absolute Pose Error (APE) against the planned reference. The gate is Root
  Mean Square Error (RMSE) < 0.05 m and max < 0.1 m. It also asserts the nose-versus-velocity
  heading < 25° and the tilt < 45°. Those are the metrics a flight-quality regression can't slip
  past.
- The smooth-tracking counterpart to the [sampling MPC](mpc-sampling.md). It's unbeatable where the
  cost is smooth and the model analytic. It can't take the Signed Distance Field (SDF) obstacle
  cost the sampling MPC handles, though.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/acados_nmpc.rrd" width="100%" height="600" frameborder="0"></iframe>

*The drone flies the blue planned reference nose-first. The orange strip is the predicted `N=20`
NMPC horizon, updating as it re-solves. The waypoint markers recolor with mission progress:
reached → green, active → gold, future → dim. A blank viewer means the recording isn't uploaded
yet: run `scripts/ci/evaluate_examples.py --upload`.*

## How it works

The controller, `nexus/examples/controllers/acados_nmpc/controller.py`, never plans: the
operator hands it the whole-path reference once through `accept_setpoint(ReferenceTrajectory)`,
and every control tick it solves a short optimal-control problem to stay on it.

### The internal model

A 13-state rigid-body quadrotor, written symbolically in CasADi. The state
$x = (p,\, q,\, v,\, \omega) \in \mathbb{R}^{13}$ holds position, attitude quaternion, world
velocity, and body rate, with the four per-rotor thrusts $u = (T_1, \dots, T_4)$ [N] as inputs:

$$
\dot p = v, \qquad
\dot q = \tfrac{1}{2}\, q \otimes \begin{pmatrix} 0 \\ \omega \end{pmatrix}, \qquad
\dot v = \begin{pmatrix} 0 \\ 0 \\ -g \end{pmatrix}
       + \frac{1}{m}\, R(q) \begin{pmatrix} 0 \\ 0 \\ \sum_i T_i \end{pmatrix}, \qquad
\dot \omega = J^{-1}\!\left( \tau - \omega \times J \omega \right)
$$

with the body torque assembled per rotor, moment arm cross thrust plus the spin-reaction yaw:

$$
\tau = \sum_{i=1}^{4} \, r_i \times \begin{pmatrix} 0 \\ 0 \\ T_i \end{pmatrix}
     + \begin{pmatrix} 0 \\ 0 \\ k_r\, T_i\, d_i \end{pmatrix}
$$

The numbers are deliberately *not* textbook constants. `assembly.py` reads the mass $m$, full
inertia matrix $J$, rotor moment arms $r_i$, and spin directions $d_i \in \{\pm 1\}$ off the real
articulated Newton model at assembly time. Those are the same constants the actuator kernel flies,
so the internal model mirrors the plant.

### The optimal-control problem

Nonlinear least-squares tracking over $N = 20$ shooting nodes spanning $t_f = 1$ s, at 50 ms node
spacing:

$$
\min_{x_{0:N},\, u_{0:N-1}}\;
\sum_{k=0}^{N-1} \left\lVert y(x_k, u_k) - y^{\mathrm{ref}}_k \right\rVert_W^2
\;+\; \left\lVert y_e(x_N) - y^{\mathrm{ref}}_N \right\rVert_{W_e}^2
$$

$$
\text{s.t.}\quad x_0 = \hat x, \qquad
x_{k+1} = f_{\mathrm{RK4}}(x_k, u_k), \qquad
0 \le u_k \le T_{\max}
$$

where $\hat x$ is the measured state, pinned as an equality through `lbx = ubx`,
$T_{\max} = c_T\,\mathrm{rpm}_{\max}^2$ is exactly the actuator's per-rotor saturation, and the
16-dimensional residual is

$$
y(x, u) = \left(\, p,\;\; e_{\mathrm{att}}(q, q^{\mathrm{ref}}),\;\; v,\;\; \omega,\;\; u \,\right),
\qquad
e_{\mathrm{att}} = \operatorname{vec}\!\left( \left(q^{\mathrm{ref}}\right)^{-1} \otimes\, q \right)
$$

The attitude error is the vector part of the relative quaternion, zero when aligned, with
$q^{\mathrm{ref}}$ fed to each node as an online parameter rather than a naive difference. The
weights are:

$$
W = \operatorname{diag}\left( Q_p,\, Q_{\mathrm{att}},\, Q_v,\, Q_\omega,\, R \right), \quad
Q_p = (200, 200, 500),\;
Q_{\mathrm{att}} = (5, 5, 200),\;
Q_v = Q_\omega = (5,5,5),\;
R = 6
$$

The heavy z-axis attitude weight, $200$ versus $5$, is what enforces **nose-first** flight: yaw
tracking is nearly as expensive as position error.

### One sequential quadratic programming real-time iteration per tick

The solver is acados-generated C with RK4 integration, a Gauss-Newton Hessian, and the HPIPM QP
solver with partial condensing. It runs **SQP-RTI**: a *single* SQP iteration per control tick,
never solved to convergence. That's sound because it warm-starts from the last tick's solution,
already near-optimal for a problem that shifted by only one 4 ms tick. It's what makes the mean
solve ~0.13 ms against the 4 ms control period. The first run compiles the generated solver once and caches it in
`~/.cache/nexus/acados_codegen/`, the reason you provision acados rather than pip-install it.

Each `exchange()` tick:

1. **Hold hover** until the operator has handed over a reference.
2. **Read the measured state** from Newton's `body_q` and `body_qd` and adapt it. Rotate the
   world-frame $\omega$ into the body frame. Re-express the Forward Right Down (FRD) authored
   body, with thrust along $-z_b$, in the upright NMPC convention, with thrust along $+z_b$. That
   uses the fixed 180°-about-x flip $q_{\mathrm{nmpc}} = q_{\mathrm{meas}} \otimes q_{\mathrm{flip}}$,
   $\omega_{\mathrm{nmpc}} = \operatorname{diag}(1,-1,-1)\,\omega_{\mathrm{body}}$. Only the state
   read needs this. The emitted per-rotor thrusts are frame-independent.
3. **Write the reference**: query the operator-planned min-snap reference at $t_0 + k \cdot 50$ ms
   for each node, as that node's $y^{\mathrm{ref}}_k$. Each node's reference holds position,
   quaternion, velocity, body rate, and collective thrust from the differential-flatness lift, with
   the collective split evenly across rotors as the input reference. The reference clock anchors
   to the first exchange after `accept_setpoint`, so the trajectory plays from its own $t = 0$.
4. **Solve once, apply the first input**: take $u_0$, four thrusts in N, and invert the thrust map
   into the actuator's normalized command,
   $\mathrm{throttle}_i = \sqrt{T_i / c_T}\, /\, \mathrm{rpm}_{\max}$. That command drives
   `ArticulatedRotors`, the same DC-motor-servo actuator PX4 flies, whose near-instant thrust
   response is what the Optimal Control Problem (OCP) assumes. The old lumped first-order lag
   bloated tracking to ~0.07 m RMSE.

Accepting a fresh reference re-anchors the clock and cold-seeds the warm start, with every node at
the current state and hover thrust. So a new trajectory never inherits a stale solution.

### Where it sits in the framework

The orchestrator runs physics CUDA-graph-captured. The NMPC solve is the one per-tick host
operation. It runs at the host-exchange seam between graph replays, the captured-host-exchange
strategy. See [Execution](../design/execution.md). Two side channels feed the artifacts.
`track_err` records the per-tick realized-versus-reference position error at node 0, the APE the
example gates on. When recording, every sixth tick logs the predicted horizon positions as the
orange `controller/mpc_horizon` strip in the preceding viewer.

## Key stats

Configuration, plus illustrative figures measured on a dev RTX 5080:

| Metric | Value |
|--------|-------|
| Solver | acados HPIPM, SQP real-time iteration |
| Model | 13-state, 4 rotor-thrust inputs, `N=20` horizon |
| Planner | min-snap polynomial + differential flatness, velocity-aligned yaw |
| Course | 7 waypoints, ~18 m, ~250° heading sweep over ~12.6 s |
| Mean solve time | **0.13 ms per tick**, ~30× under the 4 ms control period → realtime |
| Track APE versus the planned reference | 0.020 m RMSE, 0.062 m max |
| Final distance to goal | 0.021 m |
| Max tilt | 15.8° |
| Nose-versus-velocity heading, median while cruising | 11.4° |

CI-measured on the pinned runner, see [Benchmarking](../reference/benchmarking.md):

<!-- example-stats: acados_nmpc -->

## Run it

acados isn't a plain pip dependency, because it code-generates and compiles a C solver, so
provision it once:

```bash
bash scripts/setup_acados.sh
uv run --extra acados -m nexus.examples acados_nmpc   # flies + asserts + writes the .rrd
```

Needs a CUDA device for the Newton sim and a C compiler for the acados code generation. The first
run compiles the generated solver, which takes a few seconds. Later runs reuse it.
