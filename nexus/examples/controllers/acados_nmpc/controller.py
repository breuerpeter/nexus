"""``AcadosNMPCController``: analytic real-time Nonlinear Model Predictive Control (NMPC) via acados, the
Controller seam.

A first-class in-process Controller, alongside the Proportional Integral Derivative (PID), PX4, sampling
Model Predictive Control (MPC), and trained-policy controllers. It solves the optimal-control problem
directly with **acados**, using the High-Performance Interior Point Method (HPIPM) quadratic-program solver
and Sequential Quadratic Programming (SQP) real-time iteration, with fixed cost weights, to **track** a
smooth flat-state reference.

A pure *tracking* controller: the planner lives with the **operator**, by default the min-snap plus
differential-flatness ``nexus.examples._lib.min_snap.MinSnapReference``, with the ruckig
``FlatnessReference`` as the jerk-limited fallback, which plans
the whole-path flat-state reference and hands it over via ``accept_setpoint(ReferenceTrajectory)``.
The controller stores that queryable reference and tracks it.

The formulation:
  * **State, 13 wide:** position, quaternion orientation, velocity, body rate.
  * **Input, 4 wide:** the four single-rotor thrusts.
  * **Cost, nonlinear least-squares:** track a full flat-state reference of position, attitude, velocity,
    and body rate, plus a thrust regulariser, weights ``Q_pos=(200,200,500)``, ``Q_att=(5,5,200)``,
    ``Q_vel=5``, ``Q_omega=5``, ``R=6``. The reference attitude/body-rate/thrust come from the position
    trajectory by differential flatness.
  * **Solver:** ``N=20`` shooting nodes, HPIPM partial condensing, SQP Real-Time Iteration (RTI) re-solved
    every control tick. The per-tick tracking error feeds the evo-style Absolute Pose Error (APE) gate.

acados / casadi are **lazy-imported**, inside the methods, so importing this module never requires the
optional ``acados`` extra; only constructing the controller does. The caller must have provisioned acados
with ``scripts/setup_acados.sh`` and set ``LD_LIBRARY_PATH`` before building the solver; the run-script
re-execs to do this.
"""

from __future__ import annotations

import os

import numpy as np

GRAVITY = 9.81
DEFAULT_CODEGEN_DIR = os.path.expanduser("~/.cache/nexus/acados_codegen/quad_nmpc")  # generated C, out of the repo


def _quat_to_rot_ca(q):
    import casadi as ca

    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _quat_mult_ca(a, b):
    import casadi as ca

    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return ca.vertcat(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_to_rot_np(qw, qx, qy, qz) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def build_acados_model(*, mass, inertia, rotor_offsets, turning_dirs, reaction_k):
    """A 13-state quadrotor model whose dynamics mirror the Newton proxy exactly: per-rotor thrust →
    body +z force and ``r×F`` moment + spin reaction, integrated as a rigid body. Inertia is the proxy's
    full body-inertia matrix; the geometry/coeffs are the same constants the actuator kernel uses.
    """
    import casadi as ca
    from acados_template import AcadosModel

    nx, nu = 13, 4
    px = ca.SX.sym("p", 3)
    q = ca.SX.sym("q", 4)  # wxyz
    v = ca.SX.sym("v", 3)
    w = ca.SX.sym("w", 3)  # body frame
    x = ca.vertcat(px, q, v, w)
    u = ca.SX.sym("u", nu)  # per-rotor thrust [N]
    q_ref = ca.SX.sym("q_ref", 4)  # online parameter: reference quaternion for the attitude-error residual

    rot = _quat_to_rot_ca(q)
    thrust = u[0] + u[1] + u[2] + u[3]
    # body torque: Σ r_i × [0,0,T_i]  +  [0,0, reaction_k·T_i·dir_i], per-rotor offset thrust → r×F
    tau = ca.SX.zeros(3)
    for i in range(nu):
        rx, ry, _ = (float(c) for c in rotor_offsets[i])
        tau[0] += ry * u[i]
        tau[1] += -rx * u[i]
        tau[2] += reaction_k * u[i] * float(turning_dirs[i])
    inertia = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
    inv_inertia = ca.DM(np.linalg.inv(inertia))
    inertia_dm = ca.DM(inertia)

    p_dot = v
    q_dot = 0.5 * _quat_mult_ca(q, ca.vertcat(0.0, w[0], w[1], w[2]))
    v_dot = ca.vertcat(0.0, 0.0, -GRAVITY) + (1.0 / float(mass)) * ca.mtimes(rot, ca.vertcat(0.0, 0.0, thrust))
    w_dot = ca.mtimes(inv_inertia, tau - ca.cross(w, ca.mtimes(inertia_dm, w)))
    f_expl = ca.vertcat(p_dot, q_dot, v_dot, w_dot)

    model = AcadosModel()
    model.name = "quad_nmpc"
    model.x = x
    model.u = u
    model.p = q_ref
    model.f_expl_expr = f_expl
    xdot = ca.SX.sym("xdot", nx)
    model.xdot = xdot
    model.f_impl_expr = xdot - f_expl
    # attitude error = vector part of (q_ref⁻¹ ⊗ q); zero when aligned. Used in the cost residual below.
    q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
    att_err = _quat_mult_ca(q_ref_inv, q)[1:4]
    return model, att_err


class AcadosNMPCController:
    """Receding-horizon NMPC. Each control tick it seeds the current measured state, sets
    a full flat-state reference over the horizon, queried from the operator-planned reference, runs one SQP
    real-time iteration with HPIPM, and applies the first rotor-thrust command, converted to the actuator's
    normalised throttle. The solver warm-starts from the earlier solution, which is what makes one
    real-time iteration per tick enough.

    This controller doesn't own the reference: the operator plans it, min-snap → flatness, and hands it
    over via :meth:`accept_setpoint`. Before a reference arrives, the one tick between the run starting and
    the operator's first host-seam tick, the controller holds hover.
    """

    def __init__(
        self,
        *,
        mass,
        inertia,
        rotor_offsets,
        turning_dirs,
        ct,
        rpm_max,
        dt,
        horizon=20,  # N=20 shooting nodes
        tf=1.0,  # horizon length [s] → node dt = tf/N = 50 ms
        reaction_k=0.05,
        q_pos=(200.0, 200.0, 500.0),
        q_att=(5.0, 5.0, 200.0),
        q_vel=(5.0, 5.0, 5.0),
        q_omega=(5.0, 5.0, 5.0),
        r_thrust=(6.0, 6.0, 6.0, 6.0),
        snapshot_every=6,  # log the predicted horizon every Nth tick, viz only
        codegen_dir=DEFAULT_CODEGEN_DIR,  # where acados writes the generated C solver
    ):
        self.n = len(turning_dirs)
        self.dt = float(dt)
        self.N = int(horizon)
        self.node_dt = float(tf) / self.N
        self.ct = float(ct)
        self.rpm_max = float(rpm_max)
        self.t_max = self.ct * self.rpm_max * self.rpm_max  # per-rotor max thrust [N] = actuator clamp at throttle 1
        self.hover = float(mass) * GRAVITY / self.n
        self.reference = None  # the operator hands this over via accept_setpoint(ReferenceTrajectory)
        self._ref_step0 = None  # the control step at which the active reference started, the time anchor
        self._logger = None  # the orchestrator hands over the Logger, None when off; gates the horizon viz
        self.track_err = []  # per-tick ‖realized − reference‖ world position error, the evo-style APE gate
        self.snapshot_every = int(snapshot_every)
        self.codegen_dir = str(codegen_dir)
        self._step = 0
        self._first = True
        self.solve_time_total = 0.0  # accumulated acados solve time → mean per-tick, the realtime metric

        self._solver = self._build_solver(
            mass=mass, inertia=inertia, rotor_offsets=rotor_offsets, turning_dirs=turning_dirs,
            reaction_k=reaction_k, tf=tf,
            weights=np.array([*q_pos, *q_att, *q_vel, *q_omega, *r_thrust], dtype=np.float64),
        )  # fmt: skip

    def _build_solver(self, *, mass, inertia, rotor_offsets, turning_dirs, reaction_k, tf, weights):
        from acados_template import AcadosOcp, AcadosOcpSolver

        model, att_err = build_acados_model(
            mass=mass, inertia=inertia, rotor_offsets=rotor_offsets, turning_dirs=turning_dirs, reaction_k=reaction_k
        )
        import casadi as ca

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = float(tf)
        nx, nu = 13, 4

        # Nonlinear least-squares tracking residual: [pos, attitude-error, vel, body-rate, thrust].
        px, v, w = model.x[0:3], model.x[7:10], model.x[10:13]
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(px, att_err, v, w, model.u)  # NY = 16
        ocp.model.cost_y_expr_e = ca.vertcat(px, att_err, v, w)  # terminal-node output dimension = 12
        ocp.cost.W = np.diag(weights)  # [Q_pos(3), Q_att(3), Q_vel(3), Q_omega(3), R(4)]
        ocp.cost.W_e = np.diag(weights[:12])
        ocp.cost.yref = np.zeros(16)  # set per stage each solve
        ocp.cost.yref_e = np.zeros(12)

        # Inputs: 0 ≤ per-rotor thrust ≤ t_max; the actuator saturates throttle at [0,1] ⇒ thrust [0, t_max].
        ocp.constraints.lbu = np.zeros(nu)
        ocp.constraints.ubu = np.full(nu, self.t_max)
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.x0 = np.zeros(nx)  # overwritten every tick with the measured state
        ocp.parameter_values = np.array([1.0, 0.0, 0.0, 0.0])  # q_ref, set per stage each solve

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # one real-time iteration per control tick
        ocp.code_export_directory = os.path.join(self.codegen_dir, "c_generated_code")
        os.makedirs(self.codegen_dir, exist_ok=True)
        return AcadosOcpSolver(ocp, json_file=os.path.join(self.codegen_dir, "acados_ocp.json"), verbose=False)

    def connect(self) -> None:
        pass

    def set_logger(self, logger) -> None:
        """The orchestrator hands over the Logger, ``None`` when off: the gate for the horizon viz."""
        self._logger = logger

    def close(self) -> None:  # lifecycle teardown: nothing to release, the horizon logs from exchange()
        pass

    # -- control surface: the thin setpoint seam ---------
    def accept_setpoint(self, sp) -> None:
        """Accept the operator-planned tracking reference, a :class:`ReferenceTrajectory` carrying a
        queryable ``FlatnessReference``. The controller stores it and tracks it from the next tick; the
        time anchor + the warm-start cold-seed reset so the trajectory plays from its start. Raises on a
        non-``ReferenceTrajectory`` variant, because acados tracks a reference, not a single goal.
        """
        from nexus._src.core.schema import ReferenceTrajectory

        if not isinstance(sp, ReferenceTrajectory):
            raise TypeError(f"AcadosNMPCController accepts a ReferenceTrajectory setpoint, got {type(sp).__name__}")
        self.reference = sp.reference
        self._ref_step0 = None  # re-anchor: the trajectory plays from t=0 at the next exchange
        self._first = True  # cold-seed the warm start at the current state for the fresh reference

    def reference_path(self, n: int = 300) -> np.ndarray:
        return self.reference.reference_path(n) if self.reference is not None else np.zeros((0, 3))

    def _hover_throttle(self) -> np.ndarray:
        """Per-rotor hover throttle, held until the operator hands over a reference."""
        throttle = float(np.sqrt(np.clip(self.hover, 0.0, self.t_max) / self.ct) / self.rpm_max)
        return np.full(self.n, throttle, dtype=np.float32)

    def exchange(self, meas, t, timeout=None):
        from nexus._src.core import Controls

        if self.reference is None:  # the operator hasn't planned/handed the reference yet → hold hover
            self._step += 1
            return Controls(command=self._hover_throttle())

        state = meas.state
        bq = state.body_q.numpy()[0]
        bqd = state.body_qd.numpy()[0]

        # measured state → NMPC state. Newton stores body_qd = (v_world, ω_world); the model uses a
        # body-frame rate, so rotate ω into the body frame. Quaternion order: warp xyzw → model wxyz.
        pos = bq[:3].astype(np.float64)
        qw, qx, qy, qz = float(bq[6]), float(bq[3]), float(bq[4]), float(bq[5])
        rot = _quat_to_rot_np(qw, qx, qy, qz)  # world ← real Forward Right Down (FRD) body
        vel = bqd[:3].astype(np.float64)
        omega_body = rot.T @ bqd[3:6].astype(np.float64)  # ω in the real FRD body frame
        # FRD → NMPC-upright adapter. The framework's vehicle USDs carry the FRD convention, body +z down and
        # thrust along −body z, handled by the production aero kernel's sign, but the NMPC model stands
        # upright, thrust = +body z. Re-express the measured attitude/rate in the NMPC's frame via the fixed
        # 180°-about-x flip q_flip=(w0,x1,y0,z0): q_nmpc = q_meas ⊗ q_flip, ω_nmpc = diag(1,−1,−1)·ω.
        # Spawned at FRD, q_meas = q_flip ⇒ q_nmpc = identity, the NMPC's hover attitude. The per-rotor
        # throttle the NMPC emits is frame-independent, so only the state read needs this remap.
        q_nmpc = np.array([-qx, qw, qz, -qy])  # wxyz, = quat_mult(q_meas, q_flip)
        omega_nmpc = np.array([omega_body[0], -omega_body[1], -omega_body[2]])
        x0 = np.concatenate([pos, q_nmpc, vel, omega_nmpc])

        if self._first:  # cold start: seed every node at the current state + hover thrust
            for k in range(self.N + 1):
                self._solver.set(k, "x", x0)
            for k in range(self.N):
                self._solver.set(k, "u", np.full(self.n, self.hover))
            self._first = False

        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)
        # Anchor the reference clock to the first exchange after accepting a reference, so the planned
        # trajectory plays from t=0 here; the operator set its start to about this position.
        if self._ref_step0 is None:
            self._ref_step0 = self._step
        t0 = (self._step - self._ref_step0) * self.dt
        for k in range(self.N + 1):
            p_r, q_r, v_r, w_r, thrust_c = self.reference.flat_state_at(t0 + k * self.node_dt)
            if k == 0:  # realized-minus-reference position error at the current tick, world frame, time-synced
                self.track_err.append(float(np.linalg.norm(pos - p_r)))
            self._solver.set(k, "p", q_r)  # reference quaternion for the attitude-error residual
            if k < self.N:
                u_ref = np.full(self.n, thrust_c / self.n)  # collective split evenly across rotors
                self._solver.set(k, "yref", np.concatenate([p_r, np.zeros(3), v_r, w_r, u_ref]))
            else:
                self._solver.set(k, "yref", np.concatenate([p_r, np.zeros(3), v_r, w_r]))

        self._solver.solve()
        self.solve_time_total += float(self._solver.get_stats("time_tot"))
        u0 = self._solver.get(0, "u")  # per-rotor thrust [N]
        # thrust [N] → actuator throttle: thrust = ct·(throttle·rpm_max)² ⇒ throttle = √(thrust/ct)/rpm_max
        throttle = np.sqrt(np.clip(u0, 0.0, self.t_max) / self.ct) / self.rpm_max

        # Component-owned horizon viz, event-driven: when recording, emit the predicted NMPC horizon to
        # controller/mpc_horizon every snapshot_every ticks; the newest entry at each time means scrubbing
        # shows the active horizon. Gated on self._logger so the solver-state extraction never runs when
        # not recording. The orchestrator set the timeline at tick start, so this just hands the path to
        # the Logger.
        if self._logger is not None and self._step % self.snapshot_every == 0:
            pred = np.array([self._solver.get(k, "x")[:3] for k in range(self.N + 1)], dtype=np.float32)
            self._logger.log_strip("controller/mpc_horizon", pred, color=(255, 140, 0))
        self._step += 1
        return Controls(command=throttle.astype(np.float32))
