"""Physics plugin backed by NVIDIA Newton, per architecture.md §2 and §5.

Lifts the bridge's model build + solver + double-buffered step + the settle after
placing the vehicle. Consumes a per-body Wrench at the shared ``state.body_f`` buffer, which
newton-actuators writes before ``step``, and doesn't launch the actuator kernels:
that split is the only structural change from the bridge's fused ``_simulate_physics``.

**The solver is configurable** via ``cfg["physics"]["solver"]``: ``mujoco``, the default, is the
high-fidelity contact solver the Software In The Loop (SITL) path uses and the determinism authority
on CPU; the gradient-capable ``semi_implicit`` / ``featherstone`` integrators back the
design-optimization path, since ``mujoco-warp`` isn't differentiable, see
``nexus/examples/design_opt/design_optimization.py``. All three share the ``solver.step(state_in, state_out,
control, contacts, dt)`` signature, so only the constructor differs.

**One class, three regimes**, selected by ``cfg["physics"]``, so the production runtime and the
standalone Model Predictive Control (MPC) examples drive the same ``reset``/``clear_forces``/``step`` seam:

* *Production*, the default: build the model from ``vehicle_builder`` + ``cfg``, with ground plane, scene,
  and the USD-authored motors, settle on the ground at ``reset``, contacts on, MuJoCo solver. The SITL
  path.
* *Free articulated*, with ``spawn`` set and contacts optional: the real multi-body vehicle placed in free
  flight at ``spawn['pos']`` with no ground-settle, rotors pre-spun to hover via ``prespin: 'hover'``. The
  acados forward-validation regime, ``nexus/examples/controllers/acados_nmpc/waypoint_tracking.py``.
* *Free single-body*: pass a pre-built ``model=``, the Universal Scene Description (USD) file collapsed
  to one rigid body, see :func:`~nexus.examples._lib.single_body.collapse_to_single_body`, with the
  ``semi_implicit`` solver and ``contacts: False``; spawned upright via the Forward Right Down (FRD)
  flip, ``attitude: 'flip'``. The differentiable sampling-MPC regime,
  ``nexus/examples/controllers/sampling_mpc/obstacle_slalom.py``.

The reset / step paths branch on ``model.body_count``: a single collapsed body uses maximal
coordinates, ``body_q``, with no joint control; the articulated vehicle uses generalized
coordinates, ``joint_q``, + the joint control buffer. Vehicle USDs follow FRD authoring, body +z down,
and spawned free they sit upright; see ``docs/conventions.md``.

Faithful detail preserved: ``eval_fk`` is intentionally not re-run after the actuator
joint update, so ``update_body_f`` reads the earlier step's joint pose, a
one-step lag in thrust direction: exactly the bridge's behavior.

Marked Capturable: the collide + solver.step + double-buffer ping-pong is a static
launch over persistent buffers; with the actuator it forms the contiguous
captured region, with the controller excluded as the host boundary. The eager
slice defers capture.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from nexus._src.core import logger
from nexus._src.recording.recorder import leaf_keys
from nexus._src.recording.state import (
    BODY_FIELDS,
    BODY_WIDTH,
    decode_body,
    make_decode_joint,
    record_body,
    record_joint,
)
from nexus._src.vehicle.actuators.layout import RPM_PER_RADS, find_rotor_joints

from ..scene.ingest import add_scene  # ingestion only: the handlers are the renderer's side

STABILIZE_VEL_THRESHOLD = 0.01  # m/s
STABILIZE_MIN_STEPS = 10
STABILIZE_MAX_STEPS = 10000
GRAVITY = 9.81


def make_solver(name: str, model, *, njmax: int = 224):
    """Construct a Newton solver by config name: the only solver-specific code.

    ``mujoco`` is the default contact solver, for SITL + the CPU determinism authority;
    ``semi_implicit`` / ``featherstone`` are the gradient-capable integrators used by the
    differentiable design-optimization rollout. All expose the same ``step`` signature.
    """
    name = (name or "mujoco").lower()
    if name == "mujoco":
        return newton.solvers.SolverMuJoCo(model, njmax=njmax)
    if name in ("semi_implicit", "semi-implicit", "semiimplicit"):
        return newton.solvers.SolverSemiImplicit(model)
    if name == "featherstone":
        return newton.solvers.SolverFeatherstone(model)
    raise ValueError(f"unknown physics solver {name!r} (expected mujoco|semi_implicit|featherstone)")


class NewtonPhysics:
    capturable = True

    def __init__(self, *, vehicle_builder=None, model=None, cfg: dict, njmax: int = 224):
        self.cfg = cfg
        # Component-owned groundtruth logging, since physics owns the true state: log() draws the generic
        # scene via the shared logger. The orchestrator hands over the Logger, self._logger, None when off,
        # and calls log() at the host seam, outside the captured graph, only when recording. The flown path
        # is not logged here: it is the base body's recorded position series, drawn by the recorder's
        # Rerun adapter at teardown.
        self._logger = None
        # Component-owned observation taps, the read-side twin of logging: when Sim observes, the
        # orchestrator hands over a Recorder and physics registers one channel per body + per joint, the
        # full model state by label. record_wp() then snapshots them each tick INSIDE the captured graph
        # with no D2H, so observing never forces the eager strategy. Empty when not observing.
        self._body_taps: list = []  # (channel, body_index)
        self._joint_taps: list = []  # (channel, q_start, nq, qd_start, nqd)
        self.base_body = None  # the base body's label, discovered: Sim's default "vehicle" entity
        phys = cfg["physics"]
        self.sim_dt = phys["dt"]
        self.vehicle_builder = vehicle_builder
        self.contacts_on = phys.get("contacts", True)
        self.spawn_cfg = phys.get("spawn")  # None → ground-settle; dict → free placement at spawn['pos']

        if model is not None:
            # Pre-built model: the single-body path passes the USD collapsed via
            # :func:`~nexus.examples._lib.single_body.collapse_to_single_body`, for the sampling-MPC
            # real sim + its batched rollout twin, and design-opt; the collapse lives in that one seam, not here.
            self.model = model
        else:
            logger.info(f"Default warp device: {wp.get_device()}")
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            add_scene(builder, cfg)  # scene USD -> builder, exactly as for the vehicle USD
            vehicle_builder.build(builder)
            # The vehicle USD authors the motors as NewtonActuator prims on the actuator joints, and add_usd
            # parses them onto model.actuators; the pairing guard refuses a model with none. A collapsed
            # single body has no joints, so no motors.
            self.model = builder.finalize()
            if vehicle_builder is not None:
                vehicle_builder.model_debug_print(self.model)

        # A collapsed single body uses maximal coordinates, body_q, with no joint actuation; the articulated
        # vehicle uses generalized coordinates, joint_q, + the joint control buffer + contacts.
        self.articulated = self.model.body_count > 1

        solver_name = phys.get("solver", "mujoco")
        logger.info(f"physics solver: {solver_name}")
        self.solver = make_solver(solver_name, self.model, njmax=njmax)
        self.state0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state0)
        self.state1 = self.model.state()
        self.control = self.model.control() if self.articulated else None
        self.contacts = self.model.collide(self.state0) if self.contacts_on else None
        logger.info(f"control dim {self.model.joint_dof_count}")

        # Free-placement rotor pre-spin: start the articulated vehicle at the hover rotor speed so it's in
        # equilibrium, with no spin-up dip. Ω = √(mg/(n·ct))/RPM_PER_RADS from the model mass + the USD ct,
        # read straight from the vehicle, the single hash-pinned actuator source, not a config value.
        self._rotor_vel_dofs = None
        if self.articulated and self.spawn_cfg and self.spawn_cfg.get("prespin") == "hover":
            if self.vehicle_builder is None:
                raise ValueError("prespin='hover' needs a vehicle_builder to read ct from the USD")
            ct = float(self.vehicle_builder.actuator_params()["ct"])
            self._rotor_vel_dofs, _pos, _bodies, _base = find_rotor_joints(self.model)
            mass = float(self.model.body_mass.numpy().sum())
            hover_thrust = mass * GRAVITY / len(self._rotor_vel_dofs)
            self._hover_omega = float(np.sqrt(hover_thrust / ct) / RPM_PER_RADS)

    @property
    def current_state(self):
        """The live ``newton.State``, ``state0``, which ``step`` mutates in place: the uniform
        accessor the shared assembly and the renderer read.
        """
        return self.state0

    def reset(self):
        if self.spawn_cfg is None:
            self._settle()  # production: step with zero forces until the body settles on the ground
        elif self.articulated:
            self._spawn_articulated()
        else:
            self._spawn_single_body()
        return self.state0

    def set_logger(self, logger) -> None:
        """The orchestrator hands over the Logger, ``None`` when off: the gate for the scene physics draws."""
        self._logger = logger

    def log(self, t) -> None:
        """Component log step, which the orchestrator calls at the host seam, outside the captured graph, at
        the decimated per-tick rate, since the orchestrator throttles its fan-out to ``log_hz``, only when
        recording. Physics owns the groundtruth state, so it draws the generic scene via the shared
        ``self._logger.log_state``. ``step`` mutates ``self.state0`` in place, so it's always current.
        """
        self._logger.log_state(self.state0, float(t.sim_time))

    def set_recorder(self, recorder) -> None:
        """The orchestrator hands over the Recorder, gated by Sim's ``observe``. Physics owns the full
        model state, so it registers one channel per body, ``physics/body/<label>``, and per joint,
        ``physics/joint/<label>``, from the finalized model's labels: every body/joint addressable by
        name. Discovery finds the base body's label, ``base_body``, as the actuator joints' shared parent, else body 0,
        for Sim's default vehicle entity: nothing hardcoded; it works for whatever model loads. That body's
        channel is also the Recorder's ``base_body``, the source of the flown path.
        """
        m = self.model
        src = type(self).__name__
        try:
            _vel, _pos, _bodies, base = find_rotor_joints(m)  # articulated: the actuator joints' shared parent
        except ValueError:
            base = 0  # single body, no actuator joints: body 0 is the base body
        body_keys = leaf_keys(list(m.body_label))  # friendly names: leaf when unique, else the full path
        self.base_body = body_keys[base]
        self._body_taps = [
            (
                recorder.channel(
                    f"physics/body/{key}", width=BODY_WIDTH, decode=decode_body, fields=BODY_FIELDS, source=src
                ),
                i,
            )
            for i, key in enumerate(body_keys)
        ]
        recorder.base_body = self._body_taps[base][0]  # the flown path is this channel's position series
        self._joint_taps = []
        if m.joint_count and self.state0.joint_q is not None:
            joint_keys = leaf_keys(list(m.joint_label))
            qs = m.joint_q_start.numpy()  # length joint_count+1, with a sentinel, → clean per-joint slices
            qds = m.joint_qd_start.numpy()
            for j, key in enumerate(joint_keys):
                nq, nqd = int(qs[j + 1] - qs[j]), int(qds[j + 1] - qds[j])
                ch = recorder.channel(
                    f"physics/joint/{key}",
                    width=1 + nq + nqd,
                    decode=make_decode_joint(nq, nqd),
                    fields=(("q", nq), ("qd", nqd)),
                    source=src,
                )
                self._joint_taps.append((ch, int(qs[j]), nq, int(qds[j]), nqd))

    def record_wp(self) -> None:
        """Capturable observation tap, the read-side twin of :meth:`log`: snapshot every body + joint of
        the live ``state0``, the persistent buffers the graph advances, into their Recorder channels with
        NO host readback, so it joins the captured graph. Launched each tick by the orchestrator inside the
        device region. A no-op when not observing.
        """
        if not self._body_taps:
            return
        bq, bqd = self.state0.body_q, self.state0.body_qd
        for ch, i in self._body_taps:
            wp.launch(record_body, dim=1, inputs=(bq, bqd, i, ch.dt, ch.maxlen, ch.buf, ch.counter))
        if self._joint_taps:
            jq, jqd = self.state0.joint_q, self.state0.joint_qd
            for ch, qs, nq, qds, nqd in self._joint_taps:
                wp.launch(record_joint, dim=1, inputs=(jq, jqd, qs, nq, qds, nqd, ch.dt, ch.maxlen, ch.buf, ch.counter))

    def clear_forces(self, state) -> None:
        state.clear_forces()

    def step(self, state, env, dt):
        contacts = self.model.collide(state) if self.contacts_on else None
        self.solver.step(state, self.state1, self.control, contacts, dt)
        state.assign(self.state1)
        return state

    def _spawn_single_body(self) -> None:
        """Place a single rigid body, in maximal coords, in free flight at ``spawn['pos']``, zero velocity.
        ``attitude`` ``'flip'`` applies the 180°-about-X, ``[1,0,0,0]`` as an x, y, z, w quaternion, that flips
        an FRD body, body +z down, upright in the world, so thrust along −body z goes to world +z: the
        standard framework placement for FRD vehicles.
        """
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state0)
        bq = self.state0.body_q.numpy()
        bq[0, 0:3] = self.spawn_cfg["pos"]
        if self.spawn_cfg.get("attitude", "flip") == "flip":
            bq[0, 3:7] = [1.0, 0.0, 0.0, 0.0]
        self.state0.body_q.assign(bq)
        bqd = self.state0.body_qd.numpy()
        bqd[0, :] = 0.0
        self.state0.body_qd.assign(bqd)

    def _spawn_articulated(self) -> None:
        """Place the articulated vehicle, in generalized coords, in free flight at ``spawn['pos']``, keeping
        the USD's authored FRD base orientation, with the rotors optionally pre-spun to hover. The native FRD
        orientation is a must: the propeller aero kernel's thrust-sign assumes body +z down, so
        re-spawning upright would drive the thrust into the ground.
        """
        jq = self.model.joint_q.numpy().copy()
        jqd = self.model.joint_qd.numpy().copy()
        jq[0:3] = self.spawn_cfg["pos"]  # reposition the free-joint base; keep the authored orientation
        jqd[:] = 0.0
        if self._rotor_vel_dofs is not None:
            for d in self._rotor_vel_dofs:
                jqd[d] = self._hover_omega
        self.state0.joint_q.assign(jq)
        self.state0.joint_qd.assign(jqd)
        newton.eval_fk(self.model, self.state0.joint_q, self.state0.joint_qd, self.state0)

    def _settle(self) -> None:
        """Step physics with zero forces until the body settles to rest at the
        start origin, the bridge stabilize: so PX4 establishes lockstep / Global
        Positioning System (GPS) origin at the resting pose, not the start height.
        """
        linear_vel = float("inf")
        for i in range(STABILIZE_MAX_STEPS):
            self.state0.clear_forces()
            self.contacts = self.model.collide(self.state0)
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.sim_dt)
            self.state0.assign(self.state1)
            wp.synchronize()
            body_qd = self.state0.body_qd.numpy()
            linear_vel = np.linalg.norm(body_qd[0, :3])
            if i > STABILIZE_MIN_STEPS and linear_vel < STABILIZE_VEL_THRESHOLD:
                logger.info(f"Stabilized after {i + 1} steps (vel={linear_vel:.4f} m/s)")
                return
        logger.warning(f"Stabilization did not converge after {STABILIZE_MAX_STEPS} steps (vel={linear_vel:.4f} m/s)")
