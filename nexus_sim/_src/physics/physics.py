"""Physics, backed by NVIDIA Newton.

Owns the model build + solver + double-buffered step + the settle after
placing the vehicle. Consumes a per-body Wrench at the shared ``state.body_f`` buffer, which the force
elements add to before ``step``, and the control inputs the command elements write into the model's
``newton.Control``; ``step`` steps every Newton actuator the vehicle declares, then the solver.

**The solver is configurable** via ``cfg["physics"]["solver"]``: ``mujoco``, the default, is the
high-fidelity contact solver the Software In The Loop (SITL) path uses and the determinism authority
on CPU; the gradient-capable ``semi_implicit`` / ``featherstone`` integrators back the
design-optimization path, since ``mujoco-warp`` isn't differentiable, see
``nexus_sim/examples/design_opt/design_optimization.py``. All three share the ``solver.step(state_in, state_out,
control, contacts, dt)`` signature, so only the constructor differs.

**One class, three regimes**, selected by ``cfg["physics"]``, so the production runtime and the
standalone Model Predictive Control (MPC) examples drive the same ``reset``, ``clear_forces`` and ``step``:

* *Production*, the default: build the model from ``vehicle_usd`` + ``cfg``, with ground plane, scene,
  and the USD-authored motors, settle on the ground at ``reset``, contacts on, MuJoCo solver. The SITL
  path.
* *Free articulated*, with ``spawn`` set and contacts optional: the real multi-body vehicle placed in free
  flight at ``spawn['pos']`` with no ground-settle, rotors pre-spun to hover via ``prespin: 'hover'``. The
  acados forward-validation regime, ``nexus_sim/examples/controllers/acados_nmpc/waypoint_tracking.py``.
* *Free single-body*: pass a pre-built ``model=``, the Universal Scene Description (USD) file collapsed
  to one rigid body, see :func:`~nexus_sim.examples._lib.single_body.collapse_to_single_body`, with the
  ``semi_implicit`` solver and ``contacts: False``; spawned upright via the Forward Right Down (FRD)
  flip, ``attitude: 'flip'``. The differentiable sampling-MPC regime,
  ``nexus_sim/examples/controllers/sampling_mpc/obstacle_slalom.py``.

The reset / step paths branch on ``model.body_count``: a single collapsed body uses maximal
coordinates, ``body_q``, with no joint control; the articulated vehicle uses generalized
coordinates, ``joint_q``, + the joint control buffer. Vehicle USDs follow FRD authoring, body +z down,
and spawned free they sit upright; see ``docs/design/conventions.md``.

``eval_fk`` is intentionally not re-run after the actuator
joint update, so ``update_body_f`` reads the earlier step's joint pose, a
one-step lag in thrust direction.

Two device stages, ``clear`` and ``step``: the collide + solver.step + double-buffer ping-pong
is a static launch over persistent buffers, so it joins a CUDA graph with the command and force stages
between them. ``step`` first zeroes ``control.joint_f`` and steps the Newton actuators, where Newton's docs
place the call, each over double-buffered state copied back in place after the step: the graph-safe twin
of the state swap, since a Python swap would freeze at a CUDA graph's capture-time binding.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from nexus_sim._src.core import logger
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.labels import leaf_keys
from nexus_sim._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints

from ..scene.ingest import add_scene  # ingestion only: the handlers are the renderer's side
from ..scene.site import GRAVITY

STABILIZE_VEL_THRESHOLD = 0.01  # m/s
STABILIZE_MIN_STEPS = 10
STABILIZE_MAX_STEPS = 10000


def _state_arrays(act_state) -> list[wp.array]:
    """The Warp arrays inside a composed ``newton.actuators`` Actuator.State, the delay + controller
    sub-states, in a construction-stable order, so two states built by the same ``actuator.state()``
    pair up positionally for the in-place copy-back.
    """
    out: list[wp.array] = []
    if act_state is None:
        return out
    for sub in (getattr(act_state, "delay_state", None), getattr(act_state, "controller_state", None)):
        if sub is None:
            continue
        for v in vars(sub).values():
            if isinstance(v, wp.array):
                out.append(v)
    return out


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
    def __init__(self, *, vehicle_usd=None, model=None, cfg: dict, njmax: int = 224, step_actuators: bool = True):
        self.cfg = cfg
        phys = cfg["physics"]
        self.sim_dt = phys["dt"]
        self.vehicle_usd = vehicle_usd
        self.contacts_on = phys.get("contacts", True)
        self.spawn_cfg = phys.get("spawn")  # None → ground-settle; dict → free placement at spawn['pos']

        if model is not None:
            # Pre-built model: the single-body path passes the USD collapsed via
            # :func:`~nexus_sim.examples._lib.single_body.collapse_to_single_body`, for the sampling-MPC
            # real sim + its batched rollout twin, and design-opt; the collapse lives in that one function, not here.
            self.model = model
        else:
            logger.info(f"Default warp device: {wp.get_device()}")
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            add_scene(builder, cfg)  # scene USD -> builder, exactly as for the vehicle USD
            vehicle_usd.build(builder)
            # The site's gravity, the one value the IMU reports too. add_usd resets the builder's gravity
            # from any PhysicsScene the USD holds, authored or not, so it is set after the last add.
            builder.gravity = -GRAVITY
            # The vehicle USD authors the motors as NewtonActuator prims on the actuator joints, and add_usd
            # parses them onto model.actuators, which step() steps before the solver. A collapsed single
            # body has no joints, so no motors.
            self.model = builder.finalize()
            if vehicle_usd is not None:
                vehicle_usd.model_debug_print(self.model)

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
        # Every Newton actuator the vehicle declares, rotor motor or not, stepped before the solver. Each
        # keeps two copies of its state, because ControllerPID is stateful and a Python swap would freeze at
        # the binding a CUDA graph captured: step() copies next → current in place after each step.
        # ``step_actuators=False`` leaves them idle, for a run on the old actuator seam: the examples'
        # ``Rotors`` holds its own motor model and sums its wrench on the base body, so the rotor joints
        # hang free there, as they did before physics stepped the motors.
        self._actuators = list(getattr(self.model, "actuators", None) or []) if step_actuators else []
        self._actuator_states = [(a.state(), a.state()) for a in self._actuators]
        self._actuator_copies = [
            list(zip(_state_arrays(cur), _state_arrays(nxt), strict=True)) for cur, nxt in self._actuator_states
        ]
        self.contacts = self.model.collide(self.state0) if self.contacts_on else None
        self.base_index = self._find_base()
        # The base body's label, the leaf of its model label when unique: Sim's default vehicle entity,
        # ``sim.physics[sim.base_body]``, and the Recorder's key for its history.
        self.base_body = leaf_keys([str(k) for k in self.model.body_label])[self.base_index]
        logger.info(f"control dim {self.model.joint_dof_count}")

        # Free-placement rotor pre-spin: start the articulated vehicle at the hover rotor speed so it's in
        # equilibrium, with no spin-up dip. Ω = √(mg/(n·ct))/RPM_PER_RADS from the model mass + the USD ct,
        # read straight from the vehicle, the single hash-pinned actuator source, not a config value.
        self._rotor_vel_dofs = None
        if self.articulated and self.spawn_cfg and self.spawn_cfg.get("prespin") == "hover":
            if self.vehicle_usd is None:
                raise ValueError("prespin='hover' needs a vehicle_usd to read ct from the USD")
            ct = float(self.vehicle_usd.actuator_params()["ct"])
            self._rotor_vel_dofs, _pos, _bodies, _base = find_rotor_joints(self.model, self.vehicle_usd.rotor_joints())
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

    def _find_base(self) -> int:
        """The index of the base body, the vehicle's airframe: the declared rotor joints' shared parent, else body 0."""
        try:  # articulated: the declared rotor joints' shared parent
            joints = self.vehicle_usd.rotor_joints() if self.vehicle_usd is not None else []
            return find_rotor_joints(self.model, joints)[3]
        except ValueError:
            return 0  # single body, no rotor joints: body 0 is the base body

    def clear_forces(self, state) -> None:
        state.clear_forces()

    def step(self, state, dt):
        """Step the Newton actuators into ``control.joint_f``, zeroed first, then the solver, which
        consumes ``joint_f`` and the force elements' ``body_f`` in one solve.
        """
        if self._actuators:
            self.control.joint_f.zero_()
            for actuator, (cur, nxt), copies in zip(
                self._actuators, self._actuator_states, self._actuator_copies, strict=True
            ):
                actuator.step(state, self.control, cur, nxt, dt)
                for dst, src in copies:  # current ← next, in place: the graph-safe double-buffer
                    wp.copy(dst, src)
        contacts = self.model.collide(state) if self.contacts_on else None
        self.solver.step(state, self.state1, self.control, contacts, dt)
        state.assign(self.state1)
        return state

    def stages(self) -> list[Stage]:
        """The ``clear`` and ``step`` device stages; the loop runs the command and force stages between them."""
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]

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
        start origin, so PX4 establishes lockstep / Global
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
