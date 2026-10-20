"""Physics, backed by Newton.

Owns the model build + solver + double-buffered step + the settle after
placing the vehicle. Consumes a per-body Wrench at the shared ``state.body_f`` buffer, which the force
elements add to before ``step``, and the control inputs the command elements write into the model's
``newton.Control``; ``step`` steps every Newton actuator the vehicle declares, then the solver.

**The solver is configurable** via ``cfg["physics"]["solver"]``: ``mujoco``, the default, is the
high-fidelity contact solver the Software In The Loop (SITL) path uses and the determinism authority
on CPU; the gradient-capable ``semi_implicit`` / ``featherstone`` integrators stay selectable, though the
gradient examples step their own recorded rollout of the same vehicle, since ``mujoco-warp`` isn't
differentiable, see ``nexus_uav/examples/_lib/articulated.py``. All three share the ``solver.step(state_in,
state_out, control, contacts, dt)`` signature, so only the constructor differs.

**One class, three regimes**, selected by ``cfg["physics"]``, so the production runtime and the
standalone Model Predictive Control (MPC) examples drive the same ``reset``, ``clear_forces`` and ``step``:

* *Production*, the default: build the model from ``vehicle_usd`` + ``cfg``, with ground plane, scene,
  and the USD-authored motors, settle on the ground at ``reset``, contacts on, MuJoCo solver. The SITL
  path.
* *Free articulated*, with ``spawn`` set and contacts optional: the real multi-body vehicle placed in free
  flight at ``spawn['pos']`` with no ground-settle, rotors pre-spun to hover via ``prespin: 'hover'``. The
  regime of the acados, Proportional Integral Derivative (PID) and sampling MPC examples.
* *Pre-built*: pass a finalized ``model=``, as a test that builds its own model does; a single rigid
  body spawns upright via the Forward Right Down (FRD) flip, ``attitude: 'flip'``.

The reset / step paths branch on ``model.body_count``: a single collapsed body uses maximal
coordinates, ``body_q``, with no joint control; the articulated vehicle uses generalized
coordinates, ``joint_q``, + the joint control buffer. Vehicle USDs follow FRD authoring, body +z down,
and spawned free they sit upright; see ``docs/design/conventions.md``.

Two device stages, ``clear`` and ``step``: the collide + solver.step + double-buffer ping-pong
is a static launch over persistent buffers, so it joins a CUDA graph with the command and force stages
between them. ``step`` first steps the vehicle's motors, :class:`~nexus_uav._src.physics.motors.Motors`,
into ``control.joint_f``, then the solver.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from nexus_uav._src.core import logger
from nexus_uav._src.core.interfaces import Stage
from nexus_uav._src.core.labels import leaf_keys
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, find_rotor_joints, rotor_of

from ..scene import world
from ..scene.ingest import add_scene  # ingestion only: the handlers are the renderer's side
from ..scene.world import GRAVITY
from .motors import Motors
from .outputs import Outputs
from .vehicle import NoRotor

STABILIZE_VEL_THRESHOLD = 0.01  # m/s
STABILIZE_MIN_STEPS = 10
STABILIZE_MAX_STEPS = 10000


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
    def __init__(self, *, vehicle_usd=None, model=None, cfg: dict, njmax: int = 224):
        self.cfg = cfg
        phys = cfg["physics"]
        self.sim_dt = phys["dt"]
        self.vehicle_usd = vehicle_usd
        self.contacts_on = phys.get("contacts", True)
        self.spawn_cfg = phys.get("spawn")  # None → ground-settle; dict → free placement at spawn['pos']

        if model is not None:
            # Pre-built model: a test that builds its own model hands it over.
            self.model = model
        else:
            logger.info(f"Default warp device: {wp.get_device()}")
            builder = newton.ModelBuilder()
            # The world's values the scene's PhysicsScene authors, which the sensors read from the model:
            # declared before any add_usd, so the importer reads them, and derived after the last.
            world.declare(builder)
            builder.add_ground_plane()
            add_scene(builder, cfg)  # scene USD -> builder, exactly as for the vehicle USD
            vehicle_usd.build(builder)
            # The world's gravity, the one value the inertial measurement unit reports too. add_usd resets the builder's gravity
            # from any PhysicsScene the USD holds, authored or not, so it is set after the last add.
            builder.gravity = -GRAVITY
            world.derive(builder)
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
        # The plant's outputs, a PoseTwist per body and a struct per joint, which the state stage writes.
        self.outputs = Outputs(self.model, self.state0)
        # Every Newton motor the vehicle declares, rotor motor or not, stepped before the solver.
        self.motors = Motors(self.model)
        self.contacts = self.model.collide(self.state0) if self.contacts_on else None
        self.base_index = self._find_base()
        # The base body's label, the leaf of its model label when unique: Sim's default vehicle entity,
        # ``sim[f"vehicle/body/{sim.base_body}"]``, and the Recorder's key for its history.
        self.base_body = leaf_keys([str(k) for k in self.model.body_label])[self.base_index]
        logger.info(f"control dim {self.model.joint_dof_count}")

        # Free-placement rotor pre-spin: start the articulated vehicle at the hover rotor speeds so it's in
        # equilibrium, with no spin-up dip. Each rotor's Ω = √(mg/(n·ct))/RPM_PER_RADS from the model mass + its
        # own ct, read straight from the vehicle's file, the single hash-pinned source, not a config value.
        self._hover_omega: dict[int, float] = {}  # each rotor's hover speed [rad/s], by its velocity dof
        if self.articulated and self.spawn_cfg and self.spawn_cfg.get("prespin") == "hover":
            if self.vehicle_usd is None:
                raise ValueError("prespin='hover' needs a vehicle_usd to read each rotor's ct from the USD")
            rotors = self.vehicle_usd.rotors()
            mass = float(self.model.body_mass.numpy().sum())
            hover_thrust = mass * GRAVITY / len(rotors)
            for rotor in rotors:
                dof, _coord, _body, _base = rotor_of(self.model, rotor.body)
                self._hover_omega[dof] = float(np.sqrt(hover_thrust / rotor.propeller["ct"]) / RPM_PER_RADS)

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
        """The index of the base body, the vehicle's airframe: the declared rotor joints' shared parent, else body 0.

        Raises:
            ValueError: A rotor's declaration is wrong, as :func:`~nexus_uav._src.physics.vehicle.rotors` states;
                a vehicle that declares no rotor builds, with body 0 as its base.
        """
        if self.vehicle_usd is None:
            return 0
        try:
            rotors = self.vehicle_usd.rotors()  # a wrong rotor declaration fails the build here
        except NoRotor:
            return 0  # no rotor declared, as a lone body: body 0 is the base body
        try:  # articulated: the declared rotor joints' shared parent
            return find_rotor_joints(self.model, [rotor.joint for rotor in rotors])[3]
        except ValueError as exc:
            logger.debug(f"no rotor joint in the model: {exc}")
            return 0  # single body, no rotor joints: body 0 is the base body

    def clear_forces(self, state) -> None:
        state.clear_forces()

    def step(self, state, dt):
        """Step the vehicle's motors into ``control.joint_f``, then the solver, which consumes ``joint_f`` and
        the force elements' ``body_f`` in one solve.
        """
        self.motors.step(state, self.control, dt)
        contacts = self.model.collide(state) if self.contacts_on else None
        self.solver.step(state, self.state1, self.control, contacts, dt)
        state.assign(self.state1)
        return state

    def stages(self) -> list[Stage]:
        """The ``clear``, ``step`` and ``state`` device stages: the loop runs the command and force stages
        between ``clear`` and ``step`` once per substep, and ``state`` once per tick after the last substep,
        which writes the plant's outputs, one signal per body and per joint, from the stepped state.
        """
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
            Stage("state", "device", lambda tick: self.outputs.write(tick.state), writes=self.outputs.signals),
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
        for d, omega in self._hover_omega.items():
            jqd[d] = omega
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
