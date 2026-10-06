---
description: "How nexus works: a fixed-order deterministic sim loop of typed, replaceable components. The data model, interfaces, controller and operator plane, actuators, observability, configuration, and packaging."
---

# Architecture

nexus is an **imperative, fixed-order, deterministic simulation loop** in which every concern is
a replaceable component behind a typed interface. A central
[`Orchestrator`](../reference/api/core.md) steps the components in a fixed order each tick. The
[`Sim`](../reference/api/simulation.md) façade builds and drives it.

## The simulation loop

Each tick runs the same fixed sequence, and the order is what makes runs reproducible:

```
t = clock.advance()
[sensor stages]                           # IMU, GPS, baro, mag into device buffers; a camera's host stage
[controller stages]                       # PX4: read + exchange host stages; PID: one device stage
[clear → actuator → step] × substeps      # command buffer → per-body forces → Newton advances the dynamics
[record]                                  # the recorder's device taps
```

Every component states its stages, and the loop cuts the ring at the host stages, so each run of
device stages replays as one CUDA graph. See [Execution](execution.md#stages-and-segments).

The **site is resolved once at build**, not sampled per tick: the scene's geodetic origin gives the
magnetic field, the air pressure and temperature, and gravity, and the sensors that read them take
them as constructor arguments, the same way every other sensor parameter arrives. One gravity value
reaches both the physics and the IMU. A **camera is a sensor**: it produces an image measurement through the renderer, so vision
controllers simply read frames.

nexus uses a fixed-order loop. It doesn't use a message bus in the style of ProjectAirSim or
Robot Operating System (ROS), nor a data-flow or Entity Component System (ECS) scheduler in the
style of Isaac. The loop gives replaceable typed interfaces and *easy* determinism at the lowest
complexity, and the typed boundaries don't prevent publishing onto a bus later for distributed
multi-vehicle work.

## Shared data model

A neutral, typed vocabulary flows between components, independent of any array library. Body frame
is **Forward Right Down (FRD)**, see [Conventions](conventions.md). All hot-loop state lives in
persistent, in-place device buffers with static shapes, so the device region can be
[CUDA-graph-captured](execution.md).

| Type | Carries |
|---|---|
| `newton.State` | pose, velocity, body rates, per-body forces: the live Newton state in `body_q`, `body_qd`, and `body_f` |
| `Controls` | one normalized command per actuator, what the controller emits |
| `Wrench` | a **per-body** spatial force over the whole articulation, the base body plus each actuator's body, realized as the shared `body_f` buffer, not a single body-level force and torque pair |
| `Measurement` | per-sensor output such as Inertial Measurement Unit (IMU) and Global Positioning System (GPS) samples, plus a free-form ground-truth `observation` slot a policy reads |
| `SimTime` | sim-time and step index |

## Component interfaces

Components are narrow, typed `Protocol`s, light on side effects, so each is independently testable and
fault-wrappable:

| Interface | Responsibility |
|---|---|
| `Clock` | sim-time and step, with real-time scaling |
| `Physics` | `reset` / `step` the Newton dynamics |
| `Actuator` | one device stage, `forces_wp(cmd, state) → Wrench` from the controller's command buffer |
| `Sensor` | a device stage into its own buffer plus `read(meas) → Measurement`, or a host stage where a camera sensor uses a renderer |
| `Controller` | its stages, one contract, many implementations: a peer's `read` and `exchange` host stages, or a device-native law |
| `Renderer` | the Kit render peer's lifecycle: RTX sensors render in a container fed poses over a socket |
| `Stage`, `Tick` | the stage contract: one unit of per-tick work, and the context every stage runs over |
| `Recorder` | cross-cutting observability |

The **scene and the vehicle aren't code interfaces**: they come from
Universal Scene Description (USD). A single `USDBuilder` reads the vehicle model through
`ModelBuilder.add_usd`. There is no hand-written builder to swap.

**Reading state.** Components never call an engine API directly. They read the live
`newton.State`, `body_q` as a transform and `body_qd` as a spatial vector, through Warp kernels.
Those kernels pin one canonical convention: **`XYZW` quaternions, world-frame velocity taken at the
center of mass, Z-up in sim**. Because every run is tensors-over-Newton-over-USD, the same sensor,
actuator, or controller runs verbatim in a CI test and in a rendered flight. The only place that
reorders a quaternion is the PX4 IMU wire, which expects scalar-first `WXYZ`.

**Dependency injection.** The **core injects** the cross-cutting handles a component needs: its
`Recorder`, a seeded Random Number Generator (RNG) sub-stream, the `SeedTree`, and the logger. So a
component holds no global state, and a test can construct it standalone. Module boundaries form a
Directed Acyclic Graph (DAG) with the core at the root, and `import-linter` enforces them in CI. No
module imports Kit: the Kit render peer's program ships as package data that nothing imports, and
the contract forbids `omni`, `usdrt`, `isaacsim` and `carb` in every tier.

## Controllers and the operator plane

Every controller states its stages, and a peer's autopilot and a device-native law fly through the
same loop:

- **`Px4MavlinkController`**: runs the MAVLink lockstep handshake against a real PX4 Software In
  The Loop (SITL) instance, its peer. It encodes `HIL_SENSOR`, `HIL_GPS`, and
  `HIL_STATE_QUATERNION`, then **blocks** for the returned actuator commands. Its work is two
  **host stages**, `read` and `exchange`, outside graph capture and not differentiable, and a lost
  connection ends the run.
- **`PidController`**: the device-native, differentiable built-in, a
  Proportional Integral Derivative (PID) controller whose gains are Warp arrays with gradients, so
  it doubles as the [design-optimization](../examples/design-optimization.md) parameter set and the
  [determinism authority](execution.md#determinism).
- **`TrainedPolicyController`**: loads a policy exported from `nexus-rl` and drives the
  vehicle from the ground-truth observation.

**What to fly is separate from how it flies.** The **operator plane**, reached through
[`sim.operator`](../reference/api/operator.md), issues typed setpoints, `PositionGoal`, `Waypoints`,
or `ReferenceTrajectory`, that a controller narrows via `accept_setpoint`. `InProcessOperator` flips
the active setpoint between graph replays. A `ruckig` or min-snap planner plans jerk-limited references.

PX4 takes no setpoint from the loop. A script commands it over its offboard link, a MAVLink link
of its own beside the lockstep link. The run owns that link's address and names it in its port map,
and the script opens its own client there: [`nexus.px4.OffboardClient`](../reference/api/px4.md) on
`sim.ports["offboard"]`. [Conventions](conventions.md#who-owns-an-address) states the rule.

## Actuators

The shipped actuator is **`ArticulatedRotors`**, already the chain a real actuator is, only unnamed.
The command scales to a rotor-speed target, the job of an Electronic Speed Controller (ESC)
reduced to one multiply. The motor is a `NewtonActuator` prim authored in the vehicle USD: a velocity servo under a
torque-speed envelope that Newton solves on the real rotor joint. So the rotor speed is a
solver-integrated state with physical lag and saturation. The propeller is the airflow-aware closed
form that turns rotor speed and inflow into thrust and in-plane force on the rotor body. The
propeller's parameters, `ct`, `cd` and the aero terms, come from the `NexusPropellerAPI` schema that
each rotor's rigid body applies in the vehicle USD. The rotor speed at full command is the motor's
no-load speed, `newton:velocityLimit`. The **mixer**, the
Collective Thrust and Body Rates (CTBR) rate loop and the `B⁻¹` control allocation, lives in the
*controllers*, not the actuator. So `Controls.command` is always one entry per actuator, and the
actuator only ever applies the forward map. The same actuator drives the PX4, PID, policy, and Model
Predictive Control (MPC) paths.

## Observability and recording

Observability is cross-cutting, split into a **write** side and a **read** side:

- **One central Rerun sink.** A single `Logger`, injected into every component, owns the one
  Rerun recording. Each row's entity path names its process, its component and its instance, such
  as `sim/vehicle/sensors/imu` or `sim/guidance/reference`, on a shared `sim_time` timeline. The
  [logging reference](../reference/api/logging.md#entity-paths) states the rule. It can serve a live
  viewer over gRPC on port `9876` or write a durable `.rrd`. A peer writes under its own root, the
  name of its folder, so one recording can hold more than one producer. Today the camera sensors log the
  Kit render peer's frames on the host, under `sim/`. Logging is **output-only**: nothing reads it back
  into the loop, so it can't perturb determinism. It decimates to a configurable rate, 50 Hz by
  default, so it doesn't cap the real-time factor.
- **The Recorder read-seam.** Components record typed samples into device-side ring buffers: body
  poses and velocities as `BodyState` and `JointState`, and sensor outputs as `SensorSample`. A
  caller reads them on demand through [`sim.physics`](../reference/api/simulation.md) and
  `sim.sensors`. A channel's key is its instance's path below the process root: `vehicle/body/…`,
  `vehicle/joints/…` and `vehicle/sensors/…`. The instance has the same path in the recording, so
  the recorder and the recording use one name for one thing. This is the capture-safe way
  to observe a run without a host round-trip each tick. At the end of a recorded run the Logger
  dumps every channel's ring as time-series entities at `sim/<key>/series/<field>`. So you can inspect the
  whole observation history in the viewer's debug tabs.

Each PX4 run also produces PX4's native `.ulg` flight log alongside the `.rrd`, both surfaced as run
artifacts.

## Configuration and vehicles

A typed [`LaunchConfig`](../reference/api/configuration.md), resolved against a `Registry`,
describes a run. A launch names the vehicle variant it flies, and a launch that names none flies the
registry's default. Resolution maps the variant to its Universal Scene Description (USD), which
declares its controller: PX4, through the `NexusPx4API` schema and its airframe. It emits a fully specified, sha-pinned **tested-configuration receipt**, so a test records
exactly what it simulated. Vehicle assets are content-addressed. The
[publish pipeline](conventions.md) converts a USD to a preview `.glb` and uploads both.

## Packaging

The framework ships as a single **`nexus`** package in a `uv` workspace. All source lives under
`nexus/_src/<area>/` and the public API is re-exported from `nexus`. Never import from
`_src`. Extras isolate the heavy optional dependencies, such as `policy` for Torch and `acados` for
the Nonlinear Model Predictive Control (NMPC) example, rather than separate packages. Neither peer is an extra. The Kit render peer's program
ships in the wheel as package data and mounts into NVIDIA's Isaac Sim image, which each machine
pulls on its first RTX run. The PX4 peer's pin and `px4-sitl` Dockerfile ship the same way,
and each machine builds the `px4-sitl` image and the pinned PX4 tree on its first PX4 run. The RL
trainer is a **separate
`uv` project**, `nexus-rl`, depending on the framework via an editable
path, so the prerelease Isaac Lab stack stays out of the core environment.
