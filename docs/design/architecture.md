---
description: "How nexus works: a fixed-order deterministic sim loop of typed, replaceable components. The data model, interfaces, the estimator, guidance and controllers, actuators, observability, configuration, and packaging."
---

# Architecture

The framework is an **imperative, fixed-order, deterministic simulation loop** in which every concern is
a replaceable [component](concepts.md#component) behind a typed interface. The
[loop](concepts.md#loop), the class [`Orchestrator`](../reference/api/core.md), steps the
components in a fixed order each [tick](concepts.md#tick), and [`Sim`](concepts.md#sim) builds
and drives it.

## The simulation loop

Each tick runs the same fixed sequence, and the order is what makes [runs](concepts.md#run) reproducible:

```
t = clock.advance()
[sensor stages]                           # each sensor's sample, a signal of its own type; a camera's host stage
[estimator stages]                        # the estimate: the base body's pose and twist; none on PX4
[guidance stage]                          # the setpoint, from the mission and the estimate; none on PX4
[controller stages]                       # PX4: truth + exchange host stages; PID: one device stage
[clear → command elements → force elements → step] × substeps
                                          # command buffer → Newton's control inputs; state → per-body
                                          # forces; step: Newton's actuators, then the solver
[record]                                  # the recorder's device taps
```

Every component states its [stages](concepts.md#stage), and the loop cuts the
[ring](concepts.md#ring) at the host stages, so each [segment](concepts.md#segment) replays as
one CUDA graph. See [Execution](execution.md#stages-and-segments).

The **site is resolved once at build**, not sampled per tick: the scene's geodetic origin gives the
magnetic field, the air pressure and temperature, and gravity, and the sensors that read them take
them as constructor arguments, the same way every other sensor parameter arrives. One gravity value
reaches both the physics and the Inertial Measurement Unit (IMU). A **camera is a sensor**: each frame
the Kit peer renders reaches its reader as a signal, an `Image`, so a vision estimator or controller
reads frames as it reads any sensor.

The framework uses a fixed-order loop. It doesn't use a message bus in the style of ProjectAirSim or
Robot Operating System (ROS), nor a data-flow or Entity Component System (ECS) scheduler in the
style of Isaac. The loop gives replaceable typed interfaces and *easy* determinism at the lowest
complexity, and the typed boundaries don't prevent publishing onto a bus later for distributed
multi-vehicle work.

## Shared data model

A typed vocabulary flows between components as [signals](concepts.md#signal). A value on the device
takes a Warp struct that core declares, or a Warp value type such as `wp.float32`, and a value on the
host takes a plain class. Body frame is **Forward Right Down (FRD)**, see [Conventions](conventions.md).
All hot-loop state lives in persistent, in-place device buffers with static shapes, so the device region
can be [CUDA-graph-captured](execution.md).

| Type | Carries |
|---|---|
| `newton.State` | pose, velocity, body rates, per-body forces: the live Newton state in `body_q`, `body_qd`, and `body_f` |
| `Controls` | one normalized command per actuator, what the controller emits |
| `PoseTwist` | the estimate: the base body's pose and twist in world axes, what the estimator writes and the guidance and the controller read |
| `Wrench` | a **per-body** spatial force over the whole articulation, the base body plus each actuator's body, realized as the shared `body_f` buffer, not a single body-level force and torque pair |
| `ImuSample`, `MagSample`, `BaroSample`, `GpsSample` | each analytic sensor's sample, a Warp struct whose first field is the sim time of the tick that took it, which the sensor writes as its signal, `imu`, `mag`, `baro` or `gps` |
| `Image`, `PointCloud` | a camera's frame and a lidar's scan, each with the sim time it shows, which the sensor writes as its host signal, `camera`, `thermal_camera` or `lidar` |
| `SimTime` | sim-time and step index |

## Component interfaces

Each component fills one [role](concepts.md#role), and each role's contract is a narrow, typed
`Protocol`, light on side effects, so each component is independently testable and
fault-wrappable. Every component states its work as [stages](concepts.md#stage) over the
[tick](concepts.md#tick), and each stage declares the [signals](concepts.md#signal) it reads and
writes. The fixed parts, the clock, the [physics](concepts.md#physics), the
[Recorder](concepts.md#recorder) and the [Logger](concepts.md#logger), keep contracts of their own,
such as `Clock` and `Physics`, and the Kit render [peer](concepts.md#peer)'s lifecycle rides the `Renderer` contract.

**A sensor's output is a signal of its own type.** No sensor fills a shared bundle, and no reader holds
a sensor: an estimator or a controller declares the signals it reads, and the loop wires them. Of two
sensors of one kind, such as two IMUs, a reader picks one through a [connection](concepts.md#connection)
on its prim, or reads both as a list. [Execution](execution.md#signals) shows how a vehicle file
authors one.

The **scene and the vehicle aren't code interfaces**: they come from
Universal Scene Description (USD). A single `VehicleUsd` reads the vehicle model through
`ModelBuilder.add_usd`. There is no hand-written vehicle model to swap.

**Reading state.** Components never call an engine API directly. They read the live
`newton.State`, `body_q` as a transform and `body_qd` as a spatial vector, through Warp kernels.
Those kernels pin one canonical convention: **`XYZW` quaternions, world-frame velocity taken at the
center of mass, Z-up in sim**. Because every run is tensors-over-Newton-over-USD, the same sensor,
rotor chain, or controller runs verbatim in a CI test and in a rendered flight. The only place that
reorders a quaternion is the PX4 IMU wire, which expects scalar-first `WXYZ`.

**Dependency injection.** The **core injects** the cross-cutting handles a component needs: its
`Recorder`, a seeded Random Number Generator (RNG) sub-stream, the `SeedTree`, and the logger. So a
component holds no global state, and a test can construct it standalone. Module boundaries form a
Directed Acyclic Graph (DAG) with the core at the root, and `import-linter` enforces them in CI. No
module imports Kit: the Kit render peer's program ships as package data that nothing imports, and
the contract forbids `omni`, `usdrt`, `isaacsim` and `carb` in every tier.

## Estimator, guidance, and controllers

Every controller states its stages, and a peer's autopilot and a device-native law fly through the
same loop:

- **`Px4MavlinkController`**: runs the MAVLink lockstep handshake against a real PX4 Software In
  The Loop (SITL) instance, its peer. It reads each sensor's sample as a signal and encodes the samples
  in `HIL_SENSOR` and `HIL_GPS`, and the base body's true state in `HIL_STATE_QUATERNION`, then
  **blocks** for the returned actuator commands. `HIL_SENSOR` goes out every tick and marks a sensor's
  fields updated, in `fields_updated`, only on a tick that brings a new sample of it, as PX4's Gazebo
  Classic bridge does. `HIL_GPS` goes out with each new sample of the Global Positioning System (GPS)
  receiver. So PX4 receives each sensor at the rate its schema declares. Of two sensors of one kind, a
  connection on its prim, such as `nexus:inputs:imu`, picks the one PX4 receives. Its work is two
  **host stages**, `truth` and `exchange`, outside graph capture and not differentiable, and a lost
  connection ends the run.
- **`PidController`**: the device-native, differentiable built-in, a
  PID controller whose gains are Warp arrays with gradients, so
  it doubles as the [design-optimization](../examples/design-optimization.md) parameter set and the
  [determinism authority](execution.md#determinism).
- **`TrainedPolicyController`**: loads a policy exported from `nexus-rl` and drives the
  vehicle from an observation it builds from the estimate.

The [estimator](concepts.md#role) is the first block of the control cascade. Its stages run each tick
after the sensors' and before the guidance's. They write the estimate, a
[signal](concepts.md#signal) of the type `PoseTwist`: the base body's position, its quaternion, and its
linear and angular velocity, in world axes. The guidance and each controller that reads the vehicle's
state read the estimate, never the physics state, so a controller written against it flies behind any
estimator. The estimator belongs to the vehicle's flight stack, beside its controller, so the run
doesn't name it: each example's assembly constructs it today. One estimator ships, `GroundTruthEstimator`,
a passthrough that hands on the base body's true pose and twist with no noise and no delay. Its stage
also runs once over the settled state before the first tick, so the guidance's first stage reads an
estimate. A run whose guidance or controller reads the estimate, and that has no estimator, fails
before any stage runs, with an error that names the readers and the estimator.

**What to fly is separate from how it flies.** A controller that takes setpoints flies the mission
of its **guidance**, the outer loop of the control cascade, reached through
[`sim.guidance`](../reference/api/guidance.md). The guidance is a component of the loop, and it
holds no other component. Its stage runs each tick after the estimator's and before the controller's,
and it reads the vehicle's position from the estimate.
It writes the setpoint, a [signal](execution.md#signals) of one setpoint type, a position goal as one
`wp.vec3` or a `ReferenceTrajectory`, and the controller reads it on that same tick. The loop checks before any
stage runs that the controller reads the type the guidance writes. When the mission is over, the
stage marks the tick done, and the loop ends the run. `MissionGuidance` sequences
position goals and advances on arrival. `TrackingGuidance` plans one reference for a tracking
controller, with a `ruckig` or min-snap planner. A flight constructs its guidance from the
guidance's own parameters and hands it to `Sim.from_orchestrator`.

PX4 takes no guidance, because its own navigator sequences its missions, and no setpoint from the
loop. It takes no estimator either, because its own estimator runs in its peer and takes the
sensors' values over the lockstep link. A script commands it over its offboard [link](concepts.md#link), a MAVLink link of its own beside the lockstep
link. The run owns that link's address and names it in its [port map](concepts.md#port-map), and the script opens its own
client there: [`nexus_sim.px4.OffboardClient`](../reference/api/px4.md) on `sim.ports["offboard"]`.
[Conventions](conventions.md#who-owns-an-address) states the rule.

## Actuators

The rotor chain has three parts, split along NVIDIA Newton's model, so nothing in nexus overlaps
Newton's actuator.

- The rotors' **[command element](concepts.md#role)** scales each command to a rotor-speed target,
  the job of an Electronic Speed Controller (ESC) reduced to one multiply, and adds a drag
  feedforward.
- The motor is **Newton's actuator**, a `NewtonActuator` prim authored in the vehicle USD: a velocity
  servo under a torque-speed envelope on the real rotor joint. Physics steps every Newton actuator
  the vehicle declares, rotor motor or not, before its solver. So the rotor speed is a
  solver-integrated state with physical lag and saturation.
- The propellers' **[force element](concepts.md#role)** is the airflow-aware closed form that turns
  rotor speed and inflow into thrust and in-plane force on the rotor body.

The propeller's parameters, `ct`, `cd` and the aero terms, come from the `NexusPropellerAPI` [schema](concepts.md#schema)
that each rotor's rigid body applies in the vehicle USD. The rotor speed at full command is the
motor's no-load speed, `newton:velocityLimit`. The **mixer**, the
Collective Thrust and Body Rates (CTBR) rate loop and the `B⁻¹` control allocation, lives in the
*controllers*, not the rotor chain. So `Controls.command` is always one entry per actuator, and a
command element only ever applies the forward map. PX4 and the acados example fly this chain. The
PID, policy and sampling Model Predictive Control (MPC) examples fly a single-body plant with a
motor lag of their own.

## Observability and recording

Observability is cross-cutting, split into a **write** side and a **read** side:

- **One central Rerun sink.** A single [Logger](concepts.md#logger), handed to every component,
  owns the one Rerun recording. Each row's entity path names its process, its component and its instance, such
  as `sim/vehicle/sensors/imu` or `sim/guidance/reference`, on a shared `sim_time` timeline. The
  [logging reference](../reference/api/logging.md#entity-paths) states the rule. It can serve a live
  viewer over gRPC on port `9876` or write a durable `.rrd`. A peer writes under its own root, the
  name of its folder, so one recording can hold more than one producer. Today the camera sensors log the
  Kit render peer's frames on the host, under `sim/`. Logging is **output-only**: nothing reads it back
  into the loop, so it can't perturb determinism. Nothing logs per tick. The Logger writes the
  Recorder's histories in blocks. A component logs live only a value with no fixed width, such as a
  camera's frame or a guidance's markers.
- **The [Recorder](concepts.md#recorder).** The Recorder keeps a [history](concepts.md#history) of
  every body, every joint and every sensor's output: body poses and velocities as `BodyState` and
  `JointState`, and sensor outputs as `SensorSample`. A device kernel writes each tick's row into a
  small staging buffer. The buffer drains onto host blocks that grow with the run, so device memory
  stays fixed and the Recorder drops no row. A caller reads a history on demand through
  [`sim.physics`](../reference/api/simulation.md) and `sim.sensors`. A history's key is its
  instance's path below the process root: `vehicle/body/…`, `vehicle/joints/…` and
  `vehicle/sensors/…`. The instance has the same path in the recording, so the Recorder and the
  recording use one name for one thing. This is the capture-safe way to read a run without a host
  round-trip each tick. Each time the staging buffers drain, the Logger writes the block as
  time-series entities at `sim/<key>/series/<field>`, and the scene's poses with it. So you can
  inspect the whole history in the viewer's debug tabs.

Each PX4 run also produces PX4's native `.ulg` flight log alongside the `.rrd`, both surfaced as run
artifacts.

## Configuration and vehicles

A typed [`LaunchConfig`](../reference/api/configuration.md), resolved against the
[catalog](concepts.md#catalog), describes a run. A launch names the vehicle variant it flies and
its scene. Resolution maps the variant to its Universal Scene Description (USD), which
declares its controller: PX4, through the `NexusPx4API` schema and its airframe. It emits a fully specified, sha-pinned **tested-configuration receipt**, so a test records
exactly what it simulated. Vehicle assets are content-addressed. The
[publish pipeline](conventions.md) converts a USD to a preview `.glb` and uploads both.

## Packaging

The framework ships as a single **`nexus_sim`** package in one `uv` project. All source lives under
`nexus_sim/_src/<area>/` and the public API is re-exported from `nexus_sim`. Never import from
`_src`. Extras isolate the heavy optional dependencies, such as `policy` for Torch and `acados` for
the Nonlinear Model Predictive Control (NMPC) example, rather than separate packages. Neither peer is an extra. The Kit render peer's program
ships in the wheel as package data and mounts into NVIDIA's Isaac Sim image, which each machine
pulls on its first RTX run. The PX4 peer's pin and `px4-sitl` Dockerfile ship the same way,
and each machine builds the `px4-sitl` image and the pinned PX4 tree on its first PX4 run. The RL
trainer is a **separate
`uv` project**, `nexus-rl`, depending on the framework via an editable
path, so the prerelease Isaac Lab stack stays out of the core environment.
