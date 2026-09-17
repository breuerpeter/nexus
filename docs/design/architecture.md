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
t        = clock.now()
env      = environment.sample(pos, t)                 # authoritative shared input
meas     = [ sensor.sample(state, env, t) … ]         # IMU, GPS, baro, mag (+ camera in the RTX runtime)
controls = controller.exchange(meas, t)               # PX4 lockstep OR in-process PID/policy
wrench   = actuator.forces(controls, state, env)      # per-actuator command → per-body forces
state    = physics.step(state, wrench, env, dt)        # Newton advances the dynamics
recorder.record(t, …);  clock.step(dt)
```

The **environment is an authoritative shared input**. Physics consumes it for aero and relative
airspeed, sensors for the magnetic field and pressure, and, in the RTX runtime, the renderer. A
**camera is a sensor**: it produces an image measurement through the renderer, so vision
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
| `EnvSample` | wind, air density, air pressure, air temperature, gravity, magnetic field, precipitation |
| `Measurement` | per-sensor output such as Inertial Measurement Unit (IMU) and Global Positioning System (GPS) samples, plus a free-form ground-truth `observation` slot a policy reads |
| `SimTime` | sim-time and step index |

## Component interfaces

Components are narrow, typed `Protocol`s, light on side effects, so each is independently testable and
fault-wrappable:

| Interface | Responsibility |
|---|---|
| `Clock` | sim-time and step, with real-time scaling |
| `Environment` | authoritative ambient fields, `sample(pos, t) → EnvSample` |
| `Physics` | `reset` / `step` the Newton dynamics |
| `Actuator` | `forces(controls, state, env) → Wrench` |
| `Sensor` | `sample(state, env, t) → Measurement`, where a camera sensor uses a renderer |
| `Controller` | `exchange(measurements, t) → Controls`, one method, many implementations |
| `Renderer` | standalone debug viewers, or in-process RTX in Isaac Sim |
| `Recorder`, `Capturable` | cross-cutting observability and capture-capability markers |

The **scene and the vehicle aren't code interfaces**: they come from
Universal Scene Description (USD). A single `USDBuilder` reads the vehicle model through
`ModelBuilder.add_usd`. There is no hand-written builder to swap.

**Reading state across runtimes.** Components never call a runtime API directly. They read the
live `newton.State`, `body_q` as a transform and `body_qd` as a spatial vector, through Warp kernels.
Those kernels pin one canonical convention: **`XYZW` quaternions, world-frame velocity taken at the
center of mass, Z-up in sim**. Because both runtimes are tensors-over-Newton-over-USD, the same
sensor, actuator, or controller runs verbatim in each. This is the mechanism behind *one component
set, two runtimes*. The only place that reorders a quaternion is the PX4 IMU wire, which expects
scalar-first `WXYZ`.

**Dependency injection.** The **core injects** the cross-cutting handles a component needs: its
`Recorder`, a seeded Random Number Generator (RNG) sub-stream, the `SeedTree`, and the logger. So a
component holds no global state, and a test can construct it standalone. Module boundaries form a
Directed Acyclic Graph (DAG) with the core at the root, and `import-linter` enforces them in CI. The
core can never import the Isaac Sim runtime.

## Controllers and the operator plane

`Controller.exchange()` unifies external and in-process autopilots behind one method:

- **`Px4MavlinkController`**: runs the MAVLink lockstep handshake against a real PX4 Software In
  The Loop (SITL) instance. It encodes `HIL_SENSOR`, `HIL_GPS`, and `HIL_STATE_QUATERNION`, then
  **blocks** for the returned actuator commands. It's a **host boundary**, outside graph capture and
  not differentiable, and a lost connection ends the run.
- **`PidController`**: the in-process, differentiable built-in, a
  Proportional Integral Derivative (PID) controller whose gains are Warp arrays with gradients, so
  it doubles as the [design-optimization](../examples/design-optimization.md) parameter set and the
  [determinism authority](execution.md#determinism).
- **`TrainedPolicyController`**: loads a policy exported from `nexus-rl` and drives the
  vehicle from the ground-truth observation.

**What to fly is separate from how it flies.** The **operator plane**, reached through
[`sim.operator`](../reference/api/operator.md), issues typed setpoints, `PositionGoal`, `Waypoints`,
or `ReferenceTrajectory`, that a controller narrows via `accept_setpoint`. `InProcessOperator` flips
the active setpoint between graph replays. `Px4Offboard` streams offboard setpoints to PX4 over a
separate MAVLink link. A `ruckig` or min-snap planner plans jerk-limited references.

## Actuators

The shipped actuator is **`ArticulatedRotors`**, already the chain a real actuator is, only unnamed.
The command scales to a rotor-speed target, the job of an Electronic Speed Controller (ESC)
reduced to one multiply. The motor is a `NewtonActuator` prim authored in the vehicle USD: a velocity servo under a
torque-speed envelope that Newton solves on the real rotor joint. So the rotor speed is a
solver-integrated state with physical lag and saturation. The propeller is the airflow-aware closed
form that turns rotor speed and inflow into thrust and in-plane force on the rotor body. The motor
and propeller parameters, `ct`, `cd`, `rpm_max` and the aero terms, come from the `motor:*` and
`propeller:*` attributes authored on the rotor joints in the vehicle USD. The **mixer**, the
Collective Thrust and Body Rates (CTBR) rate loop and the `B⁻¹` control allocation, lives in the
*controllers*, not the actuator. So `Controls.command` is always one entry per actuator, and the
actuator only ever applies the forward map. The same actuator drives the PX4, PID, policy, and Model
Predictive Control (MPC) paths.

## Observability and recording

Observability is cross-cutting, split into a **write** side and a **read** side:

- **One central Rerun sink.** A single `Logger`, injected into every component, owns the one
  Rerun recording, writing under namespaced entity paths such as `physics/…`, `sensors/…`,
  `operator/…`, and `controller/…` on a shared `sim_time` timeline. It can serve a live viewer over
  gRPC on port `9876` or write a durable `.rrd`. Out-of-process producers, PX4's own logger and the
  Isaac Sim runtime, merge into the same view. Logging is **output-only**: nothing reads it back
  into the loop, so it can't perturb determinism. It decimates to a configurable rate, 50 Hz by
  default, so it doesn't cap the real-time factor.
- **The Recorder read-seam.** Components record typed samples into device-side ring buffers: body
  poses and velocities as `BodyState` and `JointState`, and sensor outputs as `SensorSample`. A
  caller reads them on demand through [`sim.physics`](../reference/api/simulation.md) and
  `sim.sensors`. That component-kind access mirrors the channel keys `physics/body/…`,
  `physics/joint/…`, and `sensors/…`. Those keys are the recorder's registry namespace, distinct
  from the similarly worded Rerun entity paths in the preceding item. This is the capture-safe way
  to observe a run without a host round-trip each tick. At the end of a recorded run the Logger
  dumps every channel's ring as time-series entities under `recording/…`. So you can inspect the
  whole observation history in the viewer's debug tabs.

Each PX4 run also produces PX4's native `.ulg` flight log alongside the `.rrd`, both surfaced as run
artifacts.

## Configuration and vehicles

A typed [`LaunchConfig`](../reference/api/configuration.md), resolved against a `Registry`,
describes a run. A launch names the vehicle variant it flies, and a launch that names none flies the
registry's default. Resolution maps the variant to its Universal Scene Description (USD) and its PX4
airframe, and emits a fully specified, sha-pinned **tested-configuration receipt**, so a test records
exactly what it simulated. Vehicle assets are content-addressed. The
[publish pipeline](conventions.md) converts a USD to a preview `.glb` and uploads both.

## Packaging

The framework ships as a single **`nexus`** package in a `uv` workspace. All source lives under
`nexus/_src/<area>/` and the public API is re-exported from `nexus`. Never import from
`_src`. Extras isolate the heavy optional dependencies, `policy` for Torch, `upload` for S3, and
`isaacsim` for the RTX runtime, rather than separate packages. The RL trainer is a **separate
`uv` project**, `nexus-rl`, depending on the framework via an editable
path, so the prerelease Isaac Lab stack stays out of the core environment.
