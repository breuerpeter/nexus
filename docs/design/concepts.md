---
description: "Every term the framework's code and docs use, each defined once, in the words the code uses: a run and its loop, the tick with its stages and signals, components and their roles, peers and links, declaration, and recording."
---

# Concepts

This page defines each term the design pages use, once. The other pages link here instead of
defining a term again. A change that adds, renames or removes a concept updates this page in the
same pull request, and a rule in the `Project` Vale style flags each word a change retires.

## A run

### Run

One flight of one vehicle in one scene, from build to teardown. A run names its vehicle and its
scene, by catalog name or by local Universal Scene Description (USD) path, and can compose an
override layer over the vehicle.
It records what it flew in its receipt, a `TestedConfig`.

### Sim

`Sim` is the one public entry to a run. It builds a run from its arguments and drives the loop on
the caller's thread, one tick per `step`. A script reads the run through it: the Recorder's
histories through `sim.physics`, and the port map through `sim.ports`.

### Catalog

The list of named vehicles and scenes a run resolves by name, the class `Catalog`. Each entry pins
a content-addressed USD file by its `url` and `sha256`. A project's `nexus.catalog.yaml`, or the
file `--catalog` names, extends the catalog bundled in the wheel,
`nexus_sim/_src/config/catalog.yaml`. [Your own assets](../guide/your-own-assets.md)
says how a run finds one.

### Builder

The code that turns a launch into a run, `build_from_launch`. It resolves the launch against the
catalog and fetches its assets. It builds the physics, and each component the vehicle's USD
declares through the component registry. It starts the run's peers and hands the loop its parts.

### Loop

The fixed-order loop that runs a run's ticks, the class `Orchestrator`. It lays the components'
stages out in its ring, captures the segments on CUDA, runs ticks until the run ends, and tears the
run down.

## The tick

### Tick

One control step of the loop: each stage in the ring runs once, and the physics steps its
substeps. `Tick` is the loop's own context, which it hands every stage: the time and the physics
state. A value one component passes to another, such as a sensor's sample, is a [signal](#signal),
not a field of the tick.

### Stage

One unit of a component's per-tick work, a `Stage`. A device stage is a launch over persistent
device buffers, which a CUDA graph replays. A host stage runs on the host between graph replays: a
peer's exchange, a solve, or an inference. A warm stage also runs once over the settled state,
before the first tick.

### Ring

The stages one tick runs, in the loop's fixed order. The sensors come first, then the estimator, the
guidance, and the controller. Once per physics substep, `clear`, the command elements, the force elements, and
`step` follow, and the record stage comes last. `build_ring` lays the ring out.

### Segment

A maximal run of device stages between two host stages of the ring. On CUDA the loop captures each
segment as one CUDA graph and replays it every tick. With no host stage the whole ring is one
segment. [Execution](execution.md#stages-and-segments) says how the loop cuts and replays them.

### Signal

A named, typed value that one component writes and others read on the tick, a `Signal`. Each
stage declares the signals it reads and writes. Before any stage runs, the loop wires each input
to an output of its name and hands the writer and each reader one buffer. Where more than one
component writes a signal, a reader takes the one a [connection](#connection) names, and a list input
takes every writer's.
[Execution](execution.md#signals) says where a signal's buffer lives and what the loop checks.

### Connection

A relationship `nexus:inputs:<signal>` on a reader's prim, which the reader's schema declares. Its
target is the prim of the component the reader takes that signal from, such as the one of two IMUs the
estimator reads. Newton's `NewtonMimicAPI` names the joint it follows the same way, with
`newton:mimicJoint`.

## The parts of a run

### Component

A class in the loop's process that fills one role in the tick. A component states its per-tick work
as stages. The builder builds it from the schema the vehicle's USD applies, or takes it as a run
argument, as it takes a guidance.

### Role

What a component does in the tick. The role fixes where the component's stages run in the ring and
who declares it. The vehicle's USD declares every role but the guidance, which the run takes as an
argument. The schema of a declared component states its role: it includes a role schema, such as
`NexusSensorRoleAPI`, as a built-in. The builder places the component by that role, never by the
methods of its class. The [schema reference](../reference/schemas.md) lists the role of each schema.

| Role | What it does |
|---|---|
| Sensor | samples the plant and writes its sample, a signal of its own type stamped with its sim time: an Inertial Measurement Unit (IMU), a Global Positioning System (GPS) receiver, a camera |
| Estimator | turns the sensors' values into the estimate, the vehicle's pose and twist that the guidance and the controller read in place of the physics state. The one that ships, `GroundTruthEstimator`, hands on the base body's true pose and twist |
| Guidance | turns a mission into the setpoint a controller tracks |
| Controller | turns the estimate, or the sensors' samples, into the controls, one command per actuator |
| Command element | turns the controls into Newton's control inputs, such as the rotors' speed targets |
| Force element | adds body wrenches to the shared `body_f` buffer from the current state, such as the propellers' thrust. It adds and never assigns, so two on one body both act |

Each role's contract is a `Protocol` in `nexus_sim/_src/core/interfaces.py`.

### Plant side and flight stack

The plant side stands in for the real drone and its world: the physics, the command elements,
the force elements and the sensors, which read and write Newton's state in place. The flight stack
is what a real drone carries: the estimator, the guidance, and the controller, which see the plant
only through the sensors. Two parts of it read the physics state instead: the ground-truth estimator,
to stand in for a real estimator, and PX4's controller, to send PX4 its ground truth. The line between them is where real drone code stops and the simulation takes over,
[Real system ↔ simulation](real-vs-sim.md).

### Physics

The part that steps NVIDIA Newton's model, `NewtonPhysics`. Its `step` steps every Newton actuator
the vehicle declares, then the solver. It's a fixed part, as the clock, the Recorder, and the
Logger are: the framework's own, configured by settings, never declared.

### Peer

A process outside the loop's process that a run starts and stops. It can be a container, a virtual
machine, or a bare process, such as PX4 Software In The Loop (SITL) or the Kit render peer. Each peer lives
in one folder under `nexus_sim/_src/peers/` and ships a fake that speaks its links.
[Conventions](conventions.md#how-a-peer-enters-a-run) says how a peer enters a run.

### Link

The defined protocol between a component and a peer, or between two peers. The PX4 controller and
PX4 SITL share the Hardware In The Loop (HIL) link, and the RTX sensors and the Kit render peer the
render link. The offboard link leaves the run for a script.

### Port map

`sim.ports`: each link that leaves the run, by name, with the address a script opens its client on,
such as `sim.ports["offboard"]`. The run owns every address,
[Conventions](conventions.md#who-owns-an-address).

## Declaration

### Schema

An applied USD API schema from the plugin in `nexus_sim/_src/usd/` that declares a component on a
prim, such as `NexusImuAPI`. Its `nexus:` attributes are the class's keyword arguments, and the
[schema reference](../reference/schemas.md) lists them.

### Component registry

The map from schema to class, `ComponentRegistry`. The default one reads the `nexus.components`
entry-point group, and a test hands the builder one of its own.

## Recording

### Logger

The fixed part that writes a run's recording to Rerun, a `.rrd` file or a live viewer. The loop
hands each component a scoped Logger, which puts the component's path before each row it logs,
[Logging](../reference/api/logging.md#entity-paths).

### Recorder

The fixed part that keeps a history of what a run records each tick: each body, each joint, and each
sensor's output, on the device. A script reads it during and after a run, and at teardown its
histories go into the recording.

### History

The rows the Recorder keeps for one body, joint or sensor, each stamped with its sim time: `history()`
reads them oldest first, and `latest()` reads the newest.
