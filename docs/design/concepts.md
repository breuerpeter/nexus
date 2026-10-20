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
override layer over the vehicle and the scene.
It records what it flew in its receipt, a `TestedConfig`.

### Sim

`Sim` is the one public entry to a run. It builds a run from its arguments, the vehicle, the scene,
the catalog, the override layer and the registry, and drives the loop on the caller's thread, one
tick per `step`. A script reads the run through it: any recorded [history](#history) by indexing the
run with the history's path, `sim["vehicle/sensors/imu/imu"]`, a [signal](#signal)'s live value by the
same path, `sim.read(path)`, the sim time through `sim.time`, and the port map through `sim.ports`.

### Catalog

The list of named vehicles and scenes a run resolves by name, the class `Catalog`. Each entry pins
a content-addressed USD file by its `url` and `sha256`. A run loads catalogs in layers: the one
bundled in the wheel, `nexus_uav/_src/config/catalog.yaml`, then the catalog of each installed
package that adds one, then a project's `nexus.catalog.yaml` or the file `--catalog` names. Each
extends the layers below it. [Your own assets](../guide/your-own-assets.md) says how a run finds them.

### Builder

The code that turns a launch into a run, `build_from_launch`. It resolves the launch against the
catalog and fetches its assets. It builds the physics, and each component the vehicle's USD
declares through the [registry](#registry). It starts the run's peers and hands the loop its parts.

### Loop

The fixed-order loop that runs a run's ticks, the class `Loop`. It lays the components'
stages out in its ring, captures the segments on CUDA, runs ticks until the run ends, and tears the
run down.

## The tick

### Tick

One control step of the loop: each stage in the ring runs once, and the physics steps its
substeps. `Tick` is the loop's own context, which it hands every stage: the time and the physics
state. A value one component passes to another, such as a sensor's sample, is a [signal](#signal),
not a field of the tick. The sim time is a whole number of nanoseconds, the tick count times the tick,
and the first tick is step one. Every row a tick records describes the state at the tick's time.

### Start

The ring without the clock's stage and the physics substeps. The loop runs it once over the settled
state at sim time zero, ahead of any capture. It records the first row, and the first tick's step applies
the controls it computes. A sensor's host stage stays out, and so does each host stage of a component
that waits on a [peer](#peer). While that peer boots, the loop holds the clock until the peer dials in,
then runs ordinary ticks until its first answer.

### Stage

One unit of a component's per-tick work, a `Stage`. A device stage is a launch over persistent
device buffers, which a CUDA graph replays. A host stage runs on the host between graph replays: a
peer's exchange, a solve, or an inference. The [start](#start) runs most stages once before the first
tick.

### Ring

The stages one tick runs, in the loop's fixed order. The clock's stage comes first. Once per physics
substep, `clear`, the command elements, the force elements, and `step` follow. Then `state` writes the
plant's outputs: one signal per body and one per joint. The sensors, the estimator, the guidance, and the
controller follow, over the state the step left, and the record stage comes last. `build_ring` lays the
ring out.

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
estimator reads, or a body's or joint's prim, whose pose or coordinates the plant writes, such as the
body an estimator's `pose` reads. A reader of the vehicle's pose with no connection takes the base body.
Newton's `NewtonMimicAPI` names the joint it follows the same way, with `newton:mimicJoint`.

## The parts of a run

### Component

A class in the loop's process that fills one role in the tick, a subclass of `Component`, the one
contract every role fills. A component states its per-tick work as stages, each declaring the signals
it reads and writes. It can state the rest of the contract. `connect` and `close` open and release
what it needs for the run. `set_logger` takes the scoped Logger the loop hands it. `attached` reads
false until the peer it waits on dials in. `requires` names the peers its class needs, `artifacts`
the files it leaves for a reader, and `name` its instance name. The loop, the builder and `Sim` read
those members from every component, whatever its role, and find none by name. The builder builds a
component from the schema the vehicle's USD applies.

### Role

What a component does in the tick. The role fixes where the component's stages run in the ring and
who declares it. The contract is the same for every role. The vehicle's USD declares every role. The
schema of a declared component states its role: it includes a role schema, such as
`NexusSensorRoleAPI`, as a built-in. The builder places the component by that role, never by the
methods of its class. The [schema reference](../reference/schemas.md) lists the role of each schema.

| Role | What it does |
|---|---|
| Sensor | samples the plant and writes its sample, a signal of its own type stamped with its sim time: an Inertial Measurement Unit (IMU), a Global Positioning System (GPS) receiver, a camera |
| Estimator | turns the sensors' values into the estimate, the vehicle's pose and twist that the guidance and the controller read in place of the physics state. The one that ships, `GroundTruthEstimator`, hands on the true pose and twist the plant writes for the base body, or for the body a connection on its prim names |
| Guidance | turns a mission into the setpoint a controller tracks |
| Controller | turns the estimate, or the sensors' samples, into the controls, one command per actuator |
| Command element | turns the controls into Newton's control inputs, such as the rotors' speed targets |
| Force element | adds body wrenches to the shared `body_f` buffer from the current state, such as the propellers' thrust. It adds and never assigns, so two on one body both act |

The fixed parts, the clock and the physics, keep contracts of their own, which no project implements.

### Plant side and flight stack

The plant side stands in for the real drone and its world: the physics, the command elements,
the force elements and the sensors, which read and write Newton's state in place. The flight stack
is what a real drone carries: the estimator, the guidance, and the controller, which see the plant
only through the sensors. Two parts of it read the physics state instead: the ground-truth estimator,
to stand in for a real estimator, and PX4's controller, to send PX4 its ground truth. The line between them is where real drone code stops and the simulation takes over,
[Real system ↔ simulation](real-vs-sim.md).

### Physics

The part that steps Newton's model, `NewtonPhysics`. Its `step` steps every
[actuator](#actuator) the vehicle declares, then the solver. It's a fixed part, as the clock, the
Recorder, and the Logger are: the framework's own, configured by settings, never declared.

### Actuator

Newton's motor: a `NewtonActuator` prim the vehicle's USD authors on a joint, such as a rotor's
motor on its rotor joint. [Physics](#physics) steps every actuator before its solver. An actuator
reads its joint's state and the targets a command element writes, and writes the joint's torque.
The word names Newton's motor alone, so the controls carry one command per actuator.

### Peer

A process outside the loop's process that a run starts and stops. It can be a container, a virtual
machine, or a bare process, such as PX4 Software In The Loop (SITL) or the Kit render peer. Each peer lives
in one folder under `nexus_uav/_src/peers/` and ships a fake that speaks its links.
[Conventions](conventions.md#how-a-peer-enters-a-run) says how a peer enters a run. Each peer states
how it keeps time in its `clock`, one of three ways. `stepped`: a component steps it each tick, as PX4
SITL runs on the sim time the lockstep link carries. `on_request`: it answers a request for a given sim
time, as the Kit render peer renders the frame a tick asks for. `free`: it runs on its own wall clock,
as a companion computer under an emulator does. Sim time and a free-running peer's clock agree only
while the run paces at a real-time factor of 1. So a run that declares one fails at any other factor
before any peer starts, and warns once when its live factor falls below 0.95.

### Link

The defined protocol between a component and a peer, or between two peers. The PX4 controller and
PX4 SITL share the Hardware In The Loop (HIL) link, and the RTX sensors and the Kit render peer the
render link. The offboard link leaves the run for a script. A serial link joins two peers as a serial
line joins a companion computer to the flight controller. The line is wiring on the vehicle, so the
declared peer's prim declares it, as a component's prim declares its inputs. The relationship
`nexus:serial:<line>` targets the prim of the peer at the other end. The run makes each link, two
pseudo-terminals joined, before any peer starts, and hands each of the two peers the path of its end.
The declaring peer gets it by the line's name, the other by the declaring prim's name. PX4 SITL
starts a MAVLink instance on each of its ends.

### Port map

`sim.ports`: each link that leaves the run, by name, with the address a script opens its client on,
such as `sim.ports["offboard"]`. The run owns every address,
[Conventions](conventions.md#who-owns-an-address).

## Declaration

### Schema

An applied USD API schema from the plugin in `nexus_uav/_src/usd/` that declares a component on a
prim, such as `NexusImuAPI`. Its `nexus:` attributes are the class's keyword arguments, and the
[schema reference](../reference/schemas.md) lists them.

### Site

Newton's placement point on a body. It's a massless shape of the model with the `SITE` flag, which
collides with nothing, at an offset and a rotation in the body's axes. Its label is its prim's path.
A sensor's prim is a site. Its schema includes `NewtonSiteAPI` as a built-in, so the importer loads
the geometry prim the schema sits on as one. The sensor's class reads its body and its mount from
the model's arrays, as `newton.sensors.SensorIMU` does.

### Origin

Where the scene's world origin sits on Earth: the latitude, longitude and altitude its `PhysicsScene`
authors as `nexus:geodeticOrigin`. The importer reads it into the model as a world value. The same
prim can author the magnetic field, the air pressure at sea level and the temperature beside it.
The build derives the field from the origin where the scene authors none. A sensor reads them from
the model, and an override layer composed over the scene moves the origin.

### Registry

The map from each name a run resolves to its class, `Registry`, a component's schema and a peer's
alike. The own rows of a registry answer first, and the install answers the rest: the `nexus.registry`
entry-point group of each installed package, which `register` adds to for a class a script defines. A
test hands the run a registry of its own, with its stand-ins and its fakes as rows, through
`Sim(registry=...)`.

## Recording

### Logger

The fixed part that writes a run's recording to Rerun, a `.rrd` file or a live viewer. It writes the
Recorder's histories a block at a time: each time their staging buffers drain, and once more at
teardown. A block holds every quantity's series, the base body's pose, the scene's poses and the
flown path. The loop hands each component a scoped Logger, which puts the component's path before
each row the component logs live, [Logging](../reference/api/logging.md#entity-paths).

### Recorder

The fixed part that keeps a history of what a run records each tick: each device [signal](#signal) a
component writes, the plant's bodies and joints among them. The loop hands it the signals once it
wires the ring.
No component records itself. A script reads it during and after a run, and the Logger writes it
into the recording.

### History

The rows the Recorder keeps for one signal over a run, each stamped with its sim time: `history()`
reads them oldest first as one structured array, and `history()[-1]` is the newest row. A
history sits under its writer's path then the signal's name, `vehicle/sensors/imu/imu`, and the
plant's under `vehicle/body/<label>` and `vehicle/joints/<label>`. Every history's rows read as
records of `t` then the quantities the signal's type declares. A device kernel writes each
tick's row into the history's staging buffer, a few thousand rows. Each time the buffer fills, the
rows drain onto host blocks that grow with the run. So device memory stays fixed and the Recorder
drops no row.
