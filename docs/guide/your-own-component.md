---
description: "Ship a component of your own: its schema plugin, its class on the one contract, its registration, its peer and its peer's fake, with PX4 as the worked example."
---

# Bringing your own component

A project beside nexus adds a component the way the framework adds its own. It ships three things. A
schema declares the component in a vehicle's Universal Scene Description (USD) file. A class fills
the one component contract. An entry maps the schema to the class. Nothing in nexus changes,
and the project patches nothing. PX4 is the worked example throughout. It's the one controller nexus
ships, and it lives in its own folders as a project's component would.

## The schema

A component is one applied API schema on the prim it belongs to. The schema comes from a USD schema
plugin with no compiled code: a `plugInfo.json` and a `generatedSchema.usda` at the root of your
package, as [Newton's schema package](https://github.com/newton-physics/newton) ships its own. The
attributes of the schema, in the `nexus:` namespace, are the keyword arguments of your class. The
schema includes its [role](../design/concepts.md#role) schema as a built-in, `NexusControllerRoleAPI`
for a controller, so the builder places the component by its role and never by the methods of its
class. The [schema reference](../reference/schemas.md) lists the shipped schemas and their
attributes, and `nexus_sim/_src/usd/generatedSchema.usda` is the file to copy from.

`NexusPx4API` is PX4's. It applies to a `Scope` under the vehicle's root prim, states the controller
role, and authors `nexus:airframe`, the one attribute its class takes from the vehicle.

## The class

Every role fills the same contract, [`Component`](../reference/api/components.md). Subclass it, state
your per-tick work in `stages()` as a list of `Stage`, and override the lifecycle members you have.
Each stage declares the `Signal`s it reads and writes. A device stage joins the captured CUDA graph,
and a host stage runs between replays. The loop wires the signals before any stage runs, so a reader
takes what a writer of the same name writes. A pair that disagrees on the type or the shape stops the
run before it starts.

```python
import numpy as np
import warp as wp

from nexus_sim import Component, ImuSample, Signal, Stage


class HoverController(Component):
    """Reads the IMU and writes one command per rotor on the host."""

    def __init__(self, *, throttle: float = 0.5):
        self.throttle = throttle
        self.imu = Signal("imu", ImuSample, shape=(1,))
        self.controls = Signal("controls", wp.float32, shape=(1, 4))

    def stages(self) -> list[Stage]:
        return [Stage("act", "host", self._act, reads=(self.imu,), writes=(self.controls,))]

    def _act(self, tick) -> None:
        self.controls.write(np.full((1, 4), self.throttle, dtype=np.float32))
```

The keyword arguments are the attributes of the schema, `nexus:throttle` here. The `controls` signal
of a controller is as wide as its actuators, and the command elements read the leading part of it.

The contract has six more members. `connect` opens what the component needs for the run, a port or
a model, and `close` releases it. `attached` reads false until the peer the component waits on has
dialed in, so the loop connects it after the capture and holds the sim clock until then. `requires`
names the peers the class needs, which the builder starts. `artifacts` names the files the component
leaves for a reader. `set_logger` takes the Logger the loop hands over, scoped to the component's
path, for a value with no fixed width such as a camera's frame. The Recorder keeps every fixed-width
signal itself, and a script reads it by path, `sim["vehicle/controllers/<name>/controls"]`.

`Px4MavlinkController`, PX4's controller, is the shape to copy for a component with a peer. It binds
the Hardware In The Loop (HIL) port in `connect` and reads `attached` from the link. One host stage
per tick exchanges the samples of the sensors for the controls of PX4. It names its `ULog` in
`artifacts` and closes the socket in `close`. A sensor's class takes the values of the run, where it
rides and the site it flies at, as its first argument. That argument's shape is changing, so a sensor
of your own waits for the next release.

## The registration

Installing your package registers it. One entry per schema in the `nexus.registry` group of your
`pyproject.toml` names the class by import path. `import nexus_sim` then registers your schema plugin
with OpenUSD, and the [registry](../design/concepts.md#registry) resolves the schema to your class:

```toml
[project.entry-points."nexus.registry"]
HoverAPI = "my_project.hover:HoverController"
```

A class a script defines rather than installs goes in through `register("HoverAPI", HoverController)`.
A test maps a schema to a stand-in without touching the table of the install. It hands the run a
registry of its own, `Sim("my_quad", scene="empty", registry=Registry({"HoverAPI": StandIn}))`. The
rows of that registry answer first, and the install answers the rest.

## The peer and its fake

A [peer](../design/concepts.md#peer) is a process outside the process of the loop, which the run
starts and stops: a container, a virtual machine, or a bare process. It fills the
[`Peer`](../reference/api/components.md) contract. `start` blocks until the process runs, or raises
naming why. `stop` is idempotent. `alive` says whether the process still runs, and `artifacts` names
the files it leaves. The run owns every address a peer uses and hands the peer its ports, so two runs
on one machine never collide. A link that leaves the run shows in `sim.ports`, and the death of a
peer reaches the loop through its link.

Every peer ships a fake: a class on the same contract that speaks the links of the peer in-process
and starts no process. A test flies the component against it in seconds. PX4's is `Px4Fake`,
which answers the HIL link with a fixed command. A test hands the row of the fake to the run through
the registry, as it hands a stand-in component.

Today the builder starts the two peers nexus ships by name: PX4 Software In The Loop (SITL) and the
Kit render peer. A peer of your own that a vehicle declares waits for the builder to start every
declared peer through the registry, the next step on the roadmap.

## What an integration exports

A project that ships a component also ships the names a script needs to command it, in a namespace
of its own. PX4's is `nexus_sim.px4`. It holds `OffboardClient`, the client a script opens on the
offboard link, and `Plan`, `MissionItem` and `read_plan`, the mission a script uploads, with the
`NAV_*` and `FRAME_*` constants a mission item names. Each lives in PX4's own folder, and the
namespace only re-exports them, so the generic surface, `nexus_sim`, names nothing of PX4. That list
is the whole of what an integration exposes. Its classes stay where they live, and a script imports
the few names it spells.
