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
attributes, and `nexus_uav/_src/usd/generatedSchema.usda` is the file to copy from.

`NexusPx4API_1` is PX4's. It applies to a `Scope` under the vehicle's root prim, states the controller
role, and authors no attribute. The airframe PX4 boots belongs to its peer's schema,
`NexusPx4SitlAPI`, on a `Scope` of its own.

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

from nexus_uav import Component, ImuSample, Signal, Stage


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

The contract has eight more members. `connect` opens what the component needs for the run, a port or
a model, and takes the link of each peer the class requires, keyed by the peer's name. `close`
releases what `connect` opened. `attached` reads false until the peer the component waits on has
dialed in, so the loop connects it after the capture and holds the sim clock until then. `requires`
names the peers the class needs, which the builder starts. `needs_instance` asks for the run's
instance, the number the run's peers number their ports from, as the keyword `instance`. `ports` names
the links the component knows of that leave the run, which `sim.ports` lists. `artifacts` names the
files the component leaves for a reader. `set_logger` takes the Logger the loop hands over, scoped to the component's
path, for a value with no fixed width such as a camera's frame. The Recorder keeps every fixed-width
signal itself, and a script reads it by path, `sim["vehicle/controllers/<name>/controls"]`.

`Px4MavlinkController`, PX4's controller, is the shape to copy for a component with a peer. It binds
the Hardware In The Loop (HIL) port in `connect` and reads `attached` from the link. One host stage
per tick exchanges the samples of the sensors for the controls of PX4. It names its `ULog` in
`artifacts` and closes the socket in `close`.

A sensor's class takes the model and its [site](../design/concepts.md#site), the path of the prim
that declares it. Then come the attributes of its schema as keywords, and `seed`, the noise seed the
builder derives from the run's seed and the prim. It reads the body it rides, its mount and the world's
values from the model, as Newton's own sensors do. A sensor whose class requires a peer, as the RTX
cameras require the Kit render peer, takes the peer's link in `connect`.

```python
class RangeSensor(Component):
    def __init__(self, model, site: str, *, seed: int = 0, noise: float = 0.01, rate: float = 0.0): ...
```

## The registration

Installing your package registers it. One entry per schema in the `nexus.registry` group of your
`pyproject.toml` names the class by import path. `import nexus_uav` then registers your schema plugin
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

A vehicle declares a peer with a schema of its own, on a `Scope` of its own under the root prim.
The registry maps the schema to the peer's class, as it maps a component's. The schema states no
role: its class is a peer, and so the build sorts it from a component. Its attributes are
the keyword arguments of the class, as `nexus:airframe` is the PX4 peer's. The run's values join
them as keywords: `pin`, `instance`, `log_dir`, `files`, the USD files the run reads, and
`cache_dir`. A peer a class requires, as an RTX camera requires `kit`, the registry maps by that
name. The class states its `name`, the folder name its pin file takes, and `check_pin`, which fails
a pin not in its form and returns the pin it checked. It states `clock`, how it keeps time: `stepped`,
`on_request` or `free`, per [Peer](../design/concepts.md#peer). A run that declares a free-running
peer needs `--rtf 1`, and fails before any peer starts without it. It can state `instance_free`, which says
whether an instance is free for it, and `ports`, the links it knows of that leave the run. A peer that
talks to another peer over a serial line, as a companion computer talks to PX4, declares the line in
its schema as a relationship `nexus:serial:<line>`. The vehicle targets the prim of the peer at the
other end. The run then hands it the path of its end in the keyword `serial`, by the line's name, per
[Link](../design/concepts.md#link). A peer that takes a component's output, as a companion computer
takes a camera's frames over a USB cable, declares each input in its schema as a relationship
`nexus:inputs:<signal>`. The vehicle targets one or more prims whose components write that signal.
The loop hands the peer each writer's value once per tick, after the writers, through
`take(signal, writer, value)`, with the writer's instance name. The peer's `take` is its end of the
link, and the peer decides what it makes of a value and how it carries it. A frame stays on its signal
until the next, so the peer sends one once, by its time. A `take` that raises fails the run naming
the peer. A run's override layer drops a declared peer by deactivating its scope, `active = false`.

Every peer ships a fake: a class on the same contract that speaks the links of the peer in-process
and starts no process. A test flies the component against it in seconds. PX4's is `Px4Fake`,
which answers the HIL link with a fixed command. A fake of a peer with an input offers the same `take`
and drops what it takes. A test hands the row of the fake to the run through the registry, as it hands
a stand-in component.

Every peer pins what it runs, a commit or an image, in `pin.ref` in its own folder, with the pin on
line one. A project overrides it with `nexus.<peer>.ref` beside its catalog, in the same form, and
the variable `NEXUS_<PEER>_REF` overrides both, where `<peer>` is the folder name of the peer. The run's
receipt records each pin and where it came from. A peer of your own follows the same rule: its
package ships `nexus.<peer>.ref` beside its catalog. A peer that no source pins runs with none.

A peer whose process is one container subclasses
[`ContainerPeer`](../reference/api/components.md), as the shipped peers do. The subclass states
its `name`, the peer's kind, and `run_options`, what its container runs: the image, the command and
the other keywords of the docker SDK's `containers.run`. The class does the rest. It removes a
leftover container of the peer's name, then runs the container under that name with the label
`nexus.peer=<name>`. It streams the console into `<name>/console.log` in the run's folder. It
removes the container at `stop`, and reads `alive` from the daemon. A peer that ships its own build
context builds its image with `ensure_image`. That tags the image by a hash of the folder, so it
builds once per machine and again when the folder changes.

```python
from pathlib import Path

from nexus_uav import ContainerPeer, ensure_image

IMAGE_DIR = Path(__file__).with_name("image")


class Radio(ContainerPeer):
    name = "radio"

    def __init__(self, *, instance, log_dir, **run):
        super().__init__(instance=instance, log_dir=log_dir)

    def run_options(self):
        return {"image": ensure_image(IMAGE_DIR, "radio"), "command": ["radio", "--serve"]}
```

Every peer's files live under a folder named for it in the run's folder, its console as
`console.log` there. PX4's are under `px4_sitl/`, its `ULog` beside the console, and Kit's under
`kit/`.

## What an integration exports

A project that ships a component also ships the names a script needs to command it, in a namespace
of its own. PX4's is `nexus_uav.px4`. It holds `OffboardClient`, the client a script opens on the
offboard link, and `Plan`, `MissionItem` and `read_plan`, the mission a script uploads, with the
`NAV_*` and `FRAME_*` constants a mission item names. Each lives in PX4's own folder, and the
namespace only re-exports them, so the generic surface, `nexus_uav`, names nothing of PX4. That list
is the whole of what an integration exposes. Its classes stay where they live, and a script imports
the few names it spells.
