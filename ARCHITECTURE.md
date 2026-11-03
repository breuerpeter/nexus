# Architecture

The shape of the tree today, one line per fact, and under `## Planned` each change the tree has not made yet, as its end shape. The design pages under `docs/design/` give the depth.

## Stack
- Newton is the physics: `ModelBuilder.add_usd` builds every model from Universal Scene Description (USD), and no code builds a vehicle by hand.
- Warp runs every per-tick kernel. The device stages replay as CUDA graphs, and Warp's tape gives the gradients.
- OpenUSD declares what a run builds: the vehicle, the scene and the run's override layer. A change to a run is a layer, a variant or an attribute, never a code path.
- PX4 Software In The Loop (SITL) is the one first-class controller. It flies over the MAVLink Hardware In The Loop (HIL) lockstep link, which `pymavlink` speaks.
- The RTX sensors render in the loop's process through `ovstage` and `ovrtx`, the extra `nexus-uav[rtx]`, pinned exactly and bumped with Isaac Lab's pins. The render stage reaches them through six calls and no other, the calls Newton's `ViewerRTX` and Isaac Lab's `OVRTXRenderer` use: `population.open_usd` or `add_usd_reference`, `write_attribute`, `advance_write_floor`, `attach_ovstage`, `step` and `map`.
- The render stage's pose write is the one place the stage binding of ovnewton takes over, once nexus runs Newton 1.7 on the ovstage pin Isaac Lab uses. The model stays on Newton's importer, and a Newton RTX sensor takes the camera class's place once one ships.
- `uv` manages two projects: the core, and the Isaac Lab trainer `nexus-rl` with its own lock, so Isaac Lab never reaches the core. Rerun is the one recording sink, and Docker runs every container peer through the `docker` SDK.

## Layout of `nexus_uav/_src/`
- `core/`: the loop, the `Component` contract, stages, signals and their value types, the clock, the registry, and the errors.
- `usd/`: the schema plugin, which holds no code, and its reader. A schema changes only by a new versioned name, and the reader fails an unknown one.
- `build/`: the one builder. It resolves the launch, fetches the assets, builds the Newton model, builds each component and peer through the registry, wires the signals, and hands the loop its parts.
- `physics/`: the fixed physics part. Each substep it zeroes the joint efforts, which the rotors' motors add their torque into, steps the model's own Newton actuators, a servo's, then the solver.
- `vehicle/sensors/`, `estimators/`, `commands/`, `forces/` and `controllers/`: the shipped components, one folder per role. `controllers/px4/` holds the one shipped controller.
- `vehicle/sensors/rtx/` holds the RTX sensors and the one render stage they require, which the build makes once per run by the name `ovrtx` and hands each sensor at `connect`.
- `guidance/`: the shipped guidance components. `scene/`: the world values the scene's `PhysicsScene` authors or the build derives.
- `peers/`: the peer contract at the top, then one folder per peer, `peers/<name>/`, with its runner, `pin.ref`, `image/` or `peer-src/` and its fake. A link's host end sits with the component that speaks it. The offboard client, which no component speaks, sits in the PX4 peer's folder.
- `recording/`: the Recorder. `logging/`: the Logger. `config/`: the catalog, the typed `LaunchConfig` and the receipt. `assets/`: the content-addressed resolver.
- `api/`: `Sim`, the one public entry. `cli/`: the `nexus` command. `nexus_uav/__init__.py` re-exports the public names lazily, and `nexus_uav.px4` is the one public namespace that names PX4.
- `.importlinter` holds the layers, keeps the examples out of the core, keeps PX4 out of every generic part, keeps Kit out of every tier, and keeps `ovrtx` and `ovstage` in the render stage.

## Contracts
- Every role has one contract: a component subclasses `Component` and states its stages, each with the signals it reads and writes.
- The roles run in one fixed order each tick: sensor, estimator, guidance, controller, link. Command and force elements run once per physics substep.
- An applied API schema on a prim declares a component. Its `nexus:` attributes are the class's keyword arguments.
- A component reads only the attributes of its own schema, and takes what it needs from another component as a signal. The build fails a class that reads a schema beside its own. `SIBLINGS_EXEMPT` in `core/components.py`, where an exception would stand, is empty.
- A rotor is three components on its rigid body, each from its own schema. The ESC turns its command into a terminal voltage from its duty map and bus. The motor turns that voltage into torque through Newton's actuator, built in code from the sheet's constants. The propeller turns speed into thrust. No `NewtonActuator` prim drives a rotor's joint, and the rotor speed droops under load as a real drive's does.
- The registry maps each schema, and each name a class requires, to its class. A project registers through the `nexus.registry` entry-point group, and a test passes `Sim(registry=...)`.
- The vehicle's USD declares each peer with a counterpart on the real vehicle. A component's class requires, by name, what the build makes once per run and hands it at `connect`: a peer with no counterpart, or an in-process service such as the render stage. Every peer ships a fake that speaks its links.
- The render stage takes poses from `body_q` on the device through DLPack, one write per frame.
- The run owns every address. The builder builds a link's end inside the run, and an end outside the run reads `sim.ports`.
- A wire on the vehicle is a USD relationship: `nexus:inputs:<signal>` for a signal, `nexus:serial:<line>` for a serial line. An input the file leaves empty takes the writer of its signal on the reader's own prim, else the vehicle's one writer. So a rotor's motor reads its own ESC with nothing authored.
- The clock, the physics, the Recorder and the Logger take settings, and nothing declares them. The Recorder keeps a history of every device signal, and the Logger writes the histories in blocks. A script reads `sim[path]` and `sim.read(path)`, and `Sim` has no accessor for a role.
- The receipt hashes the vehicle, the scene, each layer and the override layer, and names each peer's pin and the class each declaration resolved to.
- A second renderer, such as Newton's tiled camera, is a second class for the same camera schema: one more entry in the registry. A bridge to a message bus, such as Robot Operating System (ROS) 2, is a peer the vehicle declares and wires with `nexus:inputs`, never a change to the loop.

## Examples and projects
- The example controllers live in `nexus_uav/examples/`. `nexus_uav/examples/_lib/` is the shared example library, and `nexus-rl` imports only `nexus_uav` and `_lib`, a ruff ban.
- A project beside the framework ships its own schema plugin, classes, fakes and catalog through entry points, and patches nothing.

## Quality gates
- Lint: ruff with its docstring rules on, typos and the lock checks from pre-commit, and `import-linter` from the lint job. Vale owns the prose rules, and the `vale-added` hook checks the lines a branch adds.
- Tests mirror the source tree. The `gpu` and `px4_sitl` markers pick the CI leg. Shared stand-ins live in `conftest.py` fixtures.
- CI: lint, cpu-pytest, the docs build and vale gate every pull request on hosted runners. The `gpu` label buys one shared GPU box. Main flies the real peers with the real-time factor gates. The benchmark matrix runs on its schedule.
- release-please cuts releases from conventional commits. mkdocs-material builds the docs at the repo root, `mkdocstrings` renders the API reference from Google-style docstrings, and the Concepts page defines each term once.

## Planned
- The Kit render peer, `peers/kit/`, the render link's host end, `rendering/`, the Kit-served sensor classes, the `cesium` scene's catalog entry and the Kit-only asset scripts move to a plugin project in this repo, `nexus-kit/`, a second `uv` project that registers through `nexus.registry`; the core keeps no Kit line.
- The render stage hands frames back on the device, and no host copy sits on the loop's path. A frame is a device signal the Recorder keeps no history of, and the Logger copies it to the host at its own cadence.
- The example controllers register through the registry as a project's do and fly through the one builder; no example assembles its own loop.
- The builder reads typed launch values and the receipt, and no scenario dict remains. `nexus plan` prints the receipt without flying.
- A batch is a copies axis on every signal, not a second code path. The RL env builds its ring through the builder inside Isaac Lab's env, which keeps the model, the solver, the cloning and the MDP, and takes its camera observation from the framework's render stage; the framework is no Isaac Lab physics backend and imports no `isaaclab_*` package.
- Types: `ty` in strict mode checks the core, the tests, the scripts and `nexus-rl` in its own environment, so `py.typed` is a checked claim.
- The shared GPU box flies the examples against the fakes.
- The workflows pin each action to a commit hash and pass `actionlint` and `zizmor`; one update bot bumps the locks and the image bases; the examples harness scores in numpy and scipy, and no GPLv3 package enters any group.
- Vale checks heading case too, and the heading-case script goes. The docs site keeps each release's docs.
