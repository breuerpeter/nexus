# Target architecture

The tree lags this file where the open issues say it does: until such an issue closes, the code keeps the older shape.

## Stack
- Newton is the physics: `ModelBuilder.add_usd` builds every model from Universal Scene Description (USD), and no code builds a vehicle by hand.
- Warp runs every per-tick kernel. The device stages replay as CUDA graphs, and Warp's tape gives the gradients.
- OpenUSD declares what a run builds: the vehicle, the scene and the run's override layer. A change to a run is a layer, a variant or an attribute, never a code path.
- PX4 Software In The Loop (SITL) is the one first-class controller. It flies over the MAVLink Hardware In The Loop (HIL) lockstep link, which `pymavlink` speaks.
- The RTX sensors render in the loop's process through `ovrtx` and `ovstage`, the extra `nexus-uav[rtx]`, pinned as `newton[rtx]` pins them. The Kit render peer and its image leave the framework.
- Rerun is the one recording sink. Docker runs every container peer through the `docker` SDK.
- `uv` manages two projects: the core, and the Isaac Lab trainer `nexus-rl` with its own lock, so Isaac Lab never reaches the core.

## Layout of `nexus_uav/_src/`
- `core/`: the loop, the `Component` contract, stages, signals and their value types, the clock, the registry, and the errors.
- `usd/`: the schema plugin, which holds no code, and its reader. A schema changes only by a new versioned name, and the reader fails an unknown one.
- `build/`: the one builder. It resolves the launch, fetches the assets, builds the Newton model, builds each component and peer through the registry, wires the signals, and hands the loop its parts.
- `physics/`: the fixed physics part. It steps every Newton actuator, then the solver.
- `vehicle/sensors/`, `estimators/`, `commands/`, `forces/` and `controllers/`: the shipped components, one folder per role. `controllers/px4/` holds the one shipped controller.
- `guidance/`: the shipped guidance components. `scene/`: the world values the scene's `PhysicsScene` authors or the build derives.
- `peers/`: the peer contract at the top, then one folder per peer, `peers/<name>/`, with its runner, `pin.ref`, `image/` or `peer-src/`, its fake and its link's host end.
- `recording/`: the Recorder. `logging/`: the Logger. `config/`: the catalog, the typed `LaunchConfig` and the receipt. `assets/`: the content-addressed resolver.
- `api/`: `Sim`, the one public entry. `cli/`: the `nexus` command.
- No folder, class or loop slot names rendering, so `rendering/` goes.
- `nexus_uav/__init__.py` re-exports the public names lazily. `nexus_uav.px4` is the one public namespace that names PX4.
- `.importlinter` holds the layers, keeps the examples out of the core and keeps PX4 out of every generic part.

## Contracts
- Every role has one contract: a component subclasses `Component` and states its stages, each with the signals it reads and writes.
- The roles run in one fixed order each tick: sensor, estimator, guidance, controller, link. Command and force elements run once per physics substep.
- An applied API schema on a prim declares a component. Its `nexus:` attributes are the class's keyword arguments.
- The registry maps each schema, and each name a class requires, to its class. A project registers through the `nexus.registry` entry-point group, and a test passes `Sim(registry=...)`.
- The vehicle's USD declares each peer with a counterpart on the real vehicle. A component's class requires each other peer. Every peer ships a fake that speaks its links.
- The run owns every address. The builder builds a link's end inside the run, and an end outside the run reads `sim.ports`.
- A wire on the vehicle is a USD relationship: `nexus:inputs:<signal>` for a signal, `nexus:serial:<line>` for a serial line.
- The clock, the physics, the Recorder and the Logger take settings, and nothing declares them.
- The Recorder keeps a history of every fixed-width device signal. A script reads `sim[path]` and `sim.read(path)`, and `Sim` has no accessor for a role.
- The builder reads typed launch values and the receipt, and no scenario dict remains.
- The receipt hashes the vehicle, the scene, each layer and the override layer, and names each peer's pin. `nexus plan` prints it without flying.
- A batch is a copies axis on every signal, not a second code path. The RL env builds its ring through the builder.

## Examples and projects
- The example controllers live in `nexus_uav/examples/`, register through the registry as a project's do, and fly through the one builder.
- `nexus_uav/examples/_lib/` is the shared example library. `nexus-rl` imports only `nexus_uav` and `_lib`, a ruff ban.
- A project beside the framework ships its own schema plugin, classes, fakes and catalog through entry points, and patches nothing.
- A second renderer, such as Newton's tiled camera, is a second class for the same camera schema: one more entry in the registry.
- A bridge to a message bus, such as Robot Operating System (ROS) 2, is a peer the vehicle declares and wires with `nexus:inputs`, never a change to the loop.

## Quality gates
- Lint: ruff with its docstring rules on, `import-linter`, typos and the lock checks, all from pre-commit. Vale owns every prose rule, heading case too.
- Types: `ty` in strict mode checks the core, the tests, the scripts and `nexus-rl` in its own environment, so `py.typed` is a checked claim.
- Tests mirror the source tree. The `gpu` and `px4_sitl` markers pick the CI leg. Shared stand-ins live in `conftest.py` fixtures.
- CI: lint and cpu-pytest gate every pull request on hosted runners. The `gpu` label buys one shared GPU box against the fakes. Main flies the real peers with the real-time factor gates. The registry matrix runs weekly.
- The workflows pin each action to a commit hash and pass `actionlint` and `zizmor`. One update bot bumps the locks and the image bases.
- The examples harness scores in numpy and scipy, and no GPLv3 package enters any group.
- release-please cuts releases from conventional commits. The docs site keeps each release's docs.
- mkdocs-material builds the docs at the repo root. `mkdocstrings` renders the API reference from Google-style docstrings. The Concepts page defines each term once.
