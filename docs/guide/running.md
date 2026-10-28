---
description: "Fly the simulated vehicle from QGroundControl over PX4 Software In The Loop (SITL): the end-to-end nexus flight workflow, with the port map and gotchas."
---

# Running a SITL flight

The first end-to-end nexus workflow: fly the simulated vehicle from a ground
station, QGroundControl with virtual joysticks, over PX4 Software In The Loop (SITL).

Everything runs on `127.0.0.1`. The **nexus** sim runs the physics, hosts the Rerun
recording, and starts **PX4 SITL** as a container that connects back over **TCP
4560** for lockstep Hardware In The Loop (HIL). PX4 talks MAVLink over the User Datagram
Protocol (UDP) to **QGroundControl** on **UDP 14550**. An optional **Rerun viewer** watches
the live recording over **gRPC 9876**. PX4 is a peer of the run: the run starts it, stops it,
and hands it its ports, see [PX4 as a peer](#px4-as-a-peer).

## Prerequisites

- `uv sync` has run. It installs the framework, including the Rerun viewer bundled
  with `rerun-sdk`, so you need no separate Rerun install.
- QGroundControl installed, with **Virtual Joystick** enabled under
  `Application Settings` → `General` → `Virtual Joystick`.
- Docker. The first PX4 run on a machine builds the small `px4-sitl` image from the Dockerfile
  the package ships, and fetches and builds the PX4 tree, see [PX4 as a peer](#px4-as-a-peer).
- For a vehicle with a camera or lidar: an NVIDIA GPU with RTX and the extra `nexus-uav[rtx]`. See
  [RTX cameras and lidar](#rtx-cameras-and-lidar).

## Terminals, in order

**1. QGroundControl.** Launch it. It listens on UDP 14550 and auto-connects once
PX4 sends a heartbeat.

**2. The nexus sim:**

```bash
uv run nexus run --vehicle astro_max_base --scene empty
```

Builds PX4 SITL, starts the PX4 container, then the physics sim, and stops the container again
on exit. Add `--view` to serve the Rerun recording on gRPC :9876, see [Observability in Rerun](#observability-in-rerun). The `astro_max_base` vehicle's
Universal Scene Description (USD) file declares PX4 and its airframe. Every file a run writes lands in one folder for the run,
`~/.cache/nexus/logs/<stamp>-<pid>/`, or the folder a script names with `Sim(run_dir=)`. It holds the recording, `nexus.rrd`, and each peer's files under a folder named for it. PX4's console is `px4_sitl/console.log`, and its `.ulg` flight log lands beside it, `px4_sitl/<date>/<time>.ulg`.

A run that never starts, or that loses PX4 or the Kit container before it ends, stops with an error
and exits non-zero. The message names each peer's console log and quotes the peer's own error lines,
so the output alone says why. A PX4 tree without the vehicle's airframe, for example, gives
`ERROR [init] Unknown model none_astro_max`. A PX4 that exits while it boots fails the run at once.
PX4 counts as lost when it sends no answer for 2 s. A script catches the same error as
`nexus_uav.RunError`, which separates a broken run from its own check failing.

**3. Rerun viewer, optional:**

```bash
uv run rerun --connect
```

Attaches as a *client* to nexus's recording. `--connect` defaults to
`rerun+http://127.0.0.1:9876/proxy`. Start it after a sim run with `--view`, which must own
:9876 first. `uv run` launches the viewer bundled with the framework, so its
version matches the logger.

## RTX cameras and lidar

The vehicle Universal Scene Description (USD) file decides, with no flags. A camera or thermal camera schema
on a `Camera` prim, or a lidar schema on an `OmniLidar` prim, declares an RTX sensor, and a run with one
makes the **render stage**. It renders the sensors in the run's own process with ovrtx, from the run's
scene and vehicle files. At a due tick it poses the vehicle's bodies, renders, and hands each due sensor its
frame, so a frame shows the tick that asked for it. A vehicle with no such prim loads no renderer.

- The renderer ships as the extra `nexus-uav[rtx]`: run `uv sync --extra rtx` in this repo, or install
  `nexus-uav[rtx]` with pip. It needs an NVIDIA GPU with RTX. A vehicle with an RTX sensor in an
  environment without the extra fails at its build, and the message names the extra.
- The first run on a machine compiles the shaders, about five minutes, and later runs reuse them from
  `~/.cache/nexus/ovrtx/`, which outlasts the environment. The console of ovrtx goes to
  `ovrtx/console.log` in the run's folder, next to the run's recording.
- A lidar's prim applies `OmniSensorGenericLidarCoreAPI` beside its sensor schema, or the build fails
  and names the prim. The render stage sets the scan rate from the sensor's rate. Each scan reaches the
  lidar a period after it completes: a 10 Hz scan that completes at 104 ms arrives at 204 ms.

- `--vehicle` accepts a catalog vehicle name, such as `astro_max_fpv`, **or a local `.usd`/`.usdz`
  path**. `--scene` accepts a catalog scene name, such as `empty`, or a local scene USD path.
  A run names both.
  Every Astro Max vehicle carries the analytic PX4 suite, an Inertial Measurement Unit (IMU),
  mag, barometer, and Global Positioning System (GPS). Each is a prim under the body it rides that
  applies its sensor schema, such as `NexusImuAPI`. The vehicle USD is the single authority for all
  sensors, and the [schema reference](../reference/schemas.md) lists the attributes of each schema.
  Each sensor declares the rate PX4's gz `x500` model flies, 250 Hz on the IMU, 100 on the mag, 50 on
  the barometer and 30 on the GPS, and PX4 receives each at its rate.
- A camera logs JPEG frames and a `Pinhole` frustum to `sim/vehicle/sensors/<name>`. A lidar logs
  world-frame `Points3D` to `sim/vehicle/sensors/<name>` too. `--debug` records the axes-only scene, which gives small `.rrd`s,
  and is the default for verification flights.

## Profiling

Every run carries an always-on loop profiler: integer-nanosecond phase marks, at negligible cost.
The end-of-run `profile [...]` line reports the sliding-window Real Time Factor (RTF), the tick p50
and p95, and the per-phase partition, plus the CUDA-event-timed GPU batch. In the partition,
`exchange` is the PX4 lockstep wait, `gpu+read` is the device batch plus the D2H sync,
`sensors.host` is the RTX sensors' render, and `other` is an explicit residual. The same numbers land
in `Loop.run_stats["profile"]`.

Every entry point shares the deep-diagnostics flags below. `nexus run` and the
examples launcher, `uv run -m nexus_uav.examples <name> --profile`, spell them identically:

- `--profile`: periodic reports every 5 s.
- `--trace <path>`: buffers spans and writes a Chrome or Perfetto trace on exit. Drag it onto
  [ui.perfetto.dev](https://ui.perfetto.dev).
- `--benchmark`, on an RTX run: the Kit peer runs the `isaacsim.benchmark.services` recorders from
  Isaac, which capture the system envelope the loop profiler can't see. They record render **GPU** frame time
  from `HydraEngineStats`, GPU memory plus host Resident Set Size (RSS), process CPU%, app-update
  frame time, and windowed-RTF stability. It writes one summary `INFO` line plus a full
  `kit/benchmark.json` in the run's folder, beside its console. It complements the profiler: the profiler
  partitions the tick, and this measures Kit and the system. The PhysX-bound recorders stay out,
  because physics here is Newton.

## Fly

Once PX4 logs `Ready for takeoff!` in `px4_sitl/console.log` in the run's folder,
QGroundControl auto-connects. Arm and fly with the on-screen virtual joysticks.
Sim physics and PX4 state both appear in the Rerun viewer, because they share the
`nexus` recording.

## Worlds for the FPV camera

A vehicle whose USD declares RTX sensors, such as the `astro_max_fpv` variant's `FpvCam` with its
`NexusCameraAPI` schema, renders them in the render stage. The FPV camera's world comes from the catalog scene:

```bash
# photoreal geolocated globe: Google Photorealistic 3D Tiles streamed live at
# the scene's origin, via Cesium for Omniverse v0.29.0:
export CESIUM_ION_TOKEN=<your ion access token>      # cesium.com/ion → Access Tokens
uv run nexus run --vehicle astro_max_fpv --scene cesium --view
```

Every scene states where it sits on Earth. Its `PhysicsScene` prim, `/World/PhysicsScene` in each
shipped scene, applies `NexusSceneAPI` and authors `nexus:geodeticOrigin`: the latitude, the
longitude and the altitude of the world origin. For the cesium globe the altitude is the WGS84
ellipsoidal height of the street, which sets the globe so the street sits on the physics ground. A
run fails on a scene whose `PhysicsScene` authors no origin.

An override layer moves the origin, as it changes any other declared value. To fly the globe over
San Francisco, write a layer that authors the new origin on the scene's `PhysicsScene`:

```usda
#usda 1.0

over "World"
{
    over "PhysicsScene"
    {
        double3 nexus:geodeticOrigin = (37.7942, -122.3954, -29)
    }
}
```

```bash
uv run nexus run --vehicle astro_max_fpv --scene cesium --layer sf.usda --view
```

If the street sits off, change the altitude in the layer. Tile selection runs one hidden-cost
viewport per camera, and the RTX lidar reflects off the tiles.
[Benchmarking](../reference/benchmarking.md) lists the measured real-time factor of each vehicle
on this scene. Without a token the run warns and falls back to a plain sky, and the flight continues.
The Cesium ion and Google Maps Platform terms govern the streamed tiles: see
[the license page](../license.md#hosted-assets).

## PX4 as a peer

The PX4 autopilot is a **peer** of the run: a process the run starts, speaks to over MAVLink, and
stops. The vehicle's USD declares it: `NexusPx4SitlAPI` on a scope of its own, `/astro_max/Px4Sitl`,
with the airframe PX4 boots in `nexus:airframe`. PX4's controller, `NexusPx4API_1`, sits on the
controller's scope, `/astro_max/Controller`. A run of a vehicle that declares the peer starts the PX4
SITL container and stops it on exit.

**An autopilot started elsewhere.** To fly a PX4 SITL of your own, or a real autopilot on a bench,
drop the declaration with an **override layer**, a small USD file the run composes over the vehicle
and the scene, which deactivates the peer's scope:

```usda
#usda 1.0

over "astro_max"
{
    over "Px4Sitl" (
        active = false
    )
    {
    }
}
```

Name the vehicle's root prim and the peer's scope in the `over`s, and pass the file with
`--layer`, or `Sim(layer=)`:

```bash
uv run nexus run --vehicle astro_max_base --scene empty --layer external_px4.usda
```

The run starts nothing and waits on instance 0's HIL port, 4560, which a stock `px4 -i 0` dials. The
run's receipt records the layer's sha256 beside the vehicle's. The same kind of layer changes a
declared value or selects a variant without re-authoring the hosted vehicle.

**Parameters and missions across runs.** PX4 keeps what it stores, the parameters set from a ground
station and the last mission it accepted, across runs, as a real vehicle keeps them across a reboot.
Set a parameter in QGroundControl, stop the sim, start it again, and PX4 boots with it. PX4 keeps one
store per airframe and instance, in `~/.cache/nexus/px4_sitl/store/<airframe>/<instance>/`. A
vehicle on another airframe keeps its own. A run takes the lowest free instance, so a restart finds
the store only on the instance it flew on, which another run on the machine can hold.

To start from the values of the airframe alone, ask the run to start clean: `--clean`, or
`Sim(clean=True)`. Every peer that keeps a store empties it first. `NEXUS_CLEAN=1` makes clean the
default for every run in the shell. The test suite sets it, and the examples start clean, so no test
or example flies on what an earlier run stored.

**The PX4 tree.** The PX4 SITL peer pins the PX4-Autopilot commit it flies, in
`nexus_uav/_src/peers/px4_sitl/pin.ref`, which ships in the package. The first managed run on
a machine fetches that commit into `~/.cache/nexus/px4_sitl/<commit>/` and builds it there, minutes
once. Later runs rebuild only what changed. Every peer resolves its pin the same way, so PX4 has
the two overrides every peer has:

- A project that keeps its own catalog, `nexus.catalog.yaml`, keeps its own pin beside it in
  `nexus.px4_sitl.ref`, of the same form, `owner/repo@<commit>`. Every vehicle of the project then
  flies that tree, and a bump is one edit. An installed package that
  [adds a catalog](your-own-assets.md#a-package-that-adds-a-catalog) ships its pin the same way, and
  a project's file overrides a package's.
- `NEXUS_PX4_SITL_REF` overrides every pin file and touches no tracked file. It takes a pin of the same form,
  or the absolute path of a checkout of your own, for work on PX4 itself. The run builds and flies
  that checkout and fetches nothing.

The Kit render peer pins its image the same way: `nexus_uav/_src/peers/kit/pin.ref` names
`nvcr.io/nvidia/isaac-sim:6.0.1`, and `nexus.kit.ref` or `NEXUS_KIT_REF` names another, a local tag
among them. The file and the variable take the peer's folder name. A pin file beside a catalog
that names no peer fails the run, and so does a pin not in its peer's form. The run's receipt records
each peer's pin under `peers`, with the file or the variable it came from, and for a checkout its
checked-out commit.

**The image.** The first run builds the `px4-sitl` image, the PX4 build toolchain, from the
Dockerfile the package ships, and tags it with a hash of that folder. An update rebuilds it only
when the Dockerfile changes.

**A script in a container.** The daemon that starts PX4 reads each folder the PX4 container mounts
on its own host. Take a script that runs in a container of its own and starts PX4 on the host's
daemon through the docker socket. It binds two folders at the same path inside as on the host: the
PX4 tree, which `NEXUS_PX4_SITL_REF` can name, and the run's folder, which the script names with
`Sim(run_dir=)`. The script's `HOME` stays the container's own, so its caches stay out of the run's
folder.

```yaml
services:
  test:
    network_mode: host
    environment:
      - NEXUS_PX4_SITL_REF=${PX4_TREE}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ${PX4_TREE}:${PX4_TREE}
      - ${RESULTS_DIR}:${RESULTS_DIR}
```

```python
with Sim("astro_max_base", scene="empty", log=True, run_dir=f"{os.environ['RESULTS_DIR']}/run") as sim:
    ...
```

The run creates the folder, and refuses one that holds files already. It writes its recording,
PX4's console and PX4's `.ulg` flight log there and nothing else. A folder the host doesn't have fails
the run before it flies: docker reports `bind source path does not exist` and the path.

**Ports.** PX4 SITL numbers every link from its **instance** N. It dials the sim's HIL server on
4560 + N, streams its offboard link to 14540 + N, and takes N + 1 as its MAVLink system id. The run
picks the lowest instance free on this machine and derives its own addresses from the same number,
so two runs on one machine take two instances and never collide. A script reads the address of a
link it opens itself from the run's port map: `sim.ports["offboard"]` holds the offboard link's
port and PX4's system id, which [`nexus_uav.px4.OffboardClient`](../reference/api/px4.md) takes.

| Port      | Protocol | Link |
|-----------|----------|------|
| 4560 + N  | TCP      | PX4 → nexus, lockstep HIL: sensors in, actuators back |
| 14540 + N | UDP      | PX4 → a script's `nexus_uav.px4.OffboardClient`, the offboard link `sim.ports["offboard"]` names |
| 14550     | UDP      | MAVLink telemetry and commands from PX4 to QGroundControl, every instance |
| 9876      | gRPC     | Rerun recording, which nexus **serves** and the viewer **connects** to |

## Observability in Rerun

Everything lands in **one** recording, app ID `nexus` and recording ID
`nexus`, and every producer writes it **from the sim's own process**:

- **the sim scene**: logged by nexus itself. The blueprint **hides** the ground plane,
  `/sim/model/shapes/shape_0`, **by default** because it occludes the vehicle.
- **framework events**: the `nexus` logger, at `sim/logs/<module>`.
- **PX4's own view of the flight**: not in the recording. PX4 keeps it in its console log,
  `px4_sitl/console.log`, and in its `ULog`, `px4_sitl/<date>/<time>.ulg`, both in the run's folder and both artifacts of the run.
- **test and driver stages**: a driver in the sim's process logs each stage with `nx.logger.info("…")`,
  which writes to the console and, when recording, the Logs pane.

**Serve or file, never both: two flags, `--view` and `--log`, both off.** A run can't produce both
a live gRPC server and a *complete* `.rrd` from one process, because rerun's serve and file sinks are
mutually exclusive, so:

- **`--view`** on the command line, or [`Sim(view=True)`][nexus_uav.Sim]: serve the recording live on
  `:9876` and connect a viewer with `uv run rerun --connect rerun+http://127.0.0.1:9876/proxy`.
  It writes no file, so **save it from the viewer** to keep an `.rrd`.
- **`--log`** on the command line, or [`Sim(log=True)`][nexus_uav.Sim]: write the full `.rrd` to disk.
  `uv run nexus run` logs its path, and a driver reads it from `sim.artifacts()`. Use it for CI,
  or when something else holds `:9876`.

The tool refuses `--view` and `--log` together, and `Sim` raises `ValueError`. A run with neither
records nothing: it serves nothing and writes no `.rrd`, so it runs at full headless speed.

Both modes carry the same content, PX4 included: there is no second recording and no merge step.

## Gotchas

!!! note "One PX4 per instance"
    Each run's container carries the run's own name, `nexus-px4_sitl-<pid>-<instance>`, and the run removes it on
    exit, so a second run stops nothing of the first. A second run takes the next free instance. A
    run killed without its teardown, a closed terminal or a harness's timeout, leaves its PX4
    running, and the next run on that instance removes it. Two runs that fly autopilots started
    elsewhere both need port 4560, so the second fails and names the process that holds it.

!!! note "Use `uv run rerun --connect …`, never a bare `rerun`"
    Two reasons. **First,** nexus hosts the Rerun gRPC server on 9876, and the viewer is
    a client. A bare `rerun` opens *its own* server on 9876 and wins the
    port, so nexus's sim data goes nowhere. **Second,** Rerun
    couples the viewer and SDK versions: `uv run rerun` launches the bundled,
    version-matched viewer, whereas a global `rerun` might be a different version.

!!! note "Agent-driven viewer over the Model Context Protocol (MCP)"
    The checked-in `.mcp.json` registers Rerun's MCP server, `uv run rerun viewer-mcp`,
    with Claude Code. An agent can then open recordings, seek the timeline, and take
    screenshots itself. The MCP server controls a separately running viewer over gRPC.
    On a machine without a display, start one with `uv run rerun --headless`.

!!! note "Preflight: strong magnetic interference"
    The `none_astro_max` SITL airframe ships generic parameters, with no real
    mag calibration, so QGroundControl shows a *Strong magnetic interference* preflight
    warning. Expected in SITL. It doesn't block flying.
