---
description: "Fly the simulated vehicle from QGroundControl over PX4 Software In The Loop (SITL): the end-to-end nexus flight workflow, with the port map and gotchas."
---

# Running a SITL flight

The first end-to-end nexus workflow: fly the simulated vehicle from a ground
station, QGroundControl with virtual joysticks, over PX4 Software In The Loop (SITL).

Everything runs on `127.0.0.1`. **nexus** runs the physics, hosts the Rerun
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
- For a vehicle with a camera or lidar: an NVIDIA GPU and the NVIDIA Container Toolkit, since its
  sensors render in the Kit container. See [RTX cameras and lidar](#rtx-cameras-and-lidar).

## Terminals, in order

**1. QGroundControl.** Launch it. It listens on UDP 14550 and auto-connects once
PX4 sends a heartbeat.

**2. nexus sim:**

```bash
uv run nexus run
```

Builds PX4 SITL, starts the PX4 container, then the physics sim and the Rerun server on
gRPC :9876, and stops the container again on exit. Defaults to the `astro_max_base` vehicle +
`--control px4-sitl`. The PX4 console is a file, next to the run's recording:
`~/.cache/nexus/logs/px4-*.log`.

**3. Rerun viewer, optional:**

```bash
uv run rerun --connect
```

Attaches as a *client* to nexus's recording. `--connect` defaults to
`rerun+http://127.0.0.1:9876/proxy`. Start it after the sim, which must own
:9876 first. `uv run` launches the viewer bundled with the framework, so its
version matches the logger.

## RTX cameras and lidar

The vehicle Universal Scene Description (USD) file decides, with no flags. A `Camera` or
`OmniLidar` prim under the vehicle's root is an RTX sensor, and a run with one starts the **Kit
render peer**. That container renders the sensors while the loop flies on the host. The run sends it
the poses at each frame and takes the frames back a tick later. A vehicle with no such prim starts
no container.

- The first RTX run on a machine pulls NVIDIA's `nvcr.io/nvidia/isaac-sim:6.0.1`, about 21 GB,
  with no NGC login. Later runs reuse it, and nexus builds nothing. The container runs the image
  as pulled. The peer program ships in the package, in `nexus/_src/peers/kit/peer-src/`. It mounts
  read-only, so an update to the peer takes effect on the next run.
- A scene that declares a Cesium tileset, such as `--scene cesium`, fetches Cesium for Omniverse
  into the asset cache on its first run. No other scene fetches it.
- Kit boots in the background while PX4 builds and the physics compiles. Its shader cache lives in
  `~/.cache/nexus/kit/`, so the first boot is the slow one. The container's console goes to
  `~/.cache/nexus/logs/console-*.log`, next to the run's recording.
- The container runs as you and reads the asset cache and the folder of a local `--vehicle` or
  `--scene` file at their host paths, read-only, so everything it writes under `~/.cache/nexus`
  stays yours.
- A run whose Kit container can't start fails and names the cause, for example an unreachable
  Docker daemon. A Kit container that dies mid-flight ends the run the way a lost autopilot does.

- `--vehicle` accepts a registry vehicle name, such as `astro_max_fpv`, **or a local `.usd`/`.usdz`
  path**. Omit it for the registry's default vehicle.
  Every Astro Max vehicle carries the analytic PX4 suite, an Inertial Measurement Unit (IMU),
  mag, barometer, and Global Positioning System (GPS) as `sensor:*` prims. The vehicle USD is the
  single authority for all sensors, and a PX4 vehicle USD authoring none fails the build loudly.
- Cameras log JPEG frames and a `Pinhole` frustum to `cameras/<name>`. The lidar logs world-frame
  `Points3D` to `lidar/<name>`. `--debug` records the axes-only scene, which gives small `.rrd`s,
  and is the default for verification flights.

## Profiling

Every run carries an always-on loop profiler: integer-nanosecond phase marks, at negligible cost.
The end-of-run `profile [...]` line reports the sliding-window Real Time Factor (RTF), the tick p50
and p95, and the per-phase partition, plus the CUDA-event-timed GPU batch. In the partition,
`exchange` is the PX4 lockstep wait, `gpu+read` is the device batch plus the D2H sync,
`sensors.host` is the RTX render link, and `other` is an explicit residual. The same numbers land
in `Orchestrator.run_stats["profile"]`.

Every entry point shares the deep-diagnostics flags below. `nexus run` and the
examples launcher, `uv run -m nexus.examples <name> --profile`, spell them identically:

- `--profile`: periodic reports every 5 s. On an RTX run the detail spans `render.send` and
  `render.wait` show what a frame costs the loop: the pose send, and the wait for a frame the peer
  hasn't finished.
- `--trace <path>`: buffers spans and writes a Chrome or Perfetto trace on exit. Drag it onto
  [ui.perfetto.dev](https://ui.perfetto.dev).
- `--benchmark`, on an RTX run: the Kit peer runs the `isaacsim.benchmark.services` recorders from
  Isaac, which capture the system envelope the loop profiler can't see. They record render **GPU** frame time
  from `HydraEngineStats`, GPU memory plus host Resident Set Size (RSS), process CPU%, app-update
  frame time, and windowed-RTF stability. It writes one summary `INFO` line plus a full
  `benchmark-<ts>.json` next to the flight logs. It complements the profiler: the profiler
  partitions the tick, and this measures Kit and the system. The PhysX-bound recorders stay out,
  because physics here is Newton.

## Fly

Once PX4 logs `Ready for takeoff!` in `~/.cache/nexus/logs/px4-*.log`,
QGroundControl auto-connects. Arm and fly with the on-screen virtual joysticks.
Sim physics and PX4 state both appear in the Rerun viewer, because they share the
`nexus` recording.

## Worlds for the FPV camera

A vehicle whose USD authors RTX sensors, such as the `astro_max_fpv` variant's `FpvCam`, renders
them in the Kit peer. The FPV camera's world comes from the registry scene:

```bash
# photoreal geolocated globe: Google Photorealistic 3D Tiles streamed live at
# the scene's lat/lon, via Cesium for Omniverse v0.29.0:
export CESIUM_ION_TOKEN=<your ion access token>      # cesium.com/ion → Access Tokens
uv run nexus run --vehicle astro_max_fpv --scene cesium --geo 37.7942,-122.3954,-32 --view   # SF
```

A cesium scene is just a `geodetic_origin` in the registry: latitude, longitude, and the WGS84
ellipsoidal height of the street, which sets the globe so the street sits on the physics ground.
If it sits off, tweak the registry value, with no conversion. Tile selection runs one
hidden-cost viewport per camera, and the RTX lidar reflects off the tiles.
[Benchmarking](../reference/benchmarking.md) lists the measured real-time factor of each vehicle
on this scene. Without a token the run warns and falls back to a plain sky, and the flight continues.
The Cesium ion and Google Maps Platform terms govern the streamed tiles: see
[the license page](../license.md#hosted-assets).

## PX4 as a peer

The PX4 autopilot is a **peer** of the run: a process the run starts, speaks to over MAVLink, and
stops. The run chooses how, with `--px4`:

- `managed`, the default: the run starts the PX4 SITL container and stops it on exit.
- `external`: the run starts nothing and waits on its HIL port for an autopilot started elsewhere,
  a PX4 SITL of your own or a real autopilot on a bench.

**The PX4 tree.** The PX4 controller pins the PX4-Autopilot commit it flies, in
`nexus/_src/peers/px4_sitl/px4.ref`, which ships in the package. The first managed run on
a machine fetches that commit into `~/.cache/nexus/px4/<commit>/` and builds it there, minutes
once. Later runs rebuild only what changed. Two overrides:

- `PX4_DIR` names a checkout of your own, for work on PX4 itself. The run builds and flies it and
  fetches nothing.
- A project that keeps its own catalog, `nexus.registry.yaml`, keeps its own pin beside it in
  `nexus.px4.ref`, of the same form, `owner/repo@<commit>`. Every vehicle of the project then
  flies that tree, and a bump is one edit.

**The image.** The first run builds the `px4-sitl` image, the PX4 build toolchain, from the
Dockerfile the package ships, and tags it with a hash of that folder. An update rebuilds it only
when the Dockerfile changes.

**Ports.** PX4 SITL numbers every link from its **instance**, `--px4-instance N`, `0` by default.
It dials the sim's HIL server on 4560 + N, streams its offboard link to 14540 + N, and takes N + 1
as its MAVLink system id. The run derives its own addresses from the same number. So two runs on
one machine take two instances and never collide.

| Port      | Protocol | Link |
|-----------|----------|------|
| 4560 + N  | TCP      | PX4 → nexus, lockstep HIL: sensors in, actuators back |
| 14540 + N | UDP      | PX4 → the run's operator, the offboard link `sim.operator` commands over |
| 14550     | UDP      | MAVLink telemetry and commands from PX4 to QGroundControl, every instance |
| 9876      | gRPC     | Rerun recording, which nexus **serves** and the viewer **connects** to |

## Observability in Rerun

Everything lands in **one** recording, app ID `nexus` and recording ID
`nexus`, and every producer writes it **from the sim's own process**:

- **the sim scene**: logged by nexus itself. The blueprint **hides** the ground plane,
  `/model/shapes/shape_0`, **by default** because it occludes the vehicle.
- **framework events**: the `newton` logger, under `logs/sim`.
- **PX4's own view of the flight**: not in the recording. PX4 keeps it in its console log,
  `~/.cache/nexus/logs/px4-*.log`, and in its `ULog`, both artifacts of the run.
- **test and driver stages**: a driver in the sim's process logs each stage with `na.logger.info("…")`,
  which writes to the console and, when recording, the `logs/sim` panel.

**Serve or file, never both: one knob, `--viewer`.** A run can't produce both a live gRPC server
and a *complete* `.rrd` from one process, because rerun's serve and file sinks are mutually exclusive, so:

- **`--viewer`**, the default: serve the recording live on `:9876` and connect a viewer with
  `uv run rerun --connect rerun+http://127.0.0.1:9876/proxy`. It writes no file, so
  **save it from the viewer** to keep an `.rrd`.
- **`--no-viewer`** on the command line, or [`Sim(viewer=False)`][nexus.Sim]: write the full
  `.rrd` to disk. `uv run nexus run` logs its path, and a driver reads it from `sim.artifacts()`.
  Use it for CI, or when something else holds `:9876`.

Both modes carry the same content, PX4 included: there is no second recording and no merge step.

## Gotchas

!!! note "One PX4 per instance"
    Each run's container carries the run's own name, `nexus-px4-<pid>-<instance>`, and the run removes it on
    exit, so a second run stops nothing of the first. Two live runs on one instance would share
    PX4's ports, so the second fails at its start and names the process that holds the instance:
    give it `--px4-instance 1`. A run killed without its teardown, a closed terminal or a harness's
    timeout, leaves its PX4 running, and the next run on that instance removes it.

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
