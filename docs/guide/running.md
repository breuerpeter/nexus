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
the live recording over **gRPC 9876**.

## Prerequisites

- `uv sync` has run. It installs the framework, including the Rerun viewer bundled
  with `rerun-sdk`, so you need no separate Rerun install.
- QGroundControl installed, with **Virtual Joystick** enabled under
  `Application Settings` → `General` → `Virtual Joystick`.
- Docker. The `px4-sitl` image pulls from the GitHub Container Registry (GHCR) on first run.
- A PX4-Autopilot checkout at `$PX4_DIR`, default `~/code/px4`, on the `p/newton`
  branch. The sim builds and runs it but doesn't supply it.

## Terminals, in order

**1. QGroundControl.** Launch it. It listens on UDP 14550 and auto-connects once
PX4 sends a heartbeat.

**2. nexus sim:**

```bash
uv run nexus run
```

Builds PX4 SITL, starts the physics sim and the Rerun server on gRPC :9876, then
launches the PX4 container against it and kills it again on exit. Defaults to the
`astro_max_base` vehicle + `--control px4-sitl`. The PX4 console is a file, next to the
run's recording: `~/.cache/nexus/logs/px4-*.log`.

**3. Rerun viewer, optional:**

```bash
uv run rerun --connect
```

Attaches as a *client* to nexus's recording. `--connect` defaults to
`rerun+http://127.0.0.1:9876/proxy`. Start it after the sim, which must own
:9876 first. `uv run` launches the viewer bundled with the framework, so its
version matches the logger.

## Isaac Sim runtime: RTX cameras + lidar

The vehicle Universal Scene Description (USD) file decides the runtime, with no flags. If it
carries RTX sensor prims, `Camera` or `OmniLidar`, `nexus run` detects them and
auto-launches the Isaac Sim container from `docker/docker-compose.yml`. A sensor-less vehicle
runs on the lean standalone runtime.

- `--vehicle` accepts a registry vehicle name, such as `astro_max_fpv`, **or a local `.usd`/`.usdz`
  path**. Omit it for the registry's default vehicle.
  Every Astro Max vehicle carries the analytic PX4 suite, an Inertial Measurement Unit (IMU),
  mag, barometer, and Global Positioning System (GPS) as `sensor:*` prims. The vehicle USD is the
  single authority for all sensors, and a PX4 vehicle USD authoring none fails the build loudly.
- Cameras log JPEG frames and a `Pinhole` frustum to `cameras/<name>`. The lidar logs world-frame
  `Points3D` to `lidar/<name>`. `--debug` records the axes-only scene, which gives small `.rrd`s,
  and is the default for verification flights.
- Recording sanity: `uv run --extra policy python tools/analyze_rrd_video.py [file.rrd]` reports the
  real frame rate, counting distinct frames, and whether the camera and body poses coincide.
- Runtime invariants, to check after container upgrades: `scripts/probes/` covers captured-physics
  validity and camera rigidity.

## Profiling

Every run carries an always-on loop profiler: integer-nanosecond phase marks, at negligible cost.
The end-of-run `profile [...]` line reports the sliding-window Real Time Factor (RTF), the tick p50
and p95, and the per-phase partition, plus the CUDA-event-timed GPU batch. In the partition,
`exchange` is the PX4 lockstep wait, `gpu+read` is the device batch plus the D2H sync,
`sensors.host` is the RTX render seam, and `other` is an explicit residual. The same numbers land
in `Orchestrator.run_stats["profile"]`.

Every entry point shares the deep-diagnostics flags below. `nexus run` and the
examples launcher, `uv run -m nexus.examples <name> --profile`, spell them identically:

- `--profile`: periodic reports every 5 s, plus detail spans `render.kit` and `grab.<camera>`.
  Inside Kit the phases also mirror into `carb.profiler` zones. Pick a backend with
  `--/app/profilerBackend=cpu|tracy|nvtx` for Tracy or Nsight deep dives.
- `--trace <path>`: buffers spans and writes a Chrome or Perfetto trace on exit. Drag it onto
  [ui.perfetto.dev](https://ui.perfetto.dev).
- `--benchmark`, on the `isaacsim` runtime: the `isaacsim.benchmark.services` recorders from Isaac
  capture the system envelope the loop profiler can't see. They record render **GPU** frame time
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

## Worlds for the FPV camera on the `isaacsim` runtime

A vehicle whose USD authors RTX sensors, such as the `astro_max_fpv` variant's `FpvCam`, routes to
the **`isaacsim`** runtime automatically: `--runtime auto` launches the Kit container.
The FPV camera's world comes from the registry scene:

```bash
# photoreal geolocated globe: Google Photorealistic 3D Tiles streamed live at
# the scene's lat/lon, via Cesium for Omniverse v0.29.0:
export CESIUM_ION_TOKEN=<your ion access token>      # cesium.com/ion → Access Tokens
uv run nexus run --scene cesium --geo 37.7942,-122.3954 --view   # SF (cesium defaults to Seattle)
```

A cesium scene is just a `geodetic_origin`, latitude and longitude, in the registry. At startup
the run measures the surface height from the streamed tiles: an iterative depth probe aligns the
street with the physics ground to centimetres, so new locations need no calibration. Tile
selection runs one hidden-cost viewport per camera, and the RTX lidar reflects off the tiles. The
whole config, two cameras at 24 fps plus full logging, holds >=1x realtime on the reference
machine. Without a token the run warns and falls back to a plain sky, and the flight continues.
The Cesium ion and Google Maps Platform terms govern the streamed tiles: see
[the license page](../license.md#hosted-assets).

## Ports

| Port  | Protocol | Link |
|-------|----------|------|
| 4560  | TCP      | PX4 ↔ nexus, lockstep HIL: sensors in, actuators back |
| 14550 | UDP      | MAVLink telemetry and commands from PX4 to QGroundControl |
| 9876  | gRPC     | Rerun recording, which nexus **serves** and the viewer **connects** to |

## Observability in Rerun

Everything lands in **one** recording, app ID `nexus` and recording ID
`nexus`, and every producer writes it **in-process**:

- **the sim scene**: logged in-process by nexus. The blueprint **hides** the ground plane,
  `/model/shapes/shape_0`, **by default** because it occludes the vehicle.
- **framework events**: the `newton` logger, under `logs/sim`.
- **PX4's own view of the flight**: not in the recording. PX4 keeps it in its console log,
  `~/.cache/nexus/logs/px4-*.log`, and in its `ULog`, both artifacts of the run.
- **test and driver stages**: an in-process driver logs each stage with `na.logger.info("…")`,
  which writes to the console and, when recording, the `logs/sim` panel.

**Serve or file, never both: one knob, `--viewer`.** A run can't produce both a live gRPC server
and a *complete* `.rrd` in-process, because rerun's serve and file sinks are mutually exclusive, so:

- **`--viewer`**, the default: serve the recording live on `:9876` and connect a viewer with
  `uv run rerun --connect rerun+http://127.0.0.1:9876/proxy`. It writes no file, so
  **save it from the viewer** to keep an `.rrd`.
- **`--no-viewer`** on the command line, or [`Sim(viewer=False)`][nexus.Sim]: write the full
  `.rrd` to disk. `uv run nexus run` logs its path, and a driver reads it from `sim.artifacts()`.
  Use it for CI, or when something else holds `:9876`.

Both modes carry the same content, PX4 included: there is no second recording and no merge step.

## Gotchas

!!! note "One PX4 at a time"
    A leftover `nexus-px4-sitl` container would keep the MAVLink ports, 18570 → 14550, bound
    and starve the next run of its heartbeat. So each launch force-removes any container of that
    name before starting its own. Nothing to clear by hand.

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
