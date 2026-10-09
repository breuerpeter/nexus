---
description: "How to log from examples and user code: the framework logger, which routes to the console and the Rerun recording."
---

# Logging

Log from examples and user code through the framework logger, `nx.logger`, which is the standard-library logger named `nexus`:

```python
import nexus_sim as nx

nx.logger.info("reached the waypoint")
nx.logger.warning("flight degraded")
```

It always writes to the **console**, on stderr. When a recording is active, through `--log` or `--view`,
or `Sim(log=…)` or `Sim(view=…)`, it **also** tees every record into the recording as a Rerun
`TextLog`, at `sim/logs/<module>`, where `<module>` is the module that logged it. So a result shows up
in the terminal, including in headless CI where no recording exists, *and* in the `.rrd` for review.
No bare `print` needed.

The framework itself logs the same way, so example output sits alongside the framework's own
`[sim/…]` lines.

## What the recording holds

Every value with a fixed width comes from the Recorder's histories: each body's and joint's state
and each sensor's output, one row per tick. The `Logger` writes them in blocks. Each time the
Recorder's staging buffers drain, every 4096 ticks, it appends that block of every history to the
file as columns. A block holds each quantity's series, the base body's pose, the scene's poses and
the flown path. The last block goes out at teardown. The `Logger` logs the scene's meshes once, at
the start. A viewer that follows a live run shows each block as it lands.

A component logs a value with no fixed width live, as it produces it, through the scoped `Logger`
the loop hands it with `set_logger`. A camera logs its frames, a lidar its points, a guidance its
waypoints and reference, and a controller its horizon.

Every stop that reaches Python writes the recording. Teardown writes the last block and closes the
file before it closes the controller and stops the peers. SIGTERM and SIGHUP stop a run as Ctrl-C
does, and a Ctrl-C that lands during the write waits until the file closes. A stop that reaches no
handler loses the rows since the last drain, about 16 s at 250 Hz, plus the block a writer thread
still has in flight. The writer sends a block within tens of milliseconds of its drain. The loop
hands it the next one only once that block is in the file, so a stalled disk costs a second block at
most. Such a stop is a SIGKILL, the kernel ending the process for lack of memory, or a crash in native
code.

## Entity paths

A row's entity path names who wrote it. It has three parts, in this order:

1. The process: `sim`, or the name of a peer's folder under `nexus_sim/_src/peers/`.
2. The component's role folder, as the source tree names it: `vehicle/sensors`,
   `vehicle/controllers` or `guidance`.
3. The instance: the name of the prim that declares it.

For example, `sim/vehicle/sensors/imu` says that the sim wrote the row, that a sensor produced it,
and that the instance is `imu`. A vehicle's Universal Scene Description (USD) file, the source tree,
and the recording read the same.

A component names only its own row, such as `horizon` or `reference`. The loop resolves the
component's path once, when it wires the component, and puts that path before the row's name.

A path says who wrote a row. It doesn't say where the row sits in space: the row's transform does.
A camera at `sim/vehicle/sensors/<name>` rides the vehicle through one static transform that names
the body's frame as its parent. The camera's pose parent reads from its transform, not from its
path.

| Path | Rows |
|---|---|
| `sim/logs/<module>` | The log records of one module |
| `sim/run/settings`, `sim/run/profile`, `sim/run/rtf` | The run's settings, its end-of-run profile and its live Real Time Factor (RTF) |
| `sim/vehicle/body` | The base body's pose, every tick |
| `sim/vehicle/body/<name>/series/<field>` | A body's recorded series |
| `sim/vehicle/joints/<name>/series/<field>` | A joint's recorded series |
| `sim/vehicle/sensors/<name>/series/<field>` | A sensor's recorded series |
| `sim/vehicle/sensors/<name>` | A camera's frames and frustum, or a lidar's points |
| `sim/vehicle/controllers/<name>/<row>` | A controller's own rows, such as its `horizon` |
| `sim/vehicle/trajectory` | The flown path |
| `sim/guidance/waypoints/wp_<i>`, `sim/guidance/reference` | The guidance's mission: its waypoints and its tracked reference |
| `sim/model/shapes/…` | The scene: each shape's mesh once, posed from the bodies' histories |
| `<peer>/logs` | A peer's log rows, under the name of its folder |
