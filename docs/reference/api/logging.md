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
`[sim/…]` lines. Components log their own *quantities*: the scene, the flown path, the guidance's
reference and waypoints, and a controller's horizon. Each logs through its own component log step,
which the central `Logger` fans out only when recording.

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
| `sim/vehicle/body` | The base body's pose, every logged tick |
| `sim/vehicle/body/<name>/series/<field>` | A body's recorded series |
| `sim/vehicle/joints/<name>/series/<field>` | A joint's recorded series |
| `sim/vehicle/sensors/<name>/series/<field>` | A sensor's recorded series |
| `sim/vehicle/sensors/<name>` | A camera's frames and frustum, or a lidar's points |
| `sim/vehicle/controllers/<name>/<row>` | A controller's own rows, such as its `horizon` |
| `sim/vehicle/trajectory` | The flown path |
| `sim/guidance/waypoints/wp_<i>`, `sim/guidance/reference` | The guidance's mission: its waypoints and its tracked reference |
| `sim/model/…`, `sim/geometry/…` | The scene that NVIDIA Newton's viewer logs |
| `<peer>/logs` | A peer's log rows, under the name of its folder |
