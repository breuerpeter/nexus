---
description: "API reference for Sim: the top-level entry point for building, running, and observing a simulation, and the records every history reads as."
---

# Simulation

```python
from nexus_uav import Sim
```

A script reads any recorded history by indexing the run with the path of the history below the
root of the run, the same path the recording shows. `sim["vehicle/body/body_frd"]` is the base body,
`sim["vehicle/joints/rotor_1_joint"]` a joint, `sim["vehicle/sensors/imu/imu"]` the samples of a
sensor, and `sim["vehicle/controllers/px4/controls"]` the controls the controller wrote.
`for path in sim` lists them. Every history reads one way, as records. `history()` gives every row
as one structured array, oldest first: `t`, the tick's sim time, then the signal's fields as its type
declares them. `rows["position"][:, 2]` holds the heights over the run, and `history()[-1]` is the
newest row. The row of an Inertial Measurement Unit (IMU) holds `row["accel"]`, a body's
`row["position"]`, `row["orientation"]`, `row["linear_velocity"]` and `row["angular_velocity"]`, and a
joint's `row["q"]` and `row["qd"]`. `arrays()` gives one array per quantity.

A script reads a signal's live value by the same path, typed by the signal:
`sim.read("vehicle/body/body_frd").position`. A struct signal reads as one instance of its type, and
a value signal as an array without its leading axis of one. A host signal, such as a camera's image,
reads as the object its writer wrote, though it keeps no history. `sim.time` is the sim time of the
last tick.

::: nexus_uav.Sim

The scripts share one argument parser, and save a run's artifacts with one call:

::: nexus_uav.sim_argparser
::: nexus_uav.save_run_artifacts
