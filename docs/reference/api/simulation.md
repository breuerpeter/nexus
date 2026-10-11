---
description: "API reference for Sim: the top-level entry point for building, running, and observing a simulation, and the records every history reads as."
---

# Simulation

```python
from nexus_sim import Sim
```

A script reads any recorded history by indexing the run with the path of the history below the
root of the run, the same path the recording shows. `sim["vehicle/body/body_frd"]` is the base body,
`sim["vehicle/joints/rotor_1_joint"]` a joint, `sim["vehicle/sensors/imu/imu"]` the samples of a
sensor, and `sim["vehicle/controllers/px4/controls"]` the controls the controller wrote.
`for path in sim` lists them. Every history reads one way, as records. `latest()` gives one row: `t`,
the tick's sim time, then the signal's fields as its type declares them. The row of an
Inertial Measurement Unit (IMU) holds `row["accel"]`, a body's `row["position"]`, `row["orientation"]`,
`row["linear_velocity"]` and `row["angular_velocity"]`, and a joint's `row["q"]` and `row["qd"]`. `history()` gives
every row as one structured array, `rows["position"][:, 2]` the heights over the run, and `arrays()`
one array per quantity.

::: nexus_sim.Sim

The scripts share one argument parser, and save a run's artifacts with one call:

::: nexus_sim.sim_argparser
::: nexus_sim.save_run_artifacts
