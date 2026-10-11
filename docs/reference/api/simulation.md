---
description: "API reference for Sim and its observation read types, BodyState and JointState: the top-level entry point for building, running, and observing a simulation."
---

# Simulation

```python
from nexus_sim import Sim, BodyState, JointState
```

A script reads any recorded history by indexing the run with the path of the history below the
root of the run, the same path the recording shows. `sim["vehicle/body/body_frd"]` is the base body,
`sim["vehicle/joints/rotor_1_joint"]` a joint, `sim["vehicle/sensors/imu/imu"]` the samples of a
sensor, and `sim["vehicle/controllers/px4/controls"]` the controls the controller wrote.
`for path in sim` lists them. A body's or a joint's history reads as `BodyState` or `JointState`. A
signal's history reads as records: `latest()` gives one row, `t`, the tick's sim time, then the
sample's fields as its type declares them, `row["accel"]` for an Inertial Measurement Unit (IMU).
`history()` gives every row as one structured array, and `arrays()` one array per quantity.

::: nexus_sim.Sim
::: nexus_sim.BodyState
::: nexus_sim.JointState

The scripts share one argument parser, and save a run's artifacts with one call:

::: nexus_sim.sim_argparser
::: nexus_sim.save_run_artifacts
