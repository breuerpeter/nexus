---
description: "API reference for Sim and its observation read types, BodyState and JointState: the top-level entry point for building, running, and observing a simulation."
---

# Simulation

```python
from nexus_sim import Sim, BodyState, JointState
```

A body's or a joint's history reads as `BodyState` or `JointState`. A sensor's history, through
`sim.sensors`, reads as records. `latest()` gives one row: `t`, the tick's sim time, then the
sample's fields as its type declares them, `row["accel"]` for an Inertial Measurement Unit (IMU).
`history()` gives every row as one structured array, and `arrays()` one array per quantity.

::: nexus_sim.Sim
::: nexus_sim.BodyState
::: nexus_sim.JointState
