---
description: "API reference for the signal value types a component reads or writes: the tick's time, the estimate, the setpoints and each shipped sensor's sample."
---

# Signal types

```python
from nexus_sim import SimTime, PoseTwist, PositionGoal, Waypoints, ReferenceTrajectory
from nexus_sim import ImuSample, MagSample, BaroSample, GpsSample, Image, PointCloud
```

A [signal](../../design/concepts.md#signal) carries one of these types between components, and a
component's stage declares its signals with them: `Signal("estimate", PoseTwist, shape=(1,))`. A
Warp struct lives on the device, and a host stage reads it as a record, `row["accel"]`. The controls
a controller writes are a plain device row of `wp.float32`, one per actuator, so they have no type here.

::: nexus_sim.SimTime
::: nexus_sim.PoseTwist
::: nexus_sim.PositionGoal
::: nexus_sim.Waypoints
::: nexus_sim.ReferenceTrajectory
::: nexus_sim.ImuSample
::: nexus_sim.MagSample
::: nexus_sim.BaroSample
::: nexus_sim.GpsSample
::: nexus_sim.Image
::: nexus_sim.PointCloud
