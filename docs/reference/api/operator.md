---
description: "API reference for the PX4 operator: the Px4Offboard client, reached through sim.operator, plus the wait_until helper."
---

# Operator

The **operator** commands a PX4 run from the ground side, over MAVLink, as a ground station does.
It's **internal**: reach the one that commands a run through [`sim.operator`](simulation.md), not a
top-level import. A controller that takes setpoints has no operator. Its
[guidance](guidance.md) runs in the loop.

```python
with na.Sim("astro_max_base", scene="empty") as sim:  # the vehicle declares PX4, so the operator is Px4Offboard
    sim.start()
    sim.operator.takeoff(2.0)
```

::: nexus._src.operator.px4_offboard.Px4Offboard
::: nexus._src.operator.operator.wait_until
