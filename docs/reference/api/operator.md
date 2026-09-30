---
description: "API reference for the Operator plane: the InProcessOperator and the PX4 Px4Offboard, reached through sim.operator, plus the wait_until helper."
---

# Operator

The **Operator** plane, Plane 5, commands the vehicle. It reads input and converts it to autopilot
commands. The input comes from a script or API call now, and from a joystick or Ground Control
Station (GCS) later. Operators are **internal**: reach the one driving a run through
[`sim.operator`](simulation.md), not a top-level import.

```python
with na.Sim("astro_max_base", scene="empty") as sim:  # the vehicle declares PX4, so the operator is Px4Offboard
    sim.start()
    sim.operator.takeoff(2.0)
```

::: nexus._src.operator.operator.Operator
::: nexus._src.operator.in_process.InProcessOperator
::: nexus._src.operator.px4_offboard.Px4Offboard
::: nexus._src.operator.operator.wait_until
