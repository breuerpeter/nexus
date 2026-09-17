---
description: "API reference for the Operator plane: the InProcessOperator and the PX4 Px4Offboard, reached through sim.operator, plus the wait_until helper."
---

# Operator

The **Operator** plane, Plane 5, commands the vehicle. It reads input and converts it to autopilot
commands. The input comes from a script or API call now, and from a joystick or Ground Control
Station (GCS) later. Operators are **internal**: reach the one driving a run through
[`sim.operator`](simulation.md), not a top-level import.

```python
with na.Sim("astro_max_base", control="policy", policy="policy.pt") as sim:
    sim.operator.set_mission([(1.5, 1.0, 1.5), (-1.5, 1.0, 2.0)])  # InProcessOperator
    sim.run()
```

::: nexus._src.operator.operator.Operator
::: nexus._src.operator.in_process.InProcessOperator
::: nexus._src.operator.px4_offboard.Px4Offboard
::: nexus._src.operator.operator.wait_until
