---
description: "API reference for the Operator plane: the InProcessOperator, reached through sim.operator, plus the wait_until helper."
---

# Operator

The **Operator** plane, Plane 5, commands the vehicle. It reads input and converts it to autopilot
commands. The input comes from a script or API call now, and from a joystick or Ground Control
Station (GCS) later. Operators are **internal**: reach the one driving a run through
[`sim.operator`](simulation.md), not a top-level import. PX4 takes no operator: a script commands
it over its offboard link with [`nexus.px4.OffboardClient`](px4.md).

```python
with na.Sim.from_orchestrator(orch) as sim:  # a setpoint controller, so the operator is InProcessOperator
    sim.operator.set_mission([(0.0, 0.0, 2.0)])
    sim.run()
```

::: nexus._src.operator.operator.Operator
::: nexus._src.operator.in_process.InProcessOperator
::: nexus._src.operator.operator.wait_until
