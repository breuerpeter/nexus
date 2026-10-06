---
description: "API reference for guidance, the in-loop seam that turns a mission into a controller's setpoint: MissionGuidance and TrackingGuidance, reached through sim.guidance."
---

# Guidance

**Guidance** is the outer loop of the control cascade: each tick it turns a mission into the setpoint
the controller tracks. It's a component of the loop, for a controller that takes setpoints, and its
stage runs before the controller's, so a setpoint applies on the tick that computes it. PX4 takes
none, because its own navigator sequences its missions.

A flight constructs its guidance and hands it to `Sim.from_orchestrator`, then reaches it through
[`sim.guidance`](simulation.md). The classes are **internal**: an example imports them from
`nexus._src.guidance`, not from the top-level package.

```python
guidance = MissionGuidance(orch.controller, reached_m=0.3, final_hold_s=2.0, stop=orch.stop)
with na.Sim.from_orchestrator(orch, guidance=guidance) as sim:
    sim.guidance.set_mission([(3.0, 0.0, 2.0), (3.0, 3.0, 2.5)])
    sim.run()
```

The guidance logs its waypoint markers at `sim/guidance/waypoints/wp_<i>` and a tracked reference at
`sim/guidance/reference`.

::: nexus._src.guidance.mission.MissionGuidance
::: nexus._src.guidance.tracking.TrackingGuidance
