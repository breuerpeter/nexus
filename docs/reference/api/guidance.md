---
description: "API reference for guidance, the role that turns a mission into a controller's setpoint: MissionGuidance and TrackingGuidance, reached through sim.guidance."
---

# Guidance

**Guidance** is the outer loop of the control cascade: each tick it turns a mission into the setpoint
the controller tracks. It's a component of the loop, for a controller that takes setpoints, and its
stage runs before the controller's, so a setpoint applies on the tick that computes it. A guidance
holds no controller and no stop. Its stage writes a changed setpoint to the tick, and the loop hands
that to the controller. When the mission is over, the stage marks the tick done, and the loop ends
the run. PX4 takes none, because its own navigator sequences its missions.

A flight constructs its guidance from the guidance's own parameters and hands it to
`Sim.from_orchestrator`, then reaches it through
[`sim.guidance`](simulation.md). The classes are **internal**: an example imports them from
`nexus_sim._src.guidance`, not from the top-level package.

```python
guidance = MissionGuidance(reached_m=0.3, final_hold_s=2.0)
with nx.Sim.from_orchestrator(orch, guidance=guidance) as sim:
    sim.guidance.set_mission([(3.0, 0.0, 2.0), (3.0, 3.0, 2.5)])
    sim.run()
```

The guidance logs its waypoint markers at `sim/guidance/waypoints/wp_<i>` and a tracked reference at
`sim/guidance/reference`.

::: nexus_sim._src.guidance.mission.MissionGuidance
::: nexus_sim._src.guidance.tracking.TrackingGuidance
