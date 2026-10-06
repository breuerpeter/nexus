---
description: "The public API of nexus, from Sim and Orchestrator to the rest, generated from the source with typed signatures."
---

# API reference

The complete public API of `nexus`. Everything documented here is importable directly from the
top-level package, for example `from nexus import Sim`. The `_src` layout is an implementation
detail and **isn't** part of the public surface. Import from `nexus`, never from
`nexus._src`.

<div class="grid cards" markdown>

-   **[Simulation](simulation.md)**: `Sim` and `SimState`, the top-level entry point for building and running a simulation.
-   **[Core types](core.md)**: `Orchestrator`, `Controls`, `Measurement`, and `SimTime`, the runtime-agnostic state, control, and observation surfaces.
-   **[Configuration](configuration.md)**: `LaunchConfig` for the launch configuration and `Registry` for the vehicle and scene catalog.
-   **[Guidance](guidance.md)**: `MissionGuidance` and `TrackingGuidance`, the in-loop seam that turns a mission into a controller's setpoint, reached via `sim.guidance`.
-   **[Operator](operator.md)**: `Px4Offboard` and `wait_until`, the client that commands PX4 from the ground side, reached via `sim.operator`.
-   **[Logging](logging.md)**: `log_event`, structured event logging into the Rerun recording.

</div>
