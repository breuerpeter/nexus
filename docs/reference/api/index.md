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
-   **[Core types](core.md)**: `Orchestrator`, `Controls`, `Measurement`, `EnvSample`, and `SimTime`, the runtime-agnostic state, control, and observation surfaces.
-   **[Configuration](configuration.md)**: `LaunchConfig` for the launch configuration and `Registry` for the component registry.
-   **[Operator](operator.md)**: `Operator`, `InProcessOperator`, `Px4Offboard`, and `wait_until`, the operator plane, Plane 5, reached via `sim.operator`.
-   **[Logging](logging.md)**: `log_event`, structured event logging into the Rerun recording.

</div>
