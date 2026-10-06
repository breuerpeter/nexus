---
description: "The public API of nexus, from Sim and Orchestrator to the rest, generated from the source with typed signatures."
---

# API reference

The complete public API of `nexus`. Everything documented here is importable directly from the
top-level package, for example `from nexus import Sim`, except PX4's own names, which sit in a
namespace beside it, `nexus.px4`. The `_src` layout is an implementation detail and **isn't** part
of the public surface. Import from `nexus`, never from `nexus._src`.

<div class="grid cards" markdown>

-   **[Simulation](simulation.md)**: `Sim`, the top-level entry point for building and running a simulation, and the `BodyState`, `JointState` and `SensorSample` values it reports.
-   **[Core types](core.md)**: `Orchestrator`, `Controls`, `Measurement`, and `SimTime`, the runtime-agnostic state, control, and observation surfaces.
-   **[Configuration](configuration.md)**: `LaunchConfig` for the launch configuration and `Registry` for the vehicle and scene catalog.
-   **[Guidance](guidance.md)**: the in-loop seam that turns a mission into a controller's setpoint, reached via `sim.guidance`. Its classes are internal, so the page is the one exception to importing from `nexus`.
-   **[PX4](px4.md)**: `OffboardClient`, the client a script opens on PX4's offboard link, and the mission plan types, imported from `nexus.px4`.
-   **[Logging](logging.md)**: `logger`, the framework logger, which writes to the console and tees into the Rerun recording.

</div>
