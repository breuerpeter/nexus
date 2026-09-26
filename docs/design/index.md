---
description: "What nexus is, the modular-component model behind it, and the foundational design decisions: single physics engine, one loop with rendering as a peer, Universal Scene Description (USD) as the single authority."
---

# Design

nexus is a **modular simulation core** for the Freefly PX4-based drones, Astro and Alta X,
built on NVIDIA Newton physics. Every functional concern, whether physics, sensors, actuators,
controller, scene, environment, or renderer, is a component behind a typed interface. You can swap
each one independently and select it through declarative configuration. It starts from a minimal
working slice, Newton dynamics ↔ PX4 Software In The Loop (SITL) with Rerun observability, and
grows by **adding components, never by rewriting**.

This section explains *how the framework works and why*. The [Guide](../guide/index.md) shows how
to *use* it. The [Reference](../reference/index.md) documents the *surface*.

<div class="grid cards" markdown>

-   **[Architecture](architecture.md)**: the data model, the typed component interfaces, the deterministic tick, and the operator, recording, and configuration seams.
-   **[Execution &amp; determinism](execution.md)**: eager, CUDA-graph-captured, and differentiable rollouts, what makes runs bit-reproducible, and where they're only tolerance-bounded.
-   **[Real system ↔ simulation](real-vs-sim.md)**: the six subsystem planes and the boundary where real drone code stops and the sim takes over.
-   **[Conventions](conventions.md)**: where a seam lives, which seams a vehicle or a scene declares, and the vehicle model's frames, rotor indexing, and structure.

</div>

## Why nexus exists

Freefly builds its own ground stack for PX4 drones: a custom Ground Control Station (GCS), a
companion computer, and cloud services. Developing and testing that stack with speed and
confidence, and moving toward data-driven autonomy, needs a simulation capability Freefly **owns
and controls**. A closed product, or a stack built around the wrong autopilot, doesn't give that.
Simulation lowers the risk and cost of real flights, gives reproducible environments, and enables
automated testing in CI. It's also the bet on data-driven autonomy: the sim is what spins up the
data flywheel.

## Three regimes, one core

The same component interfaces serve three execution regimes:

| Regime | What runs | Used for |
|---|---|---|
| **Online PX4 SITL** | Real PX4 firmware flies the vehicle over a MAVLink lockstep link | Automated SITL testing, operator training, customer preview |
| **In-process differentiable** | A built-in Proportional Integral Derivative (PID) controller computes controls in-process. The whole rollout records a gradient | [Design optimization](../examples/design-optimization.md) |
| **Trained policy** | The framework loads a network trained on Isaac Lab as a Controller | [Reinforcement learning](../examples/isaac-lab-rl.md) deployment |

RL *training* itself runs in the separate `nexus-rl` app on Isaac Lab
over the same Newton physics. The framework consumes only its exported policy.

## Foundational decisions

These stances shape everything else and are load-bearing across the codebase:

- **A single physics engine, NVIDIA Newton on Warp.** No solver zoo, no second backend. The
  modularity that matters is sensors, controllers, scene, and renderer, not physics engines.
  The same Newton dynamics run everywhere.
- **One loop, rendering as a peer.** Every run's loop runs in one process on the host, rendered or
  not, and it's the authority for CI and determinism. A vehicle that authors RTX sensors renders
  them in the **Kit render peer**, a container holding Isaac Sim, Cesium worlds, and no nexus code.
  The host sends it poses, takes back frames, and streams the video itself. No nexus module imports
  Isaac Sim or Omniverse.
- **Neutral USD is the single vehicle and scene authority.** A vehicle is one OpenUSD model,
  authored once, carrying geometry, mass, rotor joints, motor and propeller parameters, and the
  whole sensor suite. The loop, the Kit peer and the training app consume it unchanged. The framework
  **only ever builds models from USD** and never synthesizes one in code. See
  [Conventions](conventions.md).
- **RL is a separate consumer.** `nexus-rl` depends on the framework, not the reverse. The
  framework consumes a trained policy through the ordinary [Controller](architecture.md#controllers-and-the-operator-plane)
  interface and never depends on Isaac Lab.
- **Determinism is foundational.** Given a scenario, seed, and pinned code, a run is
  bit-reproducible, the property that underpins CI testing and bug reproduction. See
  [Execution &amp; determinism](execution.md).
