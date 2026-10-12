---
description: "What nexus is and what it adds to Newton, the component model behind it, and the foundational design decisions: one physics engine, one loop with rendering as a peer, and Universal Scene Description (USD) as the single authority."
---

# Design

Built on Newton, Warp and OpenUSD, nexus lets you develop drone subsystems and test them as part of
the full flight stack. Each part of the flight stack is a [component](concepts.md#component) in a
fixed [role](concepts.md#role), and so is each part of the plant it flies except the physics. The
vehicle's Universal Scene Description (USD) file declares them all. A process outside the loop, such
as PX4 Software In The Loop (SITL), joins the run as a [peer](concepts.md#peer). The framework grows
by **adding components and peers, never by rewriting**.

This section explains *how the framework works and why*. The [Guide](../guide/index.md) shows how
to *use* it. The [Reference](../reference/index.md) documents the *surface*.

<div class="grid cards" markdown>

-   **[Concepts](concepts.md)**: every term the framework uses, each defined once, in the words the code uses.
-   **[Architecture](architecture.md)**: the data model, the typed component interfaces, the deterministic tick, the guidance, the recording, and the configuration.
-   **[Execution &amp; determinism](execution.md)**: eager, CUDA-graph-captured, and differentiable rollouts, what makes runs bit-reproducible, and where they're only tolerance-bounded.
-   **[Real system ↔ simulation](real-vs-sim.md)**: the six subsystem planes and the boundary where real drone code stops and the sim takes over.
-   **[Conventions](conventions.md)**: where a contract lives, who declares each part, how a peer enters a run, the project's names, and the vehicle model's frames, rotor indexing, and structure.

</div>

## Why nexus exists

A drone system is many subsystems: the estimator, control laws and guidance of its flight controller,
its sensors and propellers, and the Ground Control Station (GCS) that commands it. You develop each
one on its own, but it counts only once it works as part of the whole system. Real flights make that
test slow, costly, and hard to repeat. In nexus, the subsystem under test flies inside a full flight
stack against Newton's physics: PX4 and its own estimator, or an estimator, guidance, and controller
of your own. The test is cheap, repeats, and runs in CI on every change.

## One loop, any controller

The same loop flies each controller: PX4, the example Proportional Integral Derivative (PID)
controller, or a trained policy.

| Controller | What runs | Used for |
|---|---|---|
| **PX4** | Real PX4 firmware flies the vehicle over a MAVLink lockstep link | SITL testing in CI and flight-stack development |
| **PID** | The example PID controller computes controls on the device, and a Warp tape records the whole rollout for its gradient | [Design optimization](../examples/design-optimization.md) |
| **Trained policy** | The example policy controller runs a network trained in Isaac Lab | [Reinforcement learning](../examples/isaac-lab-rl.md) deployment |

Policy *training* runs in the separate `nexus-rl` app, on Isaac Lab over Newton. The framework
consumes only its exported policy.

## Foundational decisions

These stances shape everything else and are load-bearing across the codebase:

- **One physics engine, Newton on Warp.** No second backend. The modularity that matters is the
  flight stack, the sensors, the scene and the peers, not physics engines. One vehicle USD feeds
  every use.
- **One loop, rendering as a peer.** Every run's loop runs in one process on the host, rendered or
  not, and it's the authority for CI and determinism. A vehicle that authors RTX sensors renders
  them in the **Kit render peer**, a container holding Isaac Sim, Cesium worlds, and the peer's own
  program, which imports no nexus module.
  The host sends it poses and takes back frames. No nexus module imports Isaac Sim or Omniverse.
- **Neutral USD is the single vehicle and scene authority.** A vehicle is one OpenUSD model,
  authored once, carrying geometry, mass, rotor joints, motor and propeller parameters, and the
  whole sensor suite. The loop, the Kit peer and the training app consume it unchanged. The framework
  **only ever builds models from USD** and never synthesizes one in code. See
  [Conventions](conventions.md).
- **RL is a separate consumer.** `nexus-rl` depends on the framework, not the reverse. The
  framework flies a trained policy as it flies any other controller, a
  [component](concepts.md#component) on the one contract, and never depends on Isaac Lab.
- **Determinism is foundational.** Given a scenario, seed, and pinned code, a run is
  bit-reproducible, the property that underpins CI testing and bug reproduction. See
  [Execution &amp; determinism](execution.md).
