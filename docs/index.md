---
description: "nexus, a GPU-accelerated drone simulation framework on NVIDIA Newton, built for fast, batched, and differentiable flight simulation."
hide:
  - navigation
  - toc
---

![nexus: GPU-accelerated drone simulation](assets/lockup-banner.svg){ .hero }

*nexus* is a GPU-accelerated drone simulation framework on NVIDIA Newton and Omniverse,
built for fast, batched, and differentiable flight simulation.

The framework provides the building blocks to design, test, and assess a whole drone
system in simulation. Compose a vehicle from a reusable, unit-tested library of sensors
and actuators, such as an Inertial Measurement Unit (IMU), propeller wrenches, and state
views. Then fly it against a real PX4 flight stack in decoupled Software In The Loop (SITL)
lockstep, with a determinism gate and content-addressed configs that keep every run
reproducible. And because the physics is differentiable, you can optimize designs with
gradients and train controllers with reinforcement learning on Isaac Lab. You can also render
photorealistic images, with the same model running everywhere from headless CI to Isaac Sim.

NVIDIA Warp, the GPU compute framework that Newton builds on, uses Just In Time (JIT)
compilation to turn the simulation's plain Python kernels into C++ and CUDA. Every step
dispatches many such kernels: on an NVIDIA GPU they run in parallel across its
cores. The same kernels also run on CPU unchanged, but serially, so CPU is a slower
fallback.

<div class="grid cards" markdown>

-   **Guide**

    ---

    Install, run SITL flights, fly the sim from real hardware, and work through the examples.

    [:octicons-arrow-right-24: Read the guide](guide/index.md)

-   **Examples**

    ---

    Differentiable design optimization, Model Predictive Control (MPC) waypoint tracking, and
    Isaac Lab RL.

    [:octicons-arrow-right-24: Browse examples](examples/index.md)

-   **Reference**

    ---

    The curated public API, the supported vehicles, and the changelog.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>
