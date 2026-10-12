---
description: "Built on Newton, Warp and OpenUSD, nexus lets you develop drone subsystems and test them as part of the full flight stack."
hide:
  - navigation
  - toc
---

![The nexus logo: flight stack simulation with Newton](assets/lockup-banner.svg){ .hero }

Built on [Newton](https://github.com/newton-physics/newton), [Warp](https://github.com/NVIDIA/warp)
and [OpenUSD](https://openusd.org/), nexus lets you develop drone subsystems and test them as part of
the full flight stack.

## What nexus adds to Newton

<div class="grid cards" markdown>

-   **One deterministic loop**

    ---

    Each tick runs the sensors, the flight stack and Newton's step, always in that order. On a GPU the
    tick replays as CUDA graphs. On CPU a flight with the Proportional Integral Derivative (PID)
    controller repeats bit for bit, which a CI gate checks.

    [:octicons-arrow-right-24: Execution](design/execution.md)

-   **Components you can swap**

    ---

    Sensors, estimators, guidance, controllers, and force elements fill fixed roles under one contract.
    Write your own, declare it in the vehicle's Universal Scene Description (USD) file beside Newton's
    physics, and the loop runs it as it runs the shipped ones.

    [:octicons-arrow-right-24: Your own component](guide/your-own-component.md)

-   **PX4 in lockstep**

    ---

    PX4 Software In The Loop (SITL) flies the vehicle in MAVLink lockstep with the loop.
    QGroundControl and offboard scripts connect through the run's links.

    [:octicons-arrow-right-24: Running SITL flights](guide/running.md)

-   **RTX sensors from Isaac Sim**

    ---

    A vehicle that declares a camera or lidar starts an Isaac Sim container beside the loop. It renders
    each frame at the loop's poses, and the loop never runs inside Isaac Sim.

    [:octicons-arrow-right-24: RTX cameras and lidar](guide/running.md#rtx-cameras-and-lidar)

-   **One vehicle file, three uses**

    ---

    PX4 flies the vehicle's USD file in CI, Isaac Lab trains a policy on the same file, and the design
    examples take gradients through Newton from it. A catalog pins each vehicle and scene by its hash.

    [:octicons-arrow-right-24: Examples](examples/index.md)

-   **Every tick recorded**

    ---

    The Recorder keeps every row of a run, and the Logger writes it to a Rerun recording.

    [:octicons-arrow-right-24: Logging](reference/api/logging.md)

</div>

## Start here

<div class="grid cards" markdown>

-   **Guide**

    ---

    Install, fly PX4 SITL from QGroundControl, and work through the examples.

    [:octicons-arrow-right-24: Read the guide](guide/index.md)

-   **Examples**

    ---

    Design optimization with Newton's gradients, Model Predictive Control (MPC) waypoint tracking, and
    policy training in Isaac Lab.

    [:octicons-arrow-right-24: Browse examples](examples/index.md)

-   **Reference**

    ---

    The curated public API, the supported vehicles, and the changelog.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>
