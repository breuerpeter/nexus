---
description: "Differentiable design optimization, Model Predictive Control (MPC) waypoint tracking, and Isaac Lab RL: runnable examples on the framework, plus the Isaac Lab training app."
---

# Examples

Runnable examples on the framework, plus the Isaac Lab RL app, a separate `uv` project. Launch any
with `uv run -m nexus.examples <name>`, and `--list` shows them all. Each example is zero-arg: its
configuration lives in the script. Each records an interactive [Rerun](https://rerun.io) `.rrd` of
its flight and needs a CUDA device. A vehicle that authors RTX sensors renders them in the Kit
render peer, which the example starts on its own.

<div class="grid cards" markdown>

-   **[PID waypoint tour](pid.md)**: the minimal in-process controller, a deterministic, differentiable Proportional Integral Derivative (PID) controller flying a waypoint square, fully CUDA-graph captured.
-   **[Differentiable design optimization](design-optimization.md)**: back-propagate through a closed-loop flight to tune a PID and recover a mass.
-   **[`acados` NMPC](mpc-acados.md)**: a real-time Nonlinear Model Predictive Control (NMPC) controller flying a curving waypoint course nose-first along a min-snap reference.
-   **[Sampling MPC slalom](mpc-sampling.md)**: a sampling-plus-gradient differentiable-simulation MPC flying an obstacle slalom.
-   **[Astro Max RL on Isaac Lab](isaac-lab-rl.md)**: a swarm of parallel agents learning to hover with Proximal Policy Optimization (PPO), then the policy deployed on the core runtime.
-   **[PX4 SITL flight through a conformed actuator](px4-sitl.md)**: a PX4 Software In The Loop (SITL) flight through a conformed per-rotor actuator, with the real-time factor reported.

</div>
