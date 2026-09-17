---
description: "A sampling-plus-gradient differentiable-simulation Model Predictive Control (MPC) flying an obstacle slalom, the framework port of NVIDIA's diffsim drone."
---

# Sampling model predictive control slalom

`nexus/examples/controllers/sampling_mpc/obstacle_slalom.py` flies the astro-max through an
**obstacle slalom** with a **sampling + gradient differentiable-simulation MPC**, the framework port of
NVIDIA's `example_diffsim_drone`.

- **The regime where differentiable simulation earns its keep:** the obstacle-avoidance cost is a
  **signed-distance field** evaluated directly from the pillar geometry. That's trivial to
  differentiate through the simulator, but impractical to hand to an analytic
  Nonlinear Model Predictive Control (NMPC) solver. The landscape is non-convex: go left *or* right
  around a pillar. That's why it **samples**.
- **The method, mirroring NVIDIA:** each control tick samples 16 noisy control trajectories around
  the nominal and evaluates them in parallel as a **batched differentiable rollout**. It refines
  every one by back-propagating the obstacle-aware cost through `SolverSemiImplicit` with Adam, and
  keeps the lowest-cost trajectory. Every step is CUDA-graph-captured.
- **The task:** three waypoints staggered into a zigzag, with a pillar mid-way along each leg. The
  active waypoint advances once the drone reaches it, and the recording ends 2 s after the goal.
- The diffsim-sampling counterpart to the analytic [`acados` NMPC](mpc-acados.md).

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/sampling_mpc.rrd" width="100%" height="600" frameborder="0"></iframe>

*The drone weaves through the three pillars, advancing waypoint to waypoint: green start, gold
waypoints, red goal. The orange MPC-horizon prediction updates on each re-plan and the blue trail
grows behind it. A blank viewer means the recording isn't uploaded yet. Run
`scripts/ci/evaluate_examples.py --upload`.*

## Key stats

Configuration:

| Metric | Value |
|--------|-------|
| Method | 16 batched differentiable rollouts, sample → gradient-refine → pick the lowest cost |
| Collision cost | signed-distance field from the pillar geometry |
| Horizon | 80 steps of 10 ms, about 0.8 s, with 5 control knots |

CI-measured on the pinned runner, see [Benchmarking](../reference/benchmarking.md):

<!-- example-stats: sampling_mpc -->

## Run it

```bash
uv run -m nexus.examples sampling_mpc            # run + assert it flies the slalom
uv run -m nexus.examples sampling_mpc   # flies + asserts + writes the .rrd
```

Needs a CUDA device for the batched rollout and the graph-captured optimization.
