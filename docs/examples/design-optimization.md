---
description: "Back-propagate through a closed-loop flight to tune a Proportional Integral Derivative (PID) controller and recover a mass, through Newton's differentiable simulation."
---

# Differentiable design optimization

`nexus_uav/examples/design_opt/` back-propagates through a differentiable simulation of the vehicle
every run flies to **tune a controller**, in `gain_tuning.py`, and **recover a physical parameter**, in
`mass_recovery.py`.

- **The plant PX4 flies:** each rollout builds the articulated astro-max from its Universal Scene
  Description (USD) file. The Newton motors the file authors turn its rotor joints, and the core's
  rotor command and propellers drive them, under `SolverSemiImplicit` at 16 solver steps per 4 ms
  control tick. The rollout steps one copy of each motor per solver step, so no step overwrites what
  another step's backward pass reads, and each gradient replays from one captured CUDA graph.
- **Gain tuning:** gradient descent on the gains of the example PID, through the whole closed-loop flight,
  flies the vehicle to a far waypoint and makes it settle. The [PID example](pid.md) flies the gains it
  converges to. The gradient matches central finite differences, with cosine ≈ 1.0.
- **Mass recovery:** gradient descent on the mass of the airframe, through an open-loop climb, recovers a
  known mass to ~0.03 %.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/gain_tuning.rrd" width="100%" height="600" frameborder="0"></iframe>

*Both flights share one timeline, with the two paths overlaid. The **mis-tuned baseline** never
settles and comes no closer than ~1.7 m to the waypoint. Then the **tuned** controller flies in,
settles with no overshoot, and holds within ~0.07 m. Each prop turns with its simulated rotor joint.
A blank viewer means the recording isn't uploaded yet: run `scripts/ci/evaluate_examples.py --upload`.*

```bash
uv run -m nexus_uav.examples gain_tuning            # tunes + flies + writes the .rrd
uv run -m nexus_uav.examples mass_recovery          # assert the system-ID mass recovery
```

## Measured

Gain tuning:

<!-- example-stats: gain_tuning -->

Mass recovery:

<!-- example-stats: mass_recovery -->

Every `gpu-examples` CI run gates both. See [Benchmarking](../reference/benchmarking.md).
