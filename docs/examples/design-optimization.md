---
description: "Back-propagate through a closed-loop flight to tune a Proportional Integral Derivative (PID) controller and recover a mass, using nexus's differentiable simulation."
---

# Differentiable design optimization

`nexus/examples/design_opt/` back-propagates through a differentiable single-body simulation to
**tune a controller**, in `gain_tuning.py`, and **recover a physical parameter**, in
`mass_recovery.py`.

- **On the real vehicle, the headline:** gradient descent on the gains of the built-in PID flies
  the production astro-max to a far waypoint and makes it settle. That vehicle is the multi-body
  Universal Scene Description (USD) model with the per-rotor actuator kernel and
  `SolverFeatherstone`. **Truncated Back-Propagation Through Time (BPTT)** bounds the long-horizon
  closed-loop back-propagation, because the articulated closed loop explodes a single full-horizon
  tape.
- **Exact-gradient and mass-recovery gates, on a lumped single body with `SolverSemiImplicit`:**
  the gradient matches finite differences, with cosine ≈ 1.0, and recovers a known mass to ~0.03 %.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/gain_tuning.rrd" width="100%" height="600" frameborder="0"></iframe>

*Both flights share one timeline, with the two paths overlaid. The **mis-tuned baseline** crawls,
never settles, and ends ~1.4 m short of the waypoint. Then the **tuned** controller flies in,
settles in ~2.2 s with no overshoot, and holds. The rotors spin, as a visual. A blank viewer means
the recording isn't uploaded yet: run `scripts/ci/evaluate_examples.py --upload`.*

```bash
uv run -m nexus.examples gain_tuning            # tunes + flies + writes the .rrd
uv run -m nexus.examples mass_recovery          # assert the system-ID mass recovery
```

## Measured

Gain tuning:

<!-- example-stats: gain_tuning -->

Mass recovery:

<!-- example-stats: mass_recovery -->

Every `gpu-examples` CI run gates both. See [Benchmarking](../reference/benchmarking.md).
