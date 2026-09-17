---
description: "The minimal in-process controller example: a deterministic, differentiable Proportional Integral Derivative (PID) controller flying a waypoint square, fully CUDA-graph captured."
---

# PID waypoint tour

The smallest end-to-end example of the in-process controller shape: a deterministic,
differentiable PID flies the collapsed single-body Astro Max around a square of waypoints.
One CUDA graph captures the whole tick: controller, actuator, physics, and sensors.
It deploys the gains the [design-optimization example](design-optimization.md) converges to.
It also doubles as the framework's **determinism authority**: on CPU it runs eager and bit-exact,
the bit-reproducible CI gate a real PX4-in-the-loop flight can't give.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/pid.rrd" width="100%" height="600" frameborder="0"></iframe>

*The square tour with the active waypoint advancing on arrival. A blank viewer means the
recording isn't uploaded yet. Run `scripts/ci/evaluate_examples.py --upload`.*

## Run it

```bash
uv run -m nexus.examples pid                # flies + asserts + writes the .rrd
uv run nexus script nexus.examples pid # the same, RTX-rendered under Kit
```

Zero-arg by design: the configuration, vehicle, waypoints, and the proven gains, lives in the
script. The example asserts the drone reaches every waypoint and dumps its trajectory for the CI
harness.

## Measured

<!-- example-stats: pid -->

## Regression gate

`scripts/ci/evaluate_examples.py` flies this example on every `gpu-examples` CI run and gates it
against `scripts/ci/examples_baselines.json`. The gate checks that every waypoint counts as
`reached`, an exact match, and that the final distance and pose Absolute Pose Error (APE) are
within bounds. It also checks that the steady Real Time Factor (RTF) stays within a fraction of its
recorded baseline. See [Benchmarking](../reference/benchmarking.md) for how the harness measures and
refreshes the numbers.
