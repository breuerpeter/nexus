---
description: "The measured performance of nexus: the Real Time Factor (RTF) matrix across vehicles, scenes and devices, per-example metrics, and how the CI regression gates keep the numbers honest."
---

# Benchmarking

CI **measures every number on this page on pinned hardware** and renders it from committed
data under `docs/data/`. Nothing here is hand-written. The data changes only through reviewed
`chore(data): refresh …` pull requests that the benchmark workflows open, so a regression, or an
improvement, is always a visible diff.

## What CI measures

**Steady Real Time Factor (RTF)**: simulated seconds per wall-clock second. The orchestrator
measures it over the whole run *excluding* the first 250 control steps, where lazy kernel
compilation and initial settle happen. Whole-run wall time is never used as a speed metric.
Alongside the steady figure, the loop profiler records an RTF **window series** of 1000-tick
windows, first window dropped, whose min / avg / max / σ show how stable a run is.

**The benchmark mission**: one PX4 flight, the same for every matrix cell. It
arms, runs `AUTO.TAKEOFF`, then flies a 4-waypoint `DO_REPOSITION` mission with confirmed arrival
at each waypoint. The script `scripts/ci/benchmark_cell.py` flies it against the `Px4Offboard`
operator. PX4 Software In The Loop (SITL) runs in the loop, in lockstep on `:4560`, so these
figures come from a closed loop, not from physics-only throughput. All cells record their flight
with `--log` on: recording is part of the measured configuration.

## Real-time factor matrix

<!-- benchmark-matrix -->

The scheduled `gpu-benchmark-matrix` workflow produces the matrix on CI, each cell on a g5.2xlarge
box of its own with one A10G, and refreshes it via a data PR. Cells that structurally don't exist aren't shown.
A visual-only scene changes nothing for a vehicle that renders nothing, since physics loads only
`UsdPhysics`-authored geometry, so the physics table flies the empty scene alone. In the RTX table
the vehicle's camera renders in the Kit peer container beside the loop. CPU is the bit-exact
determinism authority rather than a performance configuration, so it's displayed only in the
physics reference row.

## Per-example performance

Every `gpu-examples` CI run scores each [example](../examples/index.md) on flight quality and
steady RTF. Flight quality is the `evo` pose Absolute Pose Error (APE) against the example's
reference, plus the example's own stats:

<!-- benchmark-examples -->

`sampling_mpc` is exempt from the RTF ≥ 1 doctrine by design, since it runs 16 parallel rollouts
of an 80-step horizon per control tick. Its gate is its recorded value instead.

## The regression gates

The same harness that produces these numbers gates them.
[`scripts/ci/evaluate_examples.py`](https://github.com/breuerpeter/nexus/blob/main/scripts/ci/evaluate_examples.py)
runs every example and compares each metric against the committed
`scripts/ci/examples_baselines.json`. A bare value gates exactly, `min`/`max` gate absolutely,
and `min_frac`/`max_frac` gate relative to the recorded baseline value once one exists. Baselines
are **hardware-pinned to the A10G runner**, and only a deliberate `--update-baselines` dispatch
re-records them. A metric without a baseline entry is report-only: the harness shows it and never
gates on it.

Beyond the gates, every upload also appends to a per-commit trend feed, `bench/<sha>.json` in the
CI artifacts bucket in github-action-benchmark custom JSON. The feed is raw, with no dashboard.

## Reading numbers elsewhere

Example pages embed their own generated stat blocks from the same data. Prose figures quoted in
design discussions, for example the dev-workstation RTFs in
[Execution & determinism](../design/execution.md), name the hardware they ran on and are
illustrative, not tracked.
