---
description: "A PX4 Software In The Loop (SITL) flight on the newton.actuators rotor motors, DC-motor servos authored in Universal Scene Description (USD) on real rotor joints, with the with-PX4 real-time factor reported."
---

# PX4 software-in-the-loop flight on `newton.actuators` rotor motors

The Astro Max flown **closed-loop by PX4** over the decoupled SITL link, on the core
**`ArticulatedRotors`** actuator. Each rotor motor is a USD-authored `newton.actuators` composition:
a `NewtonActuator` prim on the real rotor joint, made of a `ControllerPID` velocity servo and the
`ClampingDCMotor` four-quadrant envelope. So the rotor speed Ω is a solver-integrated joint state
with physical lag and saturation, and the props spin for real. The aerodynamics stay a Warp `body_f`
kernel over that solver-integrated Ω: airflow-aware thrust, H-force, and reaction drag, from the
`propeller:*` attributes. PX4 runs its real flight stack. The only thing it talks to is the
Hardware In The Loop (HIL) link on `:4560`.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/px4_sitl.rrd" width="100%" height="600" frameborder="0"></iframe>

## Key stats

| Metric | Value |
| --- | --- |
| Flight | PX4 closed-loop: `AUTO.TAKEOFF` → arm → climb → 4 yaw sweeps, the one flight profile |
| Takeoff | **confirmed**, climbed **+5.0 m**, yawed 90/180/270/0° |
| PX4 warnings | **0**, gated: the run fails on any `WARN` or `ERROR` line about the sim |
| Actuator | `ArticulatedRotors`, with USD-authored `newton.actuators` DC-motor servos on the real rotor joints, one `NewtonActuator` prim each, plus `propeller:*` aero |
| **Sim speed with PX4** | **~3.7×** real-time on the GPU with the captured strategy, RTX 5080, for the full takeoff+yaw profile with recording on |

**Clean log gate.** The flight asserts **zero PX4 warnings**, the `px4_warnings` metric gated to 0.
A healthy closed-loop SITL run must not log a single `WARN` or `ERROR` line about the sim. Two
known-benign lines are content-allowlisted. One is the boot-time `Parameter <name> not found.` line
for a parameter the airframe sets and this PX4 build lacks, and every other `param` failure still
gates. The other is the pre-arm "no heading reference" transient while PX4's EKF2 estimator
converges. The takeoff
script's failure-free wait rides that transient out. This caught a real fidelity bug. The environment
fed the magnetometer a *Zurich* World Magnetic Model (WMM) field while the
Global Positioning System (GPS) origin is *Seattle*, so PX4's strict mag check, `EKF2_MAG_CHK_STR`,
intermittently failed with `Strong magnetic interference`. The fix computes the field at the GPS
origin from PX4's own coarse WMM table, `nexus._src.core.geomag`, ported from PX4's
`geo_mag_declination.cpp` as PegasusSimulator and `PX4-SITL_gazebo` do, plus a realistic
magnetometer σ. The warning is now gone and arming is faster.

**DC-motor servos are effectively free.** The `newton.actuators` motor step plus aero kernel cost ~2%
Real Time Factor (RTF) versus the old lumped first-order-lag wrench model. The mujoco-warp solver
dominates the step. The fidelity gain is real: the rotor speed is a solver-integrated joint state, and
the props spin physically in every renderer. The tracking error of the `acados` Nonlinear Model
Predictive Control (NMPC) example **halved** when it switched to the same actuator.

The sim speed is the **real-time factor of the flight itself**: sim-time advanced divided by wall-time,
with PX4 in the lockstep loop. The orchestrator reports it on exit. The loop runs on the
**GPU** under the **captured execution strategy**, which `--device auto` selects automatically: a
CUDA-graph replay of the device region, with the PX4 MAVLink seam at the host boundary. The per-tick
budget is roughly **53% GPU step**, **20% PX4 MAVLink exchange**, and **27% `.rrd` logging**. The GPU
step is mujoco-warp at `batch=1`: a single drone is latency-bound, not throughput-bound, so this is the
floor for one env on this solver. Logging decimates to 50 Hz. Logging the full scene at the 250 Hz sim
rate was the dominant cost before and capped the loop at ~2.3×. For reference, the earlier CPU-eager
default ran at ~1.03×. The pure in-process loop has no host seam. It captures the whole tick and runs
far faster still, ~19× on `quad_x`. Pass `--profile` to log the per-tick breakdown. See architecture.md
§5 for the strategy details.

**Throughput compared to flight pacing.** `--no-rerun` drops all recording and the raw loop hits
**~6.3×**, then 71% GPU, 23% PX4, and 6% actuator write: pure compute. But the *closed-loop takeoff*
doesn't complete that fast. PX4's intermittent "magnetic interference" pre-arm check needs a sustained
wall-clock window of being clear for the external Ground Control Station (GCS) to command arm. Faster
than ~5×, sim-time races past those windows so it never arms. So ~6.3× is the **throughput** ceiling
and ~4.7× is the reliable **closed-loop-flight** ceiling, gated by the GCS and PX4 arming handshake,
not compute. The example keeps recording on: it flies and produces the `.rrd`. Use `--no-rerun` only
for non-flight throughput runs.

## Run it

This is the decoupled workflow, headless and end-to-end. `docs/running.md` has the interactive
QGroundControl version:

```bash
# closed-loop flight, one na.Sim run: nexus serves the HIL :4560 in-process + records the .rrd +
# reports the with-PX4 RTF; PX4 SITL connects; the harness flies THE one PX4 flight profile
# (a script against sim.operator over MAVLink :14540: arm + AUTO.TAKEOFF + yaw sweeps). Zero-arg.
uv run -m nexus.examples px4_sitl

# or the example + the evaluation/regression gates at once
uv run --group ci python scripts/ci/evaluate_examples.py --only px4_sitl
```

Prerequisites, per `docs/running.md`: docker, a built PX4 checkout with the `none_astro_max` airframe,
set by `PX4_DIR` with default `~/code/px4`, and a CUDA host.

## Regression gate

`scripts/ci/evaluate_examples.py` runs this example with the rest of the default set on the GPU
runner, in the `gpu-examples` workflow via the shared `gpu-runner.yml`. `scripts/ci/provision_px4.sh`
provisions the PX4 checkout from the pin in `scripts/ci/px4.ref`. The script then gates the fresh run
against `scripts/ci/examples_baselines.json`. It fails if the closed-loop takeoff stops working,
checked by `takeoff_confirmed` and `climb_m`, or if PX4 warns about the sim, checked by
`px4_warnings`. It also fails if the with-PX4 sim falls behind its recorded speed, checked by `rtf`.
The same harness evaluates every example. See [Benchmarking](../reference/benchmarking.md).

CI-measured on the pinned runner:

<!-- example-stats: px4_sitl -->
