---
description: "nexus's execution strategies, eager, CUDA-graph-captured, and differentiable rollout, plus its determinism model: where runs are bit-reproducible, where they're only tolerance-bounded, and why."
---

# Execution & determinism

The same components and the same fixed-order [tick](architecture.md#the-simulation-loop) run under
different execution strategies. **The strategy is automatic**, with no separate knob. The one
user-facing control is `nexus run --device {auto,cpu,cuda}`. The default, `auto`, captures on
the GPU when one is present and runs CPU-eager otherwise. The `Orchestrator` selects the strategy
from the active device and the components' capability markers.

## Execution strategies

| Strategy | When | How it runs |
|---|---|---|
| **Eager** | CPU device, a device-region component that doesn't support capture, or the debug or parity path | Python steps the components one by one: the standard eager execution and the bit-exact determinism authority |
| **Captured** | CUDA device whose device region, physics plus actuator plus in-graph sensors, supports capture | The orchestrator records the contiguous device region once into a CUDA graph and replays it. The controller joins the graph when it supports capture. Otherwise it exchanges at the host seam between replays |
| **Differentiable rollout** | Design optimization | A Warp tape records the whole loop, including the in-process Proportional Integral Derivative (PID) controller. A loss back-propagates through it |

The rule is single: **capture everything that supports capture, and host-exchange only the
components that need it.** A *controller* that doesn't support capture never forces the rest of the
tick eager. Only a device-region component that doesn't support capture, whether physics, actuator,
or an in-graph sensor, does.

**Captured execution benefits online Software In The Loop (SITL) runs, not just batch.** The
captured region is *the whole device-side step minus the controller*. An external PX4 controller is
a host boundary. There the orchestrator captures actuator → physics → sensors *around* the blocking
MAVLink exchange. The only host↔device traffic per tick is then the small controls and measurements
vectors the lockstep already moves. An in-process **host solver** rides the same seam. Examples are
the per-tick optimization of a Model Predictive Control (MPC) controller and a torch policy. Its
solve runs between replays while everything else stays captured. With the in-process PID or policy
mixer, a device-native `exchange`, there is no host hop and the **whole** tick captures.

Illustrative real-time factors on a dev RTX 5080 follow. The tracked, CI-measured numbers live in
[Benchmarking](../reference/benchmarking.md). The Astro Max, on the `newton.actuators` DC-motor
rotor servos, flies its full takeoff and yaw-sweep profile at **~3.7× real-time with PX4 in the
lockstep loop**. The earlier CPU-eager default reached ~1.0×. The fully in-process PID loop
reaches **~19×**. The in-process host solvers ride the same captured region: acados reaches ~1.4×
and the trained policy ~4.8×, versus full-eager fallbacks of 0.7× and 0.5×. Capture forces one
subtlety: sensor noise must **dither per replay**, through a device step counter incremented inside
the graph. Otherwise a frozen captured sensor stream reads to PX4's estimator as a stuck sensor and
position fusion never starts.

### Capture and automatic differentiation share one prerequisite

The *same* thing gates CUDA-graph capture and reverse-mode automatic differentiation: neither can
cross a host, marshalling, or process boundary. The work that keeps the per-tick loop device-native
unlocks both. The eager and differentiable paths share the per-tick control sequence of observe →
control → actuate. They don't share the *outer* loop. They differ in their memory model, with
persistent double-buffers for capture versus a per-step history for backprop-through-time, in host
scaffolding, and in the physics assembly. So components take their buffers as arguments, and one
set of Warp kernels serves both callers.

## Representation & language boundaries

Interfaces are the contract. Representation isn't mandated. **Inside a captured or differentiable
region** all components share one device representation, Warp by default, and any
CUDA-array-interface or DLPack framework can join zero-copy. **Across a host boundary** a component
can be any language, process, or device, at the cost of a per-tick copy and no capture or automatic
differentiation across that seam. For the current stack there is exactly **one** such boundary: the
PX4 controller. In the Isaac Sim runtime, RTX rendering shares the same device stage in-process, so
it adds no marshalling boundary.

## Determinism

Determinism is a foundational engine property, separate from the preceding strategy. Given a
scenario, a seed, and pinned code, PX4, and parameter versions, a run reproduces bit-for-bit. This
requires one seeded Random Number Generator (RNG) tree with named sub-streams, fixed-step lockstep,
deterministic PX4 initialization, and no wall-clock or global RNG in the loop.

**The CPU backend is the bit-exact authority. The GPU is tolerance-gated.** Newton's production
contact and constraint solve, `mujoco-warp`, accumulates forces with floating-point atomic adds
whose ordering is race-dependent. So a contact-rich GPU run diverges run-to-run: a perturbation at
the scale of one Unit In The Last Place (ULP) amplifies chaotically once contacts engage. The Warp
**CPU backend is single-threaded and bit-exact**, so the CI determinism gate runs there. The gate
runs a scenario twice and asserts a bit-exact match. GPU runs assert only tolerance-bounded
agreement.

**Deterministic initialization.** Before the steady loop, initialization settles the vehicle at the
North East Down (NED) origin and samples the sensors once to seed the finite-difference Inertial
Measurement Unit (IMU). It also pins the Global Positioning System (GPS) sub-rate phase to the loop
origin and computes the magnetic field from the scenario's GPS origin. An inconsistent field gives
PX4 an invalid heading estimate, and it can't arm.

**The in-process controller is the determinism authority.** The built-in PID, flown through the
unchanged orchestrator on the CPU backend, reproduces bit-for-bit run-to-run. Real PX4 SITL is
instead *tolerance-gated*. The PX4 multi-threaded work-queue interleaves same-tick estimator and
controller threads in an OS-dependent order. So armed flight isn't bit-reproducible, even though
the sim it runs on is a pure deterministic function of the actuator stream it receives. Replaying a
recorded actuator stream open-loop into the sim reproduces a flight exactly. Bit-exact real-PX4
lockstep is a tracked, reachable goal rather than a closed door.
