---
description: "The execution model of nexus, a ring of device and host stages that CUDA graphs replay where they can, and its determinism model: where runs are bit-reproducible, where they're only tolerance-bounded, and why."
---

# Execution & determinism

The same components and the same fixed-order [tick](architecture.md#the-simulation-loop) run on a
CPU or a CUDA device. **Nothing selects a strategy**, and there is no knob for one. The one
user-facing control is `--device {auto,cpu,cuda}`, as in
`nexus run --vehicle astro_max_base --scene empty --device cpu`. The default, `auto`, takes the GPU
when one is present. The `Orchestrator` builds the tick from the stages each component states and
captures graphs on CUDA.

## Stages and segments

A tick is an ordered ring of **stages**. Every component states its per-tick work as a list of
stages, and each stage states its kind:

| Stage kind | What | How it runs |
|---|---|---|
| **Device stage** | A static launch over persistent device buffers: physics, a command stage, a force element, a sensor's sampling kernel, the record taps, the Proportional Integral Derivative (PID) law | On CUDA, replayed as part of a CUDA graph. On a CPU device, run stage by stage, the bit-exact determinism authority |
| **Host stage** | Work that leaves the device or the process: PX4's `read` and `exchange`, a Model Predictive Control (MPC) solve, a torch policy's inference, an RTX sensor's frame exchange with the Kit peer | On the host, between graph replays |
| **Differentiable rollout** | Design optimization | A Warp tape records the whole loop, including the PID law. A loss back-propagates through it |

The loop lays the stages out in one canonical order. The sensors come first, then the guidance when
the run has one, then the controller. `clear`, the command stages, the force stages and `step` follow,
once per physics substep, and record comes last. The command stages write Newton's control inputs, the
force stages add the body forces, and `step` steps Newton's actuators and then the solver. The loop cuts
the ring at its host stages. It rotates
the ring to start after the last cut, so the ring's tail folds into the first run. **Each maximal
run of device stages becomes one CUDA graph.** The host stages run between the replays. With no
host stage the whole ring is one graph. A component that states no stages fails the build with an
error that names it, and so does a stage of a kind the loop doesn't know. Nothing falls back to a
slower path in silence. A run logs its plan once at start, for example
`stage plan: graph(clear -> rotors -> propellers -> step -> record -> imu -> mag -> baro -> gps) host(read) host(truth) host(exchange)`.

## Signals

Components pass values to each other as **signals**. A signal is a named, typed value that one
component writes and others read on the tick. Each stage declares the signals it reads and the signals
it writes, so a component's inputs and outputs are those of its stages.

A signal's type fixes where its buffer lives. A device type's buffer is a Warp array. Device stages
read and write it in their kernels, and a host stage reads it with a copy and writes it in place
between graph replays. Any other type is a host type, whose buffer holds one object. Core ships the
types of the values between roles. `Controls` holds the per-actuator commands a controller writes and
the command elements read, on the device. The setpoints are `PositionGoal`, on the device, and
`ReferenceTrajectory`, on the host: a guidance writes one, and its controller reads it. A component
can also declare its own type.

When the loop builds the ring, it wires each input to the one output of its name. It allocates one
buffer per signal and hands it to the writer and to every reader, before any stage runs, so a captured
graph replays buffers that existed at the capture. A reader's default fills the buffer until the
writer first writes, and an input that no component writes reads its default. The run stops before
any stage runs, with an error that names both ends, when:

- a reader and its writer disagree on the type
- a reader needs more of an axis than its writer's buffer holds. A reader reads the leading part of a
  wider buffer, as the rotors read the first four of the 16 controls PX4 sends
- an input has no writer, and its component gives no default
- a device stage declares a host signal
- two components write one signal that a third reads

**A guidance writes the setpoint.** Its stage is a host stage, and it holds no controller. It writes
a changed setpoint, which the controller reads on that tick. The loop also runs that stage once before
the first tick, over the settled state, so the controller holds its first setpoint before its own
first stage. A tracking guidance stays out of that pass, because its plan starts the reference's
clock. A stage that marks the tick done ends the run once the tick completes, which is how a guidance
ends its mission. A run whose guidance writes a setpoint no stage reads, such as a guidance on a PX4
run, fails.

## Capture

**Captured execution benefits online Software In The Loop (SITL) runs, not just batch.** The PX4
controller states three host stages. Its `read` stage is the sensor fan-in into the `Measurement`.
Its `truth` stage copies the base body's true state, the ground truth PX4 logs. Its `exchange` stage is
the blocking MAVLink lockstep. The graph captures the command and force stages,
physics, record and sensors *around* them. The only host↔device traffic per tick is then the small controls, measurements
and base body vectors the lockstep already moves. A **host solver** states the `read` and `exchange` stages. Examples are the
per-tick optimization of an MPC controller and a torch policy. Its solve runs between replays while
everything else stays captured. The PID law is a device stage, so there is no host stage and the
**whole** tick is one graph.

Illustrative real-time factors on a dev RTX 5080 follow. The tracked, CI-measured numbers live in
[Benchmarking](../reference/benchmarking.md). The Astro Max, on the `newton.actuators` DC-motor
rotor servos, flies its full takeoff and yaw-sweep profile at **~3.7× real-time with PX4 in the
lockstep loop**. The earlier CPU default reached ~1.0×. The PID loop, one graph, reaches **~19×**.
The host solvers ride the same graph: acados reaches ~1.4× and the trained policy ~4.8×, versus
CPU runs of 0.7× and 0.5×. Capture forces one
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
CUDA-array-interface or DLPack framework can join zero-copy. **At a host stage** a component can
be any language, process, or device, at the cost of a per-tick copy and no capture or automatic
differentiation across that stage. For the current stack there are **two** kinds: the PX4
controller's `read`, `truth` and `exchange`, every tick, and the RTX sensors' frame exchange, at their render
rate. Each talks to a peer: a process the run starts, and speaks to over a link. The Kit render peer runs in a
process of its own. At a frame's due tick the host copies the body poses and sends them, a few
hundred bytes. It takes the frame on a later tick, so the loop waits only when the peer falls a full
frame behind.

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

**The PID law is the determinism authority.** The built-in PID, flown through the
unchanged orchestrator on the CPU backend, reproduces bit-for-bit run-to-run. Real PX4 SITL is
instead *tolerance-gated*. The PX4 multi-threaded work-queue interleaves same-tick estimator and
controller threads in an OS-dependent order. So armed flight isn't bit-reproducible, even though
the sim it runs on is a pure deterministic function of the actuator stream it receives. Replaying a
recorded actuator stream open-loop into the sim reproduces a flight exactly. Bit-exact real-PX4
lockstep is a tracked, reachable goal rather than a closed door.
