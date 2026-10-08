---
description: "The execution model of nexus, a ring of device and host stages that CUDA graphs replay where they can, and its determinism model: where runs are bit-reproducible, where they're only tolerance-bounded, and why."
---

# Execution & determinism

The same [components](concepts.md#component) and the same fixed-order [tick](concepts.md#tick) run on a
CPU or a CUDA device. **Nothing selects a strategy**, and there is no knob for one. The one
user-facing control is `--device {auto,cpu,cuda}`, as in
`nexus run --vehicle astro_max_base --scene empty --device cpu`. The default, `auto`, takes the GPU
when one is present. The [loop](concepts.md#loop) builds the tick from the stages each component states
and captures graphs on CUDA.

## Stages and segments

A [tick](concepts.md#tick) runs the [ring](concepts.md#ring) of [stages](concepts.md#stage) the
components state, in the loop's fixed order. Device stages include the [physics](concepts.md#physics), the command and
force elements, a sensor's sampling kernel, the record taps, and the Proportional Integral
Derivative (PID) law. On CUDA a device stage replays as part of a CUDA graph. On a CPU device it
runs stage by stage, the bit-exact determinism authority. Host stages include PX4's `truth` and
`exchange`, a Model Predictive Control (MPC) solve, a torch policy's inference, and an RTX
sensor's frame exchange with the Kit peer. A host stage runs on the host between graph replays. A
differentiable rollout, for design optimization, records the whole loop on a Warp tape, the PID
law included, and a loss back-propagates through it.

The loop cuts the ring at its host stages. It rotates the ring to start after the last cut, so the
ring's tail folds into the first run.
**Each [segment](concepts.md#segment) becomes one CUDA graph.** The host stages run between the replays. With no
host stage the whole ring is one graph. A component that states no stages fails the build with an
error that names it, and so does a stage of a kind the loop doesn't know. Nothing falls back to a
slower path in silence. A [run](concepts.md#run) logs its plan once at start, for example
`stage plan: graph(clock -> clear -> rotors -> propellers -> step -> record -> imu -> mag -> baro -> gps) host(truth) host(exchange)`.

## Signals

Components pass values to each other as [signals](concepts.md#signal). Each stage declares the
signals it reads and the signals it writes, so a component's inputs and outputs are those of its
stages.

A signal's type fixes where its buffer lives. A signal whose type is a Warp struct or a Warp value type,
such as `wp.float32`, lives on the device: its buffer is a Warp array. Device stages read and write
it in their kernels, and a host stage reads it with a copy and writes it in place between graph replays.
Any other type is a host type, whose buffer holds one object. Core ships the types of the values between
[roles](concepts.md#role):

- The controls, the per-actuator commands a controller writes and the command elements read, are a row
  of `wp.float32` on the device.
- A setpoint is a position goal as one `wp.vec3`, on the device, or a `ReferenceTrajectory`, on the
  host: a guidance writes one, and its controller reads it.
- `PoseTwist` is the estimate, a struct on the device: the estimator writes the base body's pose and
  twist, and the guidance and the controller read it in place of the physics state.
- Each sensor writes its sample to a signal of its own type. `ImuSample`, `MagSample`, `BaroSample` and
  `GpsSample` are structs on the device, the signals `imu`, `mag`, `baro` and `gps`. The first field of
  each is the sim time of the tick that took the sample. A camera's frame is an `Image` and a lidar's
  scan a `PointCloud`, on the host, each with the sim time it shows: the signals `camera`,
  `thermal_camera` and `lidar`. An estimator reads them, and so does PX4's controller, which hands them to
  PX4's estimator.
- The tick's sim time is the signal `time`, a `wp.float64` on the device. When a stage reads it, the
  loop's clock stage opens each tick and adds one control tick, in step with the host clock. So a sensor
  stamps its sample with the tick's time inside a captured graph too.

A component can also declare its own type, a Warp struct of its own.

When the loop builds the ring, it allocates one buffer per output and hands it to its writer. It does so
before any stage runs, so a captured graph replays buffers that existed at the capture. An input takes
the buffer of the one output of its name. Where two components write one signal, as two IMUs do, a
reader takes the writer that a connection on its prim names, or every writer as a list. A connection is
a relationship `nexus:inputs:<signal>` that the reader's schema declares, whose target is the writer's
prim:

```usda
def Scope "Estimator" (
    prepend apiSchemas = ["MyEstimatorAPI"]
)
{
    rel nexus:inputs:imu = </vehicle/body/ImuQuiet>
}
```

A reader that takes every writer declares the signal's type as a list, such as `list[ImuSample]`, and
reads one sample per writer, in the order the vehicle declares them. A reader's default fills the buffer
it takes until its writer first writes, and an input that no component writes reads its default. The run
stops before any stage runs, with an error that names both ends, when:

- a reader and its writer disagree on the type
- a reader needs more of an axis than its writer's buffer holds. A reader reads the leading part of a
  wider buffer, as the rotors read the first four of the 16 controls PX4 sends
- an input has no writer, and its component gives no default
- a device stage declares a host signal
- two components write one signal that a third reads through neither a connection nor a list
- a connection names a prim whose component writes no such signal
- a guidance or a controller reads the estimate, and the run has no estimator to write it

**An estimator writes the estimate, and a guidance writes the setpoint.** The estimator's stage runs
after the sensors', so it reads their values of that tick. It runs before the guidance's and the
controller's, so they read its estimate of that tick. The guidance's stage is a host stage, and it
holds no controller. It reads the vehicle's position from the estimate and writes a changed setpoint,
which the controller reads on that tick. The loop also runs both stages once before the first tick,
over the settled state, the estimator's first. So the guidance's first stage reads an estimate, and
the controller holds its first setpoint before its own first stage. A tracking guidance stays out of that pass, because its plan starts the reference's
clock. A stage that marks the tick done ends the run once the tick completes, which is how a guidance
ends its mission. A run whose guidance writes a setpoint no stage reads, such as a guidance on a PX4
run, fails.

## Capture

**Captured execution benefits online Software In The Loop (SITL) runs, not just batch.** The PX4
controller states two host stages. Its `truth` stage copies the base body's true state, the ground truth
PX4 logs. Its `exchange` stage reads each sensor's sample and runs the blocking MAVLink lockstep. The
graph captures the clock, the command and force elements, physics, record and sensors *around* them.
The only host↔device traffic per tick is then the small controls, sensor samples, and base body vectors
the lockstep already moves. A **host solver** states the `exchange` stage. Examples are the per-tick
optimization of an MPC controller and a torch policy. Its solve runs between replays while
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
persistent double-buffers for capture versus one state per step for backprop-through-time, in host
scaffolding, and in the physics assembly. So components take their buffers as arguments, and one
set of Warp kernels serves both callers.

## Representation & language boundaries

Interfaces are the contract. Representation isn't mandated. **Inside a captured or differentiable
region** all components share one device representation, Warp by default, and any
CUDA-array-interface or DLPack framework can join zero-copy. **At a host stage** a component can
be any language, process, or device, at the cost of a per-tick copy and no capture or automatic
differentiation across that stage. For the current stack there are **two** kinds: the PX4
controller's `truth` and `exchange`, every tick, and the RTX sensors' frame exchange, at their render
rate. Each talks to a [peer](concepts.md#peer) over a [link](concepts.md#link). The Kit render peer runs in a
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
