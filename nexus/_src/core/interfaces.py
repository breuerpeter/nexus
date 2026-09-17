"""Narrow, typed component Protocols + capability markers, per architecture.md §2.

Side-effect-light contracts so each component is independently testable and
fault-wrappable. For the eager slice the per-body ``Wrench`` takes the form of the
shared ``state.body_f`` device buffer the Actuator writes in place, the
shared-buffer contract, so ``Actuator.forces`` and ``Physics.step`` don't
pass a Wrench value type: Physics reads ``state.body_f``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .schema import Controls, EnvSample, Measurement, Setpoint, SimTime

if TYPE_CHECKING:
    import newton


@runtime_checkable
class Clock(Protocol):
    dt: float

    def now(self) -> SimTime: ...
    def advance(self) -> SimTime: ...
    def throttle(self) -> None: ...


class Environment(Protocol):
    def sample(self, pos, t: SimTime) -> EnvSample: ...


class Physics(Protocol):
    def reset(self) -> newton.State: ...
    def clear_forces(self, state: newton.State) -> None: ...
    def step(self, state: newton.State, env: EnvSample, dt: float) -> newton.State: ...


@runtime_checkable
class Actuator(Protocol):
    """The actuator seam the Orchestrator drives once per tick: the controller's command in, forces out.

    Two class markers: ``requires_articulated`` says the actuator drives real joints, so the pairing
    guard refuses a model with nothing on ``model.actuators``; ``capturable`` says its device region
    joins a CUDA graph, with no per-tick host op.
    """

    requires_articulated: bool
    capturable: bool

    def forces(self, controls: Controls, state: newton.State, env: EnvSample) -> None:
        """Write the per-body Wrench into the shared ``state.body_f`` buffer."""


class Sensor(Protocol):
    # Optional class markers:
    #   capturable = True:  sample_wp/read split exists; the sensor joins the captured CUDA graph.
    #   host_rate = True:   a Kit/RTX renderable, camera or lidar: host-bound + low-rate; sampled at the
    #                         host seam of every loop, self-decimating to its own rate. Never vetoes the
    #                         captured strategy; excluded from the capture gate + the graph.
    def sample(self, state: newton.State, env: EnvSample, t: SimTime, out: Measurement) -> None:
        """Fill this sensor's fields into the shared per-tick Measurement (FRD).

        Reads body state from the live ``newton.State``, the shared contract, so the
        component is runtime-agnostic, the same in ``runtime-standalone`` and
        ``runtime-isaacsim``. The orchestrator fans the live state to every sensor per tick.
        """


class Controller(Protocol):
    # True for a host-boundary controller whose ``exchange`` is a blocking off-device round-trip,
    # for example PX4 MAVLink lockstep. Uncapturable, so the captured strategy records only the device
    # region and runs the controller seam between replays. In-process controllers leave this False,
    # the default via getattr, and are themselves ``capturable``, since their exchange runs on-device.
    host_boundary: bool

    def connect(self) -> None: ...
    def exchange(self, meas: Measurement, t: SimTime, timeout: float | None) -> Controls | None:
        """Serialize meas -> HIL_*, send, block for actuators. None on timeout."""

    def close(self) -> None: ...

    def accept_setpoint(self, sp: Setpoint) -> None:
        """Write the controller's own setpoint buffer **in place** from a ``Setpoint``.

        The thin control surface: the operator commands an *in-process*
        autopilot by flipping its persistent setpoint buffer, a §6 value-mutation with a static address
        and zero re-capture on the next replay, then the in-loop ``exchange`` reads it. Each controller
        narrows the ``Setpoint`` union to the variant it supports, PositionGoal for policy/pid,
        Waypoints for sampling Model Predictive Control (MPC), ReferenceTrajectory for acados, and raises
        on the rest.

        **PX4 has no in-process setpoint**, since its mission lives in the external PX4 process, so the
        ``Px4MavlinkController`` does *not* offer this, and its control surface is ``None``; instead,
        ``sim.operator``, a ``Px4Offboard``, commands PX4 over MAVLink. Optional on the
        protocol for exactly that reason.
        """


class Renderer(Protocol):
    def render(self, state: newton.State, env: EnvSample, t: SimTime) -> None: ...


class Recorder(Protocol):
    def log(self, t: SimTime, state: newton.State) -> None:
        """Hand one tick's state to the logger. Output-only, architecture.md §10.

        The minimal per-tick seam: the core loop calls this so something can log the
        evolving state, for example newton-logging drives NVIDIA Newton's ``ViewerRerun.log_state``,
        without core depending on Rerun. Components log their own *events* through the
        ``newton`` logger directly.
        """

    def close(self) -> None: ...


@runtime_checkable
class Capturable(Protocol):
    """Marker: a static Warp launch over persistent in-place buffers that can join
    a CUDA graph. The orchestrator forms the maximal contiguous captured region
    from adjacent Capturable components, actuator -> physics -> sensors; the
    controller is the host boundary. The eager slice defers capture, but
    the marker + persistent/static buffers exist now so it's not a retrofit.
    """

    capturable: bool
