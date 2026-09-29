"""Narrow, typed component Protocols and the stage contract, per architecture.md §2.

Side-effect-light contracts so each component is independently testable and
fault-wrappable. Every loop component states its per-tick work as a list of
:class:`Stage`, and the loop runs each stage over the shared :class:`Tick`. The per-body
``Wrench`` takes the form of the shared ``state.body_f`` device buffer the Actuator writes in
place, the shared-buffer contract, so the actuator's stage and ``Physics.step`` don't pass a
Wrench value type: Physics reads ``state.body_f``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .schema import Measurement, Setpoint, SimTime

if TYPE_CHECKING:
    import newton


@dataclass(slots=True)
class Tick:
    """The context one control tick hands every stage: the shared state and buffers a stage reads and
    writes in place. ``controls`` is the controller's persistent ``(1, n)`` device buffer of normalized
    per-actuator commands, which its stage sets and the actuator's device stage reads, so a captured
    graph reads the same buffer every replay. ``dt`` is one physics step, the control timestep over
    ``physics_substeps``. ``sensors`` are the sensors with device stages, whose ``read`` fills ``meas``
    at a controller's ``read`` host stage. ``timeout`` bounds a host stage's wait on its peer.
    """

    state: Any
    t: SimTime
    dt: float
    meas: Measurement
    controls: Any = None
    sensors: list = field(default_factory=list)
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class Stage:
    """One unit of a component's per-tick work, the shape every loop component states in its
    ``stages()``. A ``device`` stage joins the CUDA graph; a ``host`` stage runs between graph replays.
    ``run(tick)`` does the work; a host stage returns ``False`` when it produced nothing because its
    peer didn't answer, which the preroll retries and the steady loop ends the run on. A ``warm``
    sensor or controller stage runs in the seed pass over the settled state, before any capture, so
    every device buffer it allocates exists first.
    """

    name: str
    kind: Literal["device", "host"]
    run: Callable[[Tick], Any]
    warm: bool = True


@runtime_checkable
class Clock(Protocol):
    dt: float

    def now(self) -> SimTime: ...
    def advance(self) -> SimTime: ...
    def throttle(self) -> None: ...


class Physics(Protocol):
    def reset(self) -> newton.State: ...
    def clear_forces(self, state: newton.State) -> None: ...
    def step(self, state: newton.State, dt: float) -> newton.State: ...
    def stages(self) -> list[Stage]:
        """The ``clear`` and ``step`` stages; the loop runs ``clear``, the actuator, ``step`` once per
        physics substep.
        """


@runtime_checkable
class Actuator(Protocol):
    """The actuator seam the Orchestrator drives once per physics substep: the controller's command
    buffer in, forces out. The class marker ``requires_articulated`` says the actuator drives real
    joints, so the pairing guard refuses a model with nothing on ``model.actuators``.
    """

    requires_articulated: bool

    def forces_wp(self, cmd: Any, state: newton.State) -> None:
        """Write the per-body Wrench into the shared ``state.body_f`` buffer from ``cmd``, the
        controller's ``(1, n)`` device buffer of normalized commands.
        """

    def stages(self) -> list[Stage]:
        """One device stage over ``forces_wp``."""


class Sensor(Protocol):
    def stages(self) -> list[Stage]:
        """The sensor's per-tick work: a device stage that samples the live ``newton.State`` into the
        sensor's device buffer, or a host stage for a sensor whose work leaves the process, an RTX
        camera's frame exchange with the Kit peer.
        """

    def read(self, out: Measurement) -> None:
        """One D2H of the device buffer into the shared host ``Measurement``, Forward Right Down (FRD), at a controller's
        ``read`` host stage. A sensor with a host stage never gets read.
        """


class Controller(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...

    def stages(self) -> list[Stage]:
        """The controller's per-tick work. A controller that blocks on a peer, PX4, or solves on the
        host, a Model Predictive Control (MPC) solver, states a ``read`` host stage for the sensor fan-in and an ``exchange`` host
        stage, :func:`nexus._src.core.stages.peer_stages`; a device-native law, the Proportional Integral Derivative (PID) example, states device
        stages. Its stage sets ``Tick.controls`` to the controller's persistent command buffer.

        A controller with a peer, one that exposes ``attached``, connects after the loop's warm pass
        and graph capture, so a peer that dials in early waits on no kernel load. Its device stages
        run and capture before ``connect()``, over buffers that exist from construction.
        """

    def accept_setpoint(self, sp: Setpoint) -> None:
        """Write the controller's own setpoint buffer **in place** from a ``Setpoint``.

        The thin control surface: the operator commands an
        autopilot by flipping its persistent setpoint buffer, a §6 value-mutation with a static address
        and zero re-capture on the next replay, then the in-loop ``exchange`` reads it. Each controller
        narrows the ``Setpoint`` union to the variant it supports, PositionGoal for policy/pid,
        Waypoints for sampling Model Predictive Control (MPC), ReferenceTrajectory for acados, and raises
        on the rest.

        **PX4 has no setpoint surface**, since its mission lives in its peer, so the
        ``Px4MavlinkController`` does *not* offer this, and its control surface is ``None``; instead,
        ``sim.operator``, a ``Px4Offboard``, commands PX4 over MAVLink. Optional on the
        protocol for exactly that reason.
        """


class Renderer(Protocol):
    def render(self, state: newton.State, t: SimTime) -> None: ...


class Recorder(Protocol):
    def log(self, t: SimTime, state: newton.State) -> None:
        """Hand one tick's state to the logger. Output-only, architecture.md §10.

        The minimal per-tick seam: the core loop calls this so something can log the
        evolving state, for example newton-logging drives NVIDIA Newton's ``ViewerRerun.log_state``,
        without core depending on Rerun. Components log their own *events* through the
        ``newton`` logger directly.
        """

    def close(self) -> None: ...
