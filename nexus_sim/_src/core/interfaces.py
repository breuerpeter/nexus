"""The component contract, the stage contract, and the contracts of the fixed parts.

One contract, :class:`Component`, serves every role: a sensor, an estimator, a guidance, a controller, a
command element or a force element. A component states its per-tick work as a list of :class:`Stage`, each
declaring the signals it reads and writes, and the loop runs each stage over the shared :class:`Tick`. The
per-body wrench takes the form of the shared ``state.body_f`` device buffer the force elements add to in
place, so a force element's stage and ``Physics.step`` pass no wrench value: physics reads ``state.body_f``.
The fixed parts, the clock and the physics, keep contracts of their own, which no project implements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .schema import SimTime

if TYPE_CHECKING:
    import newton

    from .signals import Signal


@dataclass(slots=True)
class Tick:
    """The context one control tick hands every stage: the loop's own state, which a stage reads and
    writes in place. A value between two components is a signal the stages declare, not a field here, a
    sensor's sample among them. ``dt`` is one physics step, the control timestep over
    ``physics_substeps``. ``timeout`` bounds a host stage's wait on its peer. ``base`` is the index of the
    vehicle's base body, its airframe, in ``state``: a stage that reads the vehicle's true state reads that
    row.

    A stage sets ``done`` to end the run, a guidance's when its mission is over: the loop completes the
    tick and takes no further one.
    """

    state: Any
    t: SimTime
    dt: float
    timeout: float | None = None
    base: int = 0
    done: bool = False


@dataclass(frozen=True, slots=True)
class Stage:
    """One unit of a component's per-tick work, the shape every loop component states in its
    ``stages()``. A ``device`` stage joins the CUDA graph; a ``host`` stage runs between graph replays.
    ``run(tick)`` does the work; a host stage returns ``False`` when it produced nothing because its
    peer didn't answer, which the preroll retries and the steady loop ends the run on. A ``warm``
    sensor, estimator or controller stage runs in the seed pass over the settled state, before any
    capture, so every device buffer it allocates exists first, and the guidance's first stage reads an
    estimate. A ``warm`` guidance stage runs in that pass too,
    so the controller holds the guidance's first setpoint before its own first stage. ``reads`` and
    ``writes`` are the signals the stage reads and writes, which the loop wires before any stage runs.
    """

    name: str
    kind: Literal["device", "host"]
    run: Callable[[Tick], Any]
    warm: bool = True
    reads: tuple[Signal, ...] = ()
    writes: tuple[Signal, ...] = ()


class Component:
    """The contract every component fills, whatever its role.

    A component states its per-tick work in :meth:`stages`, each stage declaring the signals it reads and
    writes, which the loop wires before any stage runs. The members below are the whole of what the loop,
    the builder and ``Sim`` read from a component, so they find none by name: a class overrides the ones it
    has and inherits the rest. The role a component fills, which the schema that declares it states, fixes
    where its stages run in the tick and who declares it; the contract is the same for every role.

    Attributes:
        name: The instance name, which ends the component's path, ``vehicle/sensors/<name>``, and keys its
            recorded histories. The builder sets it for a declared component from its prim; ``None`` reads
            as the class's name.
        requires: The names of the peers the class needs, such as ``("kit",)`` for an RTX sensor: the builder
            starts each once, whatever the number of components that require it, and the loop hands the
            component each one's link at :meth:`connect`.
        attached: Whether the component's peer has dialed in. ``True`` for a component with no peer, the
            default. A component that reads ``False`` before it connects waits on a peer: the loop connects
            it after the graph capture, so a peer that dials in early waits on no kernel load, and holds the
            sim clock until it reads ``True``.
    """

    name: str | None = None
    requires: tuple[str, ...] = ()
    attached: bool = True

    def stages(self) -> list[Stage]:
        """The component's per-tick work, in ring order: one or more stages, each declaring what it reads and
        writes. A device stage joins the captured graph; a host stage runs between replays. None by default,
        which fails the build: every component states at least one.
        """
        return []

    def connect(self, links: Mapping[str, Any]) -> None:
        """Open what the component needs for the run: bind a port, load a model. Nothing by default. The loop
        connects a component before the warm pass, or after the graph capture when it waits on a peer.

        Args:
            links: The link of each peer the class requires, by the peer's name, as ``requires`` lists them:
                the run started the peer and holds its link. Empty for a class that requires none.
        """

    def close(self) -> None:
        """Release what :meth:`connect` opened, at the run's end. Nothing by default. Idempotent."""

    def set_logger(self, logger) -> None:
        """Take the Logger scoped to the component's path, or ``None`` when the run records nothing. A component
        that logs a value with no fixed width live, a camera's frame or a guidance's markers, keeps it and logs
        only its own row's name; every fixed-width signal the Recorder keeps itself. Nothing by default.
        """

    def artifacts(self) -> dict:
        """The files the component leaves for a reader after the run, by name, which ``Sim.artifacts`` collects
        beside the recording: PX4's controller names its ULog. Empty by default.
        """
        return {}


@runtime_checkable
class Clock(Protocol):
    dt: float

    def now(self) -> SimTime: ...
    def advance(self) -> SimTime: ...
    def throttle(self) -> None: ...


class Physics(Protocol):
    base_body: str
    """The base body's label, the leaf of its model label when unique: the body a reader of the vehicle's pose
    takes with no connection, and the key of its history under ``vehicle/body/``."""

    def reset(self) -> newton.State: ...
    def clear_forces(self, state: newton.State) -> None: ...
    def step(self, state: newton.State, dt: float) -> newton.State: ...
    def stages(self) -> list[Stage]:
        """The ``clear``, ``step`` and ``state`` stages; the loop runs ``clear``, the command elements, the force
        elements and ``step`` once per physics substep, ``step`` steps Newton's actuators before the solver, and
        ``state`` runs once per tick after the last substep and writes the plant's outputs, one signal per body,
        ``body/<label>``, and per joint, ``joints/<label>``, from the stepped state; the loop runs it once more
        over the settled state before the warm pass.
        """
