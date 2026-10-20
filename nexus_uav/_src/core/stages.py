"""The ring of stages one control tick runs, and its partition into captured segments.

A tick is an ordered ring of stages, per docs/design/execution.md. :func:`build_ring` lays every
component's stages out in the canonical order: the clock, ``clear`` -> the command elements -> the force
elements -> ``step`` per physics substep, ``state``, then the sensors, the estimator, the guidance and the
controller over the state the step left, then the loop end of each declared peer's inputs, then record. So the
physics step opens a tick, as a fixed-step
simulator's does, and every row the tick records describes the state at the tick's time. :func:`partition`
cuts the ring at its host stages, in ring order, so each maximal run of device stages becomes one CUDA graph.
A controller that blocks on a peer or solves on the host states its own host stage, which writes its controls
signal, of the width its actuators take.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .interfaces import Stage

KINDS = ("device", "host")


@dataclass(frozen=True, slots=True)
class Bound:
    """A stage bound to the component that states it and the role it fills in the ring."""

    stage: Stage
    component: object
    role: str  # clock, sensor, estimator, guidance, controller, physics, command, force, actuator or record


@dataclass(frozen=True, slots=True)
class Segment:
    """A maximal run of device stages, one captured graph, or one host stage."""

    kind: str
    stages: tuple[Stage, ...]


def stages_of(component, role: str) -> list[Bound]:
    """The stages a component states, each checked for a kind the loop knows.

    Raises:
        ValueError: The component states no stages, an empty list included, or a stage of an unknown
            kind.
    """
    name = type(component).__name__
    stages = component.stages()
    if not stages:
        raise ValueError(f"{name} states no stages: every {role} lists its per-tick work as Stage objects")
    bound = []
    for st in stages:
        if st.kind not in KINDS:
            raise ValueError(f"{name} stage {st.name!r} has kind {st.kind!r}, not one of {KINDS}")
        bound.append(Bound(st, component, role))
    return bound


# The roles whose stages run once per physics substep, between ``clear`` and ``step``, in this order: the old
# actuator stage, which no shipped component fills, runs with the force stages, since it writes the body
# forces too.
SUBSTEP_ROLES = ("command", "force", "actuator")
# The roles whose stages run once per tick over the state the step left, in this order. The link role is last:
# the loop end of each declared peer's inputs, which reads what every writer of the tick wrote.
TICK_ROLES = ("sensor", "estimator", "guidance", "controller", "link")


def build_ring(*, clock, physics, roles: Mapping[str, Sequence], record: Stage, substeps: int) -> list[Bound]:
    """Every component's stages in the canonical order the domain fixes: the clock, then ``clear`` -> each
    substep role's components -> ``step`` unrolled ``substeps`` times, then ``state``, which writes the plant's
    outputs from the stepped state, then each tick role's components, then ``record``. ``roles`` maps a role to
    its components, in the order they run; a role it leaves out has none.

    The step applies the controls of the tick before, and every later stage reads the state it left. The
    estimator sits after the sensors, so it reads their values of that tick, and before the guidance and the
    controller, so they read its estimate of that tick. The guidance sits before the controller, so the
    setpoint it writes is the one the controller reads on that tick.

    Raises:
        ValueError: A component states no stages, a stage of an unknown kind, or physics states no
            ``clear``, ``step`` and ``state``.
    """
    ring = stages_of(clock, "clock")
    phys = {b.stage.name: b for b in stages_of(physics, "physics")}
    if any(name not in phys for name in ("clear", "step", "state")):
        raise ValueError(f"{type(physics).__name__} states {sorted(phys)}, not the clear, step and state stages")
    inner = [b for role in SUBSTEP_ROLES for c in roles.get(role, ()) for b in stages_of(c, role)]
    for _ in range(substeps):
        ring += [phys["clear"], *inner, phys["step"]]
    ring.append(phys["state"])
    ring += [b for role in TICK_ROLES for c in roles.get(role, ()) for b in stages_of(c, role)]
    ring.append(Bound(record, None, "record"))
    return ring


def partition(ring: list[Bound]) -> list[Segment]:
    """Cut the ring at its host stages, in ring order: each maximal run of device stages is one segment."""
    segments: list[Segment] = []
    run: list[Stage] = []
    for b in ring:
        if b.stage.kind == "host":
            if run:
                segments.append(Segment("device", tuple(run)))
                run = []
            segments.append(Segment("host", (b.stage,)))
        else:
            run.append(b.stage)
    if run:
        segments.append(Segment("device", tuple(run)))
    return segments


def start_stages(ring: list[Bound], waiting=()) -> list[Stage]:
    """The stages of the start, which runs once over the settled state at sim time zero, before any capture:
    the ring without the clock's stage and the physics substeps. So every device buffer exists and every
    kernel has loaded before the capture, and the first tick's step applies the controls the start computed.
    A sensor's host stage stays out, so a camera's frame exchange starts with the first tick, and so does
    each host stage of a component in ``waiting``, whose peer hasn't dialed in yet.
    """
    out = []
    for b in ring:
        if b.role in ("clock", "command", "force", "actuator") or (b.role == "physics" and b.stage.name != "state"):
            continue
        if b.stage.kind == "host" and (b.role == "sensor" or any(b.component is c for c in waiting)):
            continue
        out.append(b.stage)
    return out


def plan_line(segments: list[Segment], captured: bool) -> str:
    """The one line a run logs at start: each segment in run order, a device segment as
    ``graph(...)`` when it captures and ``eager(...)`` when it runs stage by stage.
    """
    parts = []
    for seg in segments:
        names = " -> ".join(st.name for st in seg.stages)
        label = ("graph" if captured else "eager") if seg.kind == "device" else "host"
        parts.append(f"{label}({names})")
    return "stage plan: " + " ".join(parts)


__all__ = [
    "SUBSTEP_ROLES",
    "TICK_ROLES",
    "Bound",
    "Segment",
    "build_ring",
    "partition",
    "plan_line",
    "start_stages",
]
