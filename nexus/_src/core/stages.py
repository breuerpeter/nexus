"""The ring of stages one control tick runs, and its partition into captured segments.

A tick is an ordered ring of stages, per docs/design/execution.md. :func:`build_ring` lays every
component's stages out in the canonical order, sensors, guidance, controller, ``clear`` -> the command
stages -> the force stages -> ``step`` per physics substep, record. :func:`partition` cuts the ring at
its host stages and rotates it to
start after the last cut, so the ring's tail folds into the first run and each maximal run of
device stages becomes one CUDA graph; with no host stage the whole ring is one segment in canonical
order. :func:`peer_stages` is the one shape for a controller that blocks on a peer or solves on the
host: a ``read`` host stage for the sensor fan-in and an ``exchange`` host stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .interfaces import Stage

if TYPE_CHECKING:
    from .interfaces import Tick

KINDS = ("device", "host")
CHANNELS = 16  # MAVLink's HIL_ACTUATOR_CONTROLS carries 16 channels, the widest command any controller sends


@dataclass(frozen=True, slots=True)
class Bound:
    """A stage bound to the component that states it and the role it fills in the ring."""

    stage: Stage
    component: object
    role: str  # sensor, guidance, controller, physics, command, force, actuator or record


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
    stages = component.stages() if hasattr(component, "stages") else []
    if not stages:
        raise ValueError(f"{name} states no stages: every {role} lists its per-tick work as Stage objects")
    bound = []
    for st in stages:
        if st.kind not in KINDS:
            raise ValueError(f"{name} stage {st.name!r} has kind {st.kind!r}, not one of {KINDS}")
        bound.append(Bound(st, component, role))
    return bound


def device_sensors(ring: list[Bound]) -> list:
    """The sensors with a device stage, in ring order, each once: the ones a ``read`` host stage reads."""
    out = []
    for b in ring:
        if b.role == "sensor" and b.stage.kind == "device" and b.component not in out:
            out.append(b.component)
    return out


def build_ring(
    *, sensors, controller, physics, commands=(), forces=(), actuator=None, record: Stage, substeps: int, guidance=None
) -> list[Bound]:
    """Every component's stages in the canonical order the domain fixes: sensors, the guidance when the
    run has one, controller, then ``clear`` -> the command stages -> the force stages -> ``step`` unrolled
    ``substeps`` times, then ``record``. The guidance sits before the controller, so the setpoint it writes
    on a tick is the one the controller reads on that tick. The old actuator seam's one stage, the
    examples' single-body ``Rotors``, runs with the force stages, since it writes the body forces too.

    Raises:
        ValueError: A component states no stages, a stage of an unknown kind, or physics states no
            ``clear`` and ``step``.
    """
    ring = []
    for s in sensors:
        ring += stages_of(s, "sensor")
    if guidance is not None:
        ring += stages_of(guidance, "guidance")
    ring += stages_of(controller, "controller")
    phys = {b.stage.name: b for b in stages_of(physics, "physics")}
    if "clear" not in phys or "step" not in phys:
        raise ValueError(f"{type(physics).__name__} states {sorted(phys)}, not the clear and step stages")
    inner = []
    for c in commands:
        inner += stages_of(c, "command")
    for f in forces:
        inner += stages_of(f, "force")
    if actuator is not None:
        inner += stages_of(actuator, "actuator")
    for _ in range(substeps):
        ring += [phys["clear"], *inner, phys["step"]]
    ring.append(Bound(record, None, "record"))
    return ring


def partition(ring: list[Bound]) -> list[Segment]:
    """Cut the ring at its host stages and rotate it to start after the last cut."""
    hosts = [i for i, b in enumerate(ring) if b.stage.kind == "host"]
    if not hosts:
        return [Segment("device", tuple(b.stage for b in ring))]
    start = hosts[-1] + 1
    rotated = ring[start:] + ring[:start]
    segments: list[Segment] = []
    run: list[Stage] = []
    for b in rotated:
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


def seed_stages(ring: list[Bound]) -> list[Stage]:
    """The stages of one pass over the settled state: the sensors' and the controller's warm device
    stages, and the controller's host stages, in ring order. Physics, the command stages, the force
    stages and the record stage never run here, so the settled state is the state the first tick starts from, and a
    sensor's host stage never does, so a camera's frame exchange starts with the first tick.
    """
    out = []
    for b in ring:
        if b.role not in ("sensor", "controller"):
            continue
        if b.stage.kind == "host":
            if b.role == "controller":
                out.append(b.stage)
        elif b.stage.warm:
            out.append(b.stage)
    return out


def warm_stages(ring: list[Bound]) -> list[Stage]:
    """The device stages of the seed pass: the warm pass that runs once before any capture, so every
    device buffer exists and every kernel has loaded first, with no peer involved.
    """
    return [st for st in seed_stages(ring) if st.kind == "device"]


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


def read_sensors(tick: Tick) -> None:
    """The sensor fan-in: one D2H per sensor with a device stage into the shared ``Measurement``.

    The ``Measurement`` holds one slot per kind of sensor. The sensors read in reverse order, so of two
    sensors of one kind the first declared writes last and fills the slot.
    """
    for s in reversed(tick.sensors):
        s.read(tick.meas)


def peer_stages(controller) -> list[Stage]:
    """The stages of a controller that blocks on a peer or solves on the host: ``bind``, a device
    stage with no kernel, binds ``Tick.controls`` to a persistent ``(1, 16)`` device command buffer in
    the warm pass, so the command stages capture over it before the peer connects; ``read`` fans
    the sensors into the ``Measurement``; ``exchange`` runs the controller's ``exchange`` and copies
    its commands into the buffer. ``None`` from the exchange reads as the peer not answering, which
    the stage reports by returning ``False``.
    """
    import warp as wp

    buf = wp.zeros((1, CHANNELS), dtype=float)
    cmd = np.zeros((1, CHANNELS), dtype=np.float32)

    def bind(tick):
        tick.controls = buf

    def exchange(tick):
        controls = controller.exchange(tick.meas, tick.t, tick.timeout)
        if controls is None:
            return False
        command = np.asarray(controls.command, dtype=np.float32).reshape(-1)
        if command.shape[0] > CHANNELS:
            raise ValueError(
                f"{type(controller).__name__} sent {command.shape[0]} commands, over the {CHANNELS} channels"
            )
        cmd[0, : command.shape[0]] = command
        buf.assign(cmd)
        tick.controls = buf
        return True

    return [Stage("bind", "device", bind), Stage("read", "host", read_sensors), Stage("exchange", "host", exchange)]


__all__ = [
    "Bound",
    "Segment",
    "build_ring",
    "device_sensors",
    "partition",
    "peer_stages",
    "plan_line",
    "read_sensors",
    "seed_stages",
    "warm_stages",
]
