"""Signals: the typed values that pass between the loop's components on the tick.

A stage declares each signal it reads and each signal it writes, by name and type. The builder wires
every input to an output of its name before the capture, checks that the two agree, and hands the writer
and each reader one buffer. Where several components write one signal, a reader takes the one a connection
on its prim names, or all of them as a list. A signal whose type is Warp's own, a struct or a value type
such as ``wp.float32`` or ``wp.vec3``, lives on the device: its buffer is a Warp array, which device stages
read and write in their kernels and a host stage reads with a copy. Any other type is a host type, whose
buffer holds one object.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .stages import Bound


class HostBuffer:
    """A host signal's buffer: one object, which the writer replaces and each reader reads."""

    __slots__ = ("value",)

    def __init__(self, value: Any = None):
        self.value = value


class Signal:
    """One value a stage reads or writes: a name, a type and, for a device type, a shape.

    Args:
        name: The name that wires a reader to its writer.
        type: A Warp struct or value type, whose buffer lives on the device, or any other class, whose
            buffer holds one object on the host. A reader that takes every writer of the signal declares
            ``list[T]`` and reads a list, one value per writer.
        shape: The device buffer's shape. A reader reads the leading part of a wider buffer.
        default: The value a reader takes until a component writes the signal, and for good when no
            component writes it. ``None`` gives no default.
    """

    buffer: Any
    """The one buffer the builder hands the writer and every reader, before the capture; for a list input,
    the buffer of each writer."""

    def __init__(self, name: str, type: type, *, shape: tuple[int, ...] | None = None, default: Any = None):
        self.name = name
        self.type = type
        self.shape = None if shape is None else tuple(int(n) for n in shape)
        self.default = default
        self.buffer = None

    @property
    def device(self) -> bool:
        """Whether the signal's buffer lives on the device: its type is a Warp struct or value type."""
        return wp.types.type_is_struct(self.type) or wp.types.type_is_value(self.type)

    def read(self) -> Any:
        """The signal's value on the host: a copy of a device buffer, or the object a host buffer holds. A
        list input reads one value per writer, or its default when no component writes the signal. The copy
        is explicit, since Warp's ``numpy()`` shares a buffer's memory on the CPU device.
        """
        if typing.get_origin(self.type) is list:
            return [_value(buffer) for buffer in self.buffer] if self.buffer else self.default
        return _value(self.buffer)

    def write(self, value: Any) -> None:
        """Write the signal's value from the host, in place: a device buffer takes a copy of ``value``, an
        array of the signal's shape, so the next graph replay reads it, and a host buffer holds ``value``.
        A struct's value is a record per element, its fields in order, such as ``[(time, accel, gyro)]``.
        """
        if self.device:
            dtype = self.type.numpy_dtype() if wp.types.type_is_struct(self.type) else None
            self.buffer.assign(np.asarray(value, dtype=dtype))
        else:
            self.buffer.value = value

    def _allocate(self) -> Any:
        """A new buffer from the signal's type: zeros on the current Warp device, or an empty host buffer."""
        return wp.zeros(self.shape, dtype=self.type) if self.device else HostBuffer()


def _value(buffer: Any) -> Any:
    """A buffer's value on the host: a copy of a Warp array, or the object a host buffer holds."""
    return buffer.numpy().copy() if isinstance(buffer, wp.array) else buffer.value


def _owner(bound: Bound) -> str:
    """The instance name of the component that states a stage: its ``name``, else its class's."""
    return str(getattr(bound.component, "name", None) or type(bound.component).__name__)


def _prim(bound: Bound) -> str | None:
    """The path of the prim that declares the component that states a stage, or ``None``."""
    return getattr(bound.component, "prim_path", None)


def _type(signal_type: Any) -> str:
    """The name of a signal's type, as an error names it."""
    if typing.get_origin(signal_type) is list:
        return f"list of {_type(typing.get_args(signal_type)[0])}"
    return getattr(signal_type, "__name__", str(signal_type))


def wire(ring: list[Bound], connections: Mapping[str, Mapping[str, str]] | None = None) -> None:
    """Wire every signal the ring's stages declare, before any stage runs, and check each pair.

    The builder allocates one buffer per output and hands it to its writer. An input takes the buffer of the
    one writer of its name, or of the writer a connection on its reader's prim names; a list input takes
    every writer's buffer, in ring order. A reader's default fills the buffer it takes until its writer
    first writes. An input that no component writes reads its own buffer, which holds its default.

    Args:
        ring: The stages of one tick.
        connections: For a reader's prim, each signal it reads and the prim whose component writes it, from
            the relationships ``nexus:inputs:<signal>`` the vehicle authors.

    Raises:
        ValueError: A device stage declares a host signal, a device signal has no shape, two components
            write one signal that a third reads through neither a connection nor a list, a connection names
            a prim whose component writes no such signal, a reader and its writer disagree on the type, a
            reader needs more of an axis than its writer's buffer holds, or an input has neither a writer nor
            a default. The message names both ends, or the stage or the reader and the signal.
    """
    writes, reads = _declared(ring)
    for written in writes.values():
        for _, output in written:
            output.buffer = output._allocate()
    defaulted: set[int] = set()  # the outputs a reader's default already fills
    for name, readers in reads.items():
        written = writes.get(name, [])
        for bound, signal in readers:
            if typing.get_origin(signal.type) is list:
                _take_all(bound, signal, written)
            else:
                connected = (connections or {}).get(_prim(bound) or "", {}).get(name)
                _take_one(bound, signal, written, connected, defaulted)


def _declared(ring: list[Bound]) -> tuple[dict, dict]:
    """Every signal the ring's stages write and read, by name, each beside the stage that declares it.

    Raises:
        ValueError: A device stage declares a host signal, or a device signal has no shape.
    """
    writes: dict[str, list[tuple[Bound, Signal]]] = {}
    reads: dict[str, list[tuple[Bound, Signal]]] = {}
    seen: set[int] = set()
    for bound in ring:
        if id(bound) in seen:  # the ring repeats the stages of each physics substep
            continue
        seen.add(id(bound))
        for verb, table, signals in (("reads", reads, bound.stage.reads), ("writes", writes, bound.stage.writes)):
            for signal in signals:
                if bound.stage.kind == "device" and not signal.device:
                    raise ValueError(
                        f"{_owner(bound)}'s device stage {bound.stage.name!r} {verb} {signal.name!r}, a "
                        f"{_type(signal.type)}, which lives on the host: a device stage reads and writes device "
                        "signals only"
                    )
                if signal.device and signal.shape is None:
                    raise ValueError(f"{_owner(bound)} declares the device signal {signal.name!r} with no shape")
                table.setdefault(signal.name, []).append((bound, signal))
    return writes, reads


def _take_one(bound: Bound, signal: Signal, written: list, connected: str | None, defaulted: set[int]) -> None:
    """Hand an input the buffer of its writer: the one its connection names, else the one there is, else a
    buffer of its own that holds its default.
    """
    name = signal.name
    if connected is not None:
        picked = [(writer, output) for writer, output in written if _prim(writer) == connected]
        if not picked:
            raise ValueError(
                f"{_owner(bound)} connects {name!r} to {connected}, which writes no {name!r}: a connection names "
                "the prim of a component that writes the signal"
            )
        writer, output = picked[0]
    elif len(written) == 1:
        writer, output = written[0]
    elif written:
        raise ValueError(
            f"{' and '.join(_prim(w) or _owner(w) for w, _ in written)} all write {name!r}, which {_owner(bound)} "
            f"reads: a connection on the reader's prim, the relationship nexus:inputs:{name} its schema declares, "
            "picks one, or a list input takes them all"
        )
    else:
        _unwritten(bound, signal)
        signal.buffer = signal._allocate()
        signal.write(signal.default)
        return
    _check(writer, output, bound, signal, signal.type)
    signal.buffer = output.buffer
    if signal.default is not None and id(output) not in defaulted:
        defaulted.add(id(output))
        signal.write(signal.default)


def _take_all(bound: Bound, signal: Signal, written: list) -> None:
    """Hand a list input every writer's buffer, in ring order, or none when no component writes it."""
    if not written:
        _unwritten(bound, signal)
    (element,) = typing.get_args(signal.type)
    for writer, output in written:
        _check(writer, output, bound, signal, element)
    signal.buffer = [output.buffer for _, output in written]


def _unwritten(bound: Bound, signal: Signal) -> None:
    """Check that an input no component writes gives a default.

    Raises:
        ValueError: It gives none; the message names the reader and the signal.
    """
    if signal.default is None:
        raise ValueError(
            f"{_owner(bound)} reads {signal.name!r}, a {_type(signal.type)}, which no component writes, and it "
            "gives no default"
        )


def _check(writer: Bound, output: Signal, bound: Bound, signal: Signal, expected: Any) -> None:
    """Check that a reader takes what its writer writes: a value of type ``expected``, and no more of an axis
    than the writer's buffer holds.

    Raises:
        ValueError: The two disagree; the message names both ends and the signal.
    """
    name = signal.name
    if output.type is not expected:
        raise ValueError(
            f"{_owner(writer)} writes {name!r} as {_type(output.type)}, and {_owner(bound)} reads it as "
            f"{_type(signal.type)}: a reader and its writer agree on the type"
        )
    if output.device and not _fits(output.shape, signal.shape or ()):
        raise ValueError(
            f"{_owner(writer)} writes {name!r} with shape {output.shape}, and {_owner(bound)} reads "
            f"{signal.shape}: a reader reads the leading part of its writer's buffer"
        )


def _fits(written: tuple[int, ...], read: tuple[int, ...]) -> bool:
    """Whether a reader of shape ``read`` takes the leading part of a buffer of shape ``written``."""
    return len(written) == len(read) and all(w >= r for w, r in zip(written, read, strict=True))


__all__ = ["HostBuffer", "Signal", "wire"]
