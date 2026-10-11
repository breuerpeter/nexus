"""The plant's outputs: a ``PoseTwist`` per body and a struct per joint, written from Newton's state.

Newton keeps the plant's state in four arrays, ``body_q`` and ``body_qd`` with one row per body, and
``joint_q`` and ``joint_qd`` with every joint's coordinates end to end. The plant publishes that state as
signals, one per body and one per joint, as a Drake plant evaluates its output ports after a step: the
physics' ``state`` stage runs :meth:`Outputs.write` once per tick, after the last substep. A body's signal
takes the name ``body/<label>`` and a joint's ``joints/<label>``, the label its leaf name when unique, so under
the plant's path, ``vehicle``, the Recorder keys them ``vehicle/body/<label>`` and ``vehicle/joints/<label>``.
A joint's struct has ``q`` and ``qd`` as vectors of that joint's widths, one type per width, since a Warp
struct's fields have fixed widths. A joint with no coordinate, a fixed joint, writes none.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp

from nexus_sim._src.core.labels import leaf_keys
from nexus_sim._src.core.schema import PoseTwist
from nexus_sim._src.core.signals import Signal


@cache
def joint_struct(nq: int, nqd: int) -> type:
    """The Warp struct of a joint with ``nq`` coordinates and ``nqd`` velocities: ``q`` and ``qd`` as fixed
    vectors of float32. One type per width, so two joints of one width share it and its kernel.
    """
    fields = {
        "q": wp.types.vector(length=nq, dtype=wp.float32),
        "qd": wp.types.vector(length=nqd, dtype=wp.float32),
    }
    return wp.struct(type(f"Joint_{nq}_{nqd}", (), {"__annotations__": fields, "__module__": __name__}))


@wp.kernel
def write_body(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    i: int,
    out: wp.array(dtype=PoseTwist),
):
    """Write body ``i``'s pose and twist, world axes, into ``out``."""
    tf = body_q[i]
    vd = body_qd[i]  # [linear, angular]
    p = PoseTwist()
    p.position = wp.transform_get_translation(tf)
    p.orientation = wp.transform_get_rotation(tf)
    p.linear_velocity = wp.spatial_top(vd)
    p.angular_velocity = wp.spatial_bottom(vd)
    out[0] = p


@wp.kernel
def write_joint(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    nq: int,
    qd_start: int,
    nqd: int,
    out: wp.array(dtype=Any),
):
    """Write one joint's coordinates and velocities, its slices of the flat arrays, into ``out``, a struct of
    that joint's widths.
    """
    s = out[0]
    for k in range(nq):
        s.q[k] = joint_q[q_start + k]
    for k in range(nqd):
        s.qd[k] = joint_qd[qd_start + k]
    out[0] = s


class Outputs:
    """The signals a plant writes, from its finalized model.

    ``signals`` lists every output in order, the bodies then the joints, for the ``state`` stage to declare.
    ``by_label`` maps each model label, a body's or joint's prim path, to its signal, for the builder to
    resolve a connection whose target is a body's or joint's prim.

    Args:
        model: The finalized Newton model.
        state: A state of the model; one whose ``joint_q`` is ``None`` has no joint outputs.
    """

    def __init__(self, model, state):
        self._bodies: list[tuple[Signal, int]] = []
        self._joints: list[tuple[Signal, int, int, int, int]] = []
        self.by_label: dict[str, Signal] = {}
        labels = [str(label) for label in model.body_label]
        for i, (label, key) in enumerate(zip(labels, leaf_keys(labels), strict=True)):
            signal = Signal(f"body/{key}", PoseTwist, shape=(1,))
            self._bodies.append((signal, i))
            self.by_label[label] = signal
        if model.joint_count and getattr(state, "joint_q", None) is not None:
            labels = [str(label) for label in model.joint_label]
            qs, qds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
            for j, (label, key) in enumerate(zip(labels, leaf_keys(labels), strict=True)):
                nq, nqd = int(qs[j + 1] - qs[j]), int(qds[j + 1] - qds[j])
                if nq == 0 and nqd == 0:
                    continue  # a fixed joint has no state to publish
                signal = Signal(f"joints/{key}", joint_struct(nq, nqd), shape=(1,))
                self._joints.append((signal, int(qs[j]), nq, int(qds[j]), nqd))
                self.by_label[label] = signal

    @property
    def signals(self) -> tuple[Signal, ...]:
        """Every output, the bodies then the joints."""
        return (*(s for s, _ in self._bodies), *(s for s, *_ in self._joints))

    def write(self, state) -> None:
        """Write every output from ``state``: one launch per body and per joint, device-only, so the stage
        joins the captured graph.
        """
        for signal, i in self._bodies:
            wp.launch(write_body, dim=1, inputs=(state.body_q, state.body_qd, i), outputs=(signal.buffer,))
        for signal, qs, nq, qds, nqd in self._joints:
            wp.launch(
                write_joint,
                dim=1,
                inputs=(state.joint_q, state.joint_qd, qs, nq, qds, nqd),
                outputs=(signal.buffer,),
            )


__all__ = ["Outputs", "joint_struct", "write_body", "write_joint"]
