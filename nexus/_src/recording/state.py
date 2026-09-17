"""The physics-state observation channels: every body → ``BodyState``, every joint → ``JointState``.

Physics registers one channel per body, ``physics/body/<label>``, and one per joint,
``physics/joint/<label>``, from the finalized model's labels, so the full Newton model state is
addressable by name: ``sim.physics["body_frd"]`` is the base body, ``sim.physics["rotor_1_joint"]``
an actuator joint. Each tick a capturable kernel
snapshots that entity into its channel's device ring buffer, with no D2H, and the decode reconstructs the
typed state at read time.

Layouts match Newton's native arrays:
* body: ``body_q[i]`` = ``[px,py,pz, qx,qy,qz,qw]``, world frame, quaternion in x, y, z, w order;
  ``body_qd[i]`` = ``[lin(0:3), ang(3:6)]``.
* joint: ``joint_q[q_start[j]:q_start[j+1]]``, the generalized coords, plus ``joint_qd[...]``, the
  generalized vels: variable width, a free base = 7q/6qd, a revolute = 1q/1qd.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

# Body row layout, 14 floats: t, pos(3), quat xyzw(4), lin vel(3), ang vel(3).
BODY_WIDTH = 14
# The declared quantity schema, name → column count after the implicit t. The registration passes
# it so every downstream consumer, history_arrays and the Rerun debug dump and tabs, derives from one
# source. Names match the BodyState dataclass attributes.
BODY_FIELDS = (("position", 3), ("quat_xyzw", 4), ("velocity", 3), ("angular_velocity", 3))


@dataclass(slots=True)
class BodyState:
    """Ground-truth state of one rigid body in the world frame: Newton Forward-Left-Up (FLU), Z-up, with
    quats in x, y, z, w order.
    """

    t: float
    """Sim time of the snapshot, in seconds: the lockstep clock, counter × dt."""
    position: tuple[float, float, float]
    """Body origin position ``(x, y, z)`` in the world frame, in meters, Z-up FLU."""
    quat_xyzw: tuple[float, float, float, float]
    """Body orientation as a quaternion in ``(x, y, z, w)`` order, world frame."""
    velocity: tuple[float, float, float]
    """Body linear velocity ``(vx, vy, vz)`` in the world frame, in meters per second."""
    angular_velocity: tuple[float, float, float]
    """Body angular velocity ``(wx, wy, wz)``, in radians per second."""

    @property
    def altitude_m(self) -> float:
        """Height over the world origin plane, in meters; the world frame is Z-up, so ``position[2]``."""
        return self.position[2]


@dataclass(slots=True)
class JointState:
    """State of one joint: its generalized coordinates ``q`` and velocities ``qd``, of variable width."""

    t: float
    """Sim time of the snapshot, in seconds: the lockstep clock, counter × dt."""
    q: tuple[float, ...]
    """Generalized coordinates, for example a revolute angle ``(θ,)`` or a free base ``(px,py,pz, qx,qy,qz,qw)``."""
    qd: tuple[float, ...]
    """Generalized velocities, for example a revolute rate ``(θ̇,)`` or a free base ``(vx,vy,vz, wx,wy,wz)``."""


@wp.kernel
def record_body(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    i: int,
    dt: float,
    maxlen: int,
    buf: wp.array(dtype=float, ndim=2),
    counter: wp.array(dtype=int),
):
    c = counter[0]  # total rows so far; advances per graph replay, on-device, not a frozen kernel arg
    s = c % maxlen
    tf = body_q[i]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)  # xyzw
    vd = body_qd[i]  # [lin(0:3), ang(3:6)]
    buf[s, 0] = dt * wp.float32(c)
    buf[s, 1] = p[0]
    buf[s, 2] = p[1]
    buf[s, 3] = p[2]
    buf[s, 4] = q[0]
    buf[s, 5] = q[1]
    buf[s, 6] = q[2]
    buf[s, 7] = q[3]
    buf[s, 8] = vd[0]
    buf[s, 9] = vd[1]
    buf[s, 10] = vd[2]
    buf[s, 11] = vd[3]
    buf[s, 12] = vd[4]
    buf[s, 13] = vd[5]
    counter[0] = c + 1


@wp.kernel
def record_joint(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    nq: int,
    qd_start: int,
    nqd: int,
    dt: float,
    maxlen: int,
    buf: wp.array(dtype=float, ndim=2),
    counter: wp.array(dtype=int),
):
    c = counter[0]
    s = c % maxlen
    buf[s, 0] = dt * wp.float32(c)
    for k in range(nq):  # generalized coords: variable width per joint type, a runtime loop
        buf[s, 1 + k] = joint_q[q_start + k]
    for k in range(nqd):  # generalized velocities
        buf[s, 1 + nq + k] = joint_qd[qd_start + k]
    counter[0] = c + 1


def decode_body(r) -> BodyState:
    """Decode one body ring-buffer row, 14 floats, into a :class:`BodyState`."""
    return BodyState(
        t=float(r[0]),
        position=(float(r[1]), float(r[2]), float(r[3])),
        quat_xyzw=(float(r[4]), float(r[5]), float(r[6]), float(r[7])),
        velocity=(float(r[8]), float(r[9]), float(r[10])),
        angular_velocity=(float(r[11]), float(r[12]), float(r[13])),
    )


def make_decode_joint(nq: int, nqd: int) -> Callable[[np.ndarray], JointState]:
    """Build the row→:class:`JointState` decode for a joint of width ``nq`` coords + ``nqd`` vels."""

    def decode(r) -> JointState:
        return JointState(
            t=float(r[0]),
            q=tuple(float(x) for x in r[1 : 1 + nq]),
            qd=tuple(float(x) for x in r[1 + nq : 1 + nq + nqd]),
        )

    return decode
