"""The plant's histories: every body → ``BodyState``, every joint → ``JointState``.

The Recorder registers one history per body, ``vehicle/body/<label>``, and one per joint,
``vehicle/joints/<label>``, from the finalized model's labels, so the full Newton model state is
addressable by name: ``sim.physics["body_frd"]`` is the base body, ``sim.physics["rotor_1_joint"]``
an actuator joint. Each tick a device-only kernel snapshots that entity into its history's staging
buffers, its values and its time, with no D2H, and the decode reconstructs the typed state at read time.

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

# Body row layout, 13 floats: pos(3), quat xyzw(4), lin vel(3), ang vel(3); the time rides beside the row.
BODY_WIDTH = 13
# The declared quantity schema, name → column count. The registration passes it so every downstream
# consumer, history_arrays and the Rerun debug dump and tabs, derives from one source. Names match the
# BodyState dataclass attributes.
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
    dt: wp.float64,
    staging: int,
    values: wp.array(dtype=float, ndim=2),
    times: wp.array(dtype=wp.float64),
    counter: wp.array(dtype=int),
):
    c = counter[0]  # total rows so far; advances per graph replay, on-device, not a frozen kernel arg
    s = c % staging  # the staging slot: the host drains the buffer before the slot comes round again
    tf = body_q[i]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)  # xyzw
    vd = body_qd[i]  # [lin(0:3), ang(3:6)]
    times[s] = dt * wp.float64(c)
    values[s, 0] = p[0]
    values[s, 1] = p[1]
    values[s, 2] = p[2]
    values[s, 3] = q[0]
    values[s, 4] = q[1]
    values[s, 5] = q[2]
    values[s, 6] = q[3]
    values[s, 7] = vd[0]
    values[s, 8] = vd[1]
    values[s, 9] = vd[2]
    values[s, 10] = vd[3]
    values[s, 11] = vd[4]
    values[s, 12] = vd[5]
    counter[0] = c + 1


@wp.kernel
def record_joint(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    nq: int,
    qd_start: int,
    nqd: int,
    dt: wp.float64,
    staging: int,
    values: wp.array(dtype=float, ndim=2),
    times: wp.array(dtype=wp.float64),
    counter: wp.array(dtype=int),
):
    c = counter[0]
    s = c % staging
    times[s] = dt * wp.float64(c)
    for k in range(nq):  # generalized coords: variable width per joint type, a runtime loop
        values[s, k] = joint_q[q_start + k]
    for k in range(nqd):  # generalized velocities
        values[s, nq + k] = joint_qd[qd_start + k]
    counter[0] = c + 1


def decode_body(t: float, r: np.ndarray) -> BodyState:
    """Decode one body row, its time and 13 floats, into a :class:`BodyState`."""
    return BodyState(
        t=float(t),
        position=(float(r[0]), float(r[1]), float(r[2])),
        quat_xyzw=(float(r[3]), float(r[4]), float(r[5]), float(r[6])),
        velocity=(float(r[7]), float(r[8]), float(r[9])),
        angular_velocity=(float(r[10]), float(r[11]), float(r[12])),
    )


def make_decode_joint(nq: int, nqd: int) -> Callable[[float, np.ndarray], JointState]:
    """Build the row→:class:`JointState` decode for a joint of width ``nq`` coords + ``nqd`` vels."""

    def decode(t: float, r: np.ndarray) -> JointState:
        return JointState(
            t=float(t),
            q=tuple(float(x) for x in r[:nq]),
            qd=tuple(float(x) for x in r[nq : nq + nqd]),
        )

    return decode
