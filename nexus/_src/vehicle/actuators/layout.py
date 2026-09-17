"""Shared rotor-layout discovery + small geometry helpers: the one home for these, imported by the
actuator coupling, the mixer, the physics builders, the runtimes and the viz, so there is no per-consumer
copy.

* :data:`RPM_PER_RADS`: the rad/s → rpm factor. The authored unit of ``ct`` is N/min², so ``thrust = ct·rpm²``.
* :func:`find_rotor_joints`: discover the rotor REVOLUTE joints in a finalized model, returning the
  Degrees Of Freedom (DOF) and coord indices, the rotor child bodies and the shared airframe parent, so the
  rotor indexing comes from the model rather than an assumption and stays correct regardless of what the
  ground plane / scene added before the vehicle.
* :func:`quat_to_R`: rotation matrix from a Newton quaternion in ``x, y, z, w`` order, the host twin of
  ``wp.quat`` → matrix.
"""

from __future__ import annotations

import math

import numpy as np

RPM_PER_RADS = 60.0 / (2.0 * math.pi)  # rad/s → rpm; ct is N/min², so thrust = ct·rpm²


def find_rotor_joints(model) -> tuple[list[int], list[int], list[int], int]:
    """Discover the rotor REVOLUTE joints in a built model. Returns
    ``(rotor_vel_dofs, rotor_pos_coords, rotor_bodies, base_body)``: the DOF/coord indices, the rotor
    child-body indices for the per-rotor geometry, and the shared parent body index, the airframe.
    """
    import newton

    revolute = int(newton.JointType.REVOLUTE)
    jt = model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    q_start = model.joint_q_start.numpy()
    child = model.joint_child.numpy()
    parent = model.joint_parent.numpy()
    vel_dofs, pos_coords, bodies, parents = [], [], [], []
    for j in range(len(jt)):
        if int(jt[j]) != revolute:
            continue
        vel_dofs.append(int(qd_start[j]))
        pos_coords.append(int(q_start[j]))
        bodies.append(int(child[j]))
        parents.append(int(parent[j]))
    if not vel_dofs:
        raise ValueError("no REVOLUTE rotor joints found in the model")
    base_body = parents[0]  # all rotors share the airframe as their parent
    return vel_dofs, pos_coords, bodies, base_body


def quat_to_R(q) -> np.ndarray:
    """Rotation matrix from a Newton quaternion in ``x, y, z, w`` order, the host twin of ``wp.quat`` → matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
