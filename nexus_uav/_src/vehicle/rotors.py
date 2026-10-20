"""Shared rotor-layout discovery + small geometry helpers: the one home for these, imported by the rotors'
ESCs, the propellers, the examples' mixer and coupling, ``VehicleUsd`` and the RL task, so there is no
per-consumer copy.

* :data:`RPM_PER_RADS`: the rad/s → rpm factor. The authored unit of ``ct`` is N/min², so ``thrust = ct·rpm²``.
* :func:`rotor_of`: find one rotor in a finalized model by its rigid body's prim path: the Degrees Of
  Freedom (DOF) and coord index of its revolute joint, its body and the joint's parent body, so a declared
  ESC or propeller finds its rotor from its own prim, with no list of joints.
* :func:`motor_speed`: the no-load speed of the Newton motor that drives a DOF, the rotor speed at full command,
  and :func:`rotor_speeds`, the same for each rotor a vehicle declares: the model is the one source of it.
* :func:`find_rotor_joints`: find the joints the vehicle declares its rotors on in a finalized model,
  returning the DOF and coord indices, the rotor child bodies and the shared airframe
  parent, so the rotor indexing comes from the declaration and the model rather than an assumption, and
  stays correct regardless of what the ground plane, the scene or another joint added.
* :func:`quat_to_R`: rotation matrix from a Newton quaternion in ``x, y, z, w`` order, the host twin of
  ``wp.quat`` → matrix.
"""

from __future__ import annotations

import math

import numpy as np

RPM_PER_RADS = 60.0 / (2.0 * math.pi)  # rad/s → rpm; ct is N/min², so thrust = ct·rpm²


def rotor_of(model, body: str) -> tuple[int, int, int, int]:
    """Find the rotor on the rigid body at the prim path `body` in a built model.

    Returns:
        ``(vel_dof, pos_coord, body, parent)``: the DOF and coord index of the revolute joint whose child body it
        is, the body's index, and the joint's parent body, the airframe.

    Raises:
        ValueError: The model has no body of that path, or no joint has it as child body; the message names the
            prim.
    """
    labels = [str(label) for label in model.body_label]
    if body not in labels:
        raise ValueError(f"{body}: the model has no rigid body of that path")
    index = labels.index(body)
    child = model.joint_child.numpy()
    joints = [j for j in range(len(child)) if int(child[j]) == index]
    if not joints:
        raise ValueError(f"{body}: no revolute joint has it as child body, so it is no rotor")
    j = joints[0]
    return (
        int(model.joint_qd_start.numpy()[j]),
        int(model.joint_q_start.numpy()[j]),
        index,
        int(model.joint_parent.numpy()[j]),
    )


def motor_speed(model, dof: int, body: str) -> float:
    """The no-load speed [rad/s] of the Newton motor that drives `dof`, its DC motor clamp's velocity limit, the
    rotor speed at full command. `body` names the rotor in the error.

    Raises:
        ValueError: No Newton motor drives the DOF, or its motor has no finite velocity limit; the message
            names the prim.
    """
    for motor in model.actuators or []:
        indices = motor.indices.numpy().tolist()
        if dof not in indices:
            continue
        k = indices.index(dof)
        for clamp in motor.clamping:
            limit = getattr(clamp, "velocity_limit", None)
            if limit is not None:
                speed = float(limit.numpy()[k])
                if math.isfinite(speed):
                    return speed
        raise ValueError(
            f"{body}: the Newton motor on its joint has no finite velocity limit, the rotor speed at full command; "
            "author the motor's no-load speed as newton:velocityLimit"
        )
    raise ValueError(f"{body}: no NewtonActuator prim drives its joint, so it has no motor")


def rotor_speeds(model, rotors) -> np.ndarray:
    """Each rotor's speed at full command [rpm], its motor's no-load speed from the model, in the order of
    `rotors`, the rotors the vehicle declares, each with its rigid body's prim path as ``body``.

    Raises:
        ValueError: As :func:`rotor_of` and :func:`motor_speed` state; the message names the prim.
    """
    return np.array([motor_speed(model, rotor_of(model, r.body)[0], r.body) * RPM_PER_RADS for r in rotors])


def find_rotor_joints(model, joints) -> tuple[list[int], list[int], list[int], int]:
    """Find the rotor joints in a built model: the ones whose labels are ``joints``, the joint paths the
    vehicle declares its rotors on, in the model's order. Returns
    ``(rotor_vel_dofs, rotor_pos_coords, rotor_bodies, base_body)``: the DOF/coord indices, the rotor
    child-body indices for the per-rotor geometry, and the shared parent body index, the airframe.

    Raises:
        ValueError: The model has none of the declared joints, as a model collapsed to a single body.
    """
    declared = {str(joint) for joint in joints}
    qd_start = model.joint_qd_start.numpy()
    q_start = model.joint_q_start.numpy()
    child = model.joint_child.numpy()
    parent = model.joint_parent.numpy()
    vel_dofs, pos_coords, bodies, parents = [], [], [], []
    for j, label in enumerate(model.joint_label):
        if label not in declared:
            continue
        vel_dofs.append(int(qd_start[j]))
        pos_coords.append(int(q_start[j]))
        bodies.append(int(child[j]))
        parents.append(int(parent[j]))
    if not vel_dofs:
        raise ValueError(f"the model has none of the declared rotor joints {sorted(declared)}")
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
