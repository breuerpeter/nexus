"""The rotor chain of many copies of one vehicle in one model: the plant the RL task trains on.

A run's loop steps the rotor chain around each solver step: each rotor's ESC sets its Newton motor's speed
target and drag feedforward, each propeller adds its rotor's thrust and drag to its body, and physics steps the
Newton motors into the rotor joints' torque. :class:`RotorChain` runs the same core parts in the same order for a
model that holds one copy of the vehicle per environment, as the RL task's model does, so training flies the
plant every run flies, each rotor with its own values.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_uav._src.physics.actuators import Actuators
from nexus_uav._src.vehicle.commands import Escs, Motors
from nexus_uav._src.vehicle.forces import Propellers
from nexus_uav._src.vehicle.rotors import command_lines, rotor_speeds

__all__ = ["RotorChain", "vehicle_rotors"]


def vehicle_rotors(model, joints) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the rotors of every copy of one vehicle in a model.

    A copy's rotor joint is a joint whose label ends with a declared joint's path below the vehicle's root prim:
    ``/Physics/rotor_0_joint`` for ``/astro_max/Physics/rotor_0_joint``.

    Args:
        model: The finalized model.
        joints: The joint path of each rotor the vehicle's Universal Scene Description (USD) file declares, each
            rotor's ``joint`` of :func:`~nexus_uav.examples._lib.single_body.rotors`.

    Returns:
        Each rotor's velocity degree of freedom, its body and its airframe body, each ``(copies, rotors)``,
        the copies in the model's order and each copy's rotors in the declared order.

    Raises:
        ValueError: The model doesn't hold every declared rotor joint the same number of times, at least once.
    """
    tails = ["/" + str(joint).split("/", 2)[2] for joint in joints]  # each joint's path below the root prim
    found: dict[str, list[int]] = {tail: [] for tail in tails}
    for j, label in enumerate(model.joint_label):
        for tail in tails:
            if str(label).endswith(tail):
                found[tail].append(j)
                break
    counts = {len(found[tail]) for tail in tails}
    if len(counts) != 1 or 0 in counts:
        raise ValueError(
            f"the model must hold each declared rotor joint once per copy; found { ({t: len(found[t]) for t in tails}) }"
        )
    index = np.array([found[tail] for tail in tails]).T  # (copies, rotors), copies in the model's order
    return (
        model.joint_qd_start.numpy()[index],
        model.joint_child.numpy()[index],
        model.joint_parent.numpy()[index],
    )


class RotorChain:
    """The rotor chain of every copy of one vehicle in a model: the core's ESCs, propellers and Newton motors,
    stepped in the order a run steps them, each rotor with its own values.

    Args:
        model: The finalized model, one copy of the vehicle per environment, with each copy's Newton motors on
            ``model.actuators``.
        rotors: The rotors the vehicle's USD file declares, in channel order,
            :func:`~nexus_uav.examples._lib.single_body.rotors`: each with its joint, its channel and its
            propeller's values. Each motor's no-load speed comes from the model, in ``rpm_max``.
    """

    def __init__(self, model, rotors):
        dofs, bodies, bases = vehicle_rotors(model, [rotor.joint for rotor in rotors])
        self.copies, self.nr = dofs.shape
        self.rotor_dofs, self.rotor_bodies, self.rotor_bases = dofs, bodies, bases  # each (copies, nr)
        self.bases = wp.array(bases[:, 0], dtype=wp.int32, device=model.device)  # each copy's airframe body
        values = {key: [rotor.propeller[key] for rotor in rotors] for key in ("ct", "cd", "aero_h", "aero_hforce")}
        self.rpm_max = rotor_speeds(rotors)  # each rotor's loaded speed at full command [rpm]
        self.command_lines = command_lines(rotors)  # (nr, 3) each rotor's u = a·Ω + b·Ω² − c
        self.command = Escs(
            channels=[rotor.channel for rotor in rotors],
            idle_duty=[rotor.esc["idle_duty"] for rotor in rotors],
            max_duty=[rotor.esc["max_duty"] for rotor in rotors],
            device=model.device,
        )
        # Each copy's bus, the ESCs' own, since no battery writes one; each rotor's voltage and current.
        bus = np.tile([rotor.esc["bus_voltage"] for rotor in rotors], (self.copies, 1))
        self.bus = wp.array(bus.astype(np.float32), dtype=float, device=model.device)
        self.voltage = wp.zeros((self.copies, self.nr), dtype=float, device=model.device)
        self.current = wp.zeros((self.copies, self.nr), dtype=float, device=model.device)
        self.motors = Motors(
            rotor_vel_dofs=wp.array(dofs, dtype=wp.int32, device=model.device),
            **{key: [rotor.motor[key] for rotor in rotors] for key in ("kv", "resistance", "current_limit")},
        )
        self.propellers = Propellers(
            rotor_vel_dofs=wp.array(dofs.ravel(), dtype=wp.int32, device=model.device),
            rotor_bodies=wp.array(bodies.ravel(), dtype=wp.int32, device=model.device),
            rotor_bases=wp.array(bases.ravel(), dtype=wp.int32, device=model.device),
            **{key: np.tile(value, self.copies) for key, value in values.items()},  # each copy's rotors, in order
        )
        self.actuators = Actuators(model)

    def step(self, state, control, cmd: wp.array, dt: float) -> None:
        """Run the rotor chain before a solver step of ``dt``: each rotor's terminal voltage from ``cmd``, one row
        of normalized commands per copy, each motor's torque into ``control.joint_f``, zeroed first, each
        propeller's force from its rotor's speed in ``state``, then the model's own actuators' torque.
        """
        control.joint_f.zero_()
        self.command.write(cmd, self.bus, self.voltage)
        self.motors.step(self.voltage, state, control, dt, self.current)
        self.propellers.add(state)
        self.actuators.step(state, control, dt)

    def reset(self, dofs: wp.array | None = None) -> None:
        """Zero the state of the model's actuators that drive ``dofs``, a mask over the model's degrees of freedom,
        or of every one when ``dofs`` is ``None``; the rotors' motors keep no state.
        """
        self.actuators.reset(dofs)
