"""The command elements: each turns the controller's command into Newton's control inputs.

A command element's stage runs once per physics substep, after the physics ``clear`` stage and before the force
stages, and writes the targets and feedforward the model's ``newton.Control`` holds; physics then steps
every Newton actuator the vehicle declares before its solver. The shipped one is the rotor's ESC,
:class:`~nexus_uav._src.vehicle.commands.esc.Esc`, one per rotor: the controller's normalized command on the
rotor's channel to its motor's speed target and drag feedforward. A component that writes another actuator's
target, a gimbal servo's, is one more class here.
"""

from .esc import Esc, Escs

__all__ = ["Esc", "Escs"]
