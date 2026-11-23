"""The command elements: each turns the controller's command into the signal an actuator takes.

A command element's stage runs once per physics substep, after the physics ``clear`` stage and before the
actuators' stages. The shipped one is the rotor's ESC, :class:`~nexus_uav._src.vehicle.commands.esc.Esc`, one per
rotor: the controller's normalized command on the rotor's channel to the terminal voltage the rotor's motor takes.
A torque motor has no command element: its drive is part of the motor, which reads its entry of the controls
itself.
"""

from .esc import Esc, Escs

__all__ = ["Esc", "Escs"]
