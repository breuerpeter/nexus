"""The command elements: each turns the controller's command into Newton's control inputs.

A command element's stage runs once per physics substep, after the physics ``clear`` stage and before the
actuators, and turns the controls the controller writes into the signal an actuator takes. The shipped one is the
rotor's ESC, :class:`~nexus_uav._src.vehicle.commands.esc.Esc`, one per rotor: the controller's normalized command
on the rotor's channel to the terminal voltage its motor, an actuator, turns into torque.
"""

from .esc import Esc, Escs

__all__ = ["Esc", "Escs"]
