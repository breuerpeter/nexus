"""The command elements: each turns the controller's command into Newton's control inputs.

A command element's stage runs once per physics substep, after the physics ``clear`` stage and before the force
stages, and writes the targets and torque the model's ``newton.Control`` holds; physics then steps the Newton
actuators the model still holds, a servo's, before its solver. The shipped ones are the rotor's ESC,
:class:`~nexus_uav._src.vehicle.commands.esc.Esc`, the controller's normalized command on the rotor's channel to
the terminal voltage, and the rotor's motor, :class:`~nexus_uav._src.vehicle.commands.motor.Motor`, that voltage
to the torque on the rotor's joint through Newton's actuator, one of each per rotor. A component that writes
another actuator's target, a gimbal servo's, is one more class here.
"""

from .esc import Esc, Escs
from .motor import Motor, Motors

__all__ = ["Esc", "Escs", "Motor", "Motors"]
