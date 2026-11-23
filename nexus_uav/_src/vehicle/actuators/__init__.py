"""The actuators: each turns a signal into torque on a joint, into the model's ``newton.Control``.

An actuator's stage runs once per physics substep, after the command elements and before the force elements, the
direction the hardware runs, and adds its torque into ``control.joint_f``, which the physics ``clear`` stage zeroes
at the substep's start. Two ship. The rotor's motor, :class:`~nexus_uav._src.vehicle.actuators.motor.Motor`, one
per rotor, turns its ESC's terminal voltage into the torque on the rotor's joint through Newton's actuator, built
in code from the sheet's constants. The torque motor,
:class:`~nexus_uav._src.vehicle.actuators.torque_motor.TorqueMotor`, one per joint that a controller of its own
moves, a gimbal's first, turns one entry of the controls, a current command, into the torque on its joint in one
kernel of its own. The build fails a ``NewtonActuator`` prim, so a nexus actuator is the one way to drive a joint.
"""

from .motor import Motor, Motors
from .torque_motor import TorqueMotor, TorqueMotors

__all__ = ["Motor", "Motors", "TorqueMotor", "TorqueMotors"]
