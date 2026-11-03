"""The actuators: each turns a signal into torque on a joint, into the model's ``newton.Control``.

An actuator's stage runs once per physics substep, after the command elements and before the force elements, the
direction the hardware runs, and adds its torque into ``control.joint_f``, which the physics ``clear`` stage zeroes
at the substep's start. The shipped one is the rotor's motor, :class:`~nexus_uav._src.vehicle.actuators.motor.Motor`,
one per rotor: its ESC's terminal voltage to the torque on the rotor's joint through Newton's actuator, built in
code from the sheet's constants. A servo on another joint is one more class here, and the build fails a
``NewtonActuator`` prim, so a nexus actuator is the one way to drive a joint.
"""

from .motor import Motor, Motors

__all__ = ["Motor", "Motors"]
