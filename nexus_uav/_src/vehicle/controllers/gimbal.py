"""The gimbal's control law: the camera's attitude error to one current command per joint motor.

:class:`GimbalLaw` runs the law over the gimbals of many vehicles at once, one row per vehicle. The camera's frame
has x forward along the view, y right and z down, and an attitude turns it into the North, East, Down frame, so
the identity is a level camera that faces North. Roll, pitch and yaw are the angles about x, y and z.
"""

from __future__ import annotations

from nexus_uav._src.core.interfaces import Component


class GimbalLaw:
    """The law over the gimbals of every vehicle of the model at once.

    Args:
        model: The finalized articulated model, from which the law reads each joint's axis and frames.
        joints: ``(vehicles, 3)`` each gimbal's three revolute joints in the model, from the vehicle to the camera.
            Entry ``i`` of a vehicle's controls is the command of its joint ``i``.
        attitude_gain: The rate command per attitude error [1/s].
        rate_limit: The size the law cuts the rate command to, as a vector [rad/s].
        rate_gain: ``(3,)`` each motor's command per rate error [s/rad].
        integral_gain: ``(3,)`` each motor's command per rate error per second [1/rad].
    """

    def __init__(self, model, joints, *, attitude_gain, rate_limit, rate_gain, integral_gain):
        pass

    def write(self, attitude, rate, locked) -> None:
        """Write each vehicle's setpoint, once: an event, not a level the law reads again.

        Args:
            attitude: ``(vehicles, 4)`` the attitude that replaces the target, a quaternion ``(x, y, z, w)``. Its
                heading is against North with the yaw lock, and against the vehicle's heading without it. NaN
                leaves the target where the rate has taken it.
            rate: ``(vehicles, 3)`` the rate that moves the target each tick [rad/s]: its z part turns the target
                about the vertical, and its x and y parts about the camera's own axes.
            locked: ``(vehicles,)`` the yaw lock bit: whether yaw holds against North, not the vehicle's heading.
        """

    def step(self, rate, attitude, angles, dt, controls) -> None:
        """Run the law once and write each vehicle's three commands, from −1 to 1, into ``controls``.

        Args:
            rate: ``(vehicles,)`` the camera's angular rate in its own frame, a ``wp.vec3`` each [rad/s].
            attitude: ``(vehicles,)`` the camera's attitude, a ``wp.quat`` each.
            angles: ``(vehicles, 3)`` each joint's angle, in the order of ``joints`` [rad].
            dt: The tick [s].
            controls: ``(vehicles, 3)`` out: each joint motor's current command.
        """


class GimbalController(Component):
    """The controller of one camera gimbal, which a `Scope` of the vehicle declares with
    ``NexusGimbalControllerAPI``.

    Args:
        model: The finalized articulated model, in which the controller finds its joints.
        path: The controller's own prim, which its errors name.
        inputs: The prims its connections target, by input: ``camera``, the camera's body, and ``joints``, the
            three joints from the vehicle to the camera.
        attitude_gain: The rate command per attitude error [1/s], ``nexus:attitudeGain``.
        rate_limit: The size the law cuts the rate command to [rad/s], ``nexus:rateLimit``.
        rate_gain: ``(3,)`` each motor's command per rate error [s/rad], ``nexus:rateGain``.
        integral_gain: ``(3,)`` each motor's command per rate error per second [1/rad], ``nexus:integralGain``.

    Raises:
        ValueError: A gain is below 0, the rate limit isn't positive, or the declaration isn't a camera's body
            behind a chain of three revolute joints; the message names the prim, and the attribute or the input.
    """

    def __init__(
        self,
        *,
        model,
        path: str,
        inputs: dict,
        attitude_gain: float = 0.0,
        rate_limit: float = 0.0,
        rate_gain=(0.0, 0.0, 0.0),
        integral_gain=(0.0, 0.0, 0.0),
    ):
        pass

    def write(self, attitude, rate, locked) -> None:
        """Write the setpoint, once, as :meth:`GimbalLaw.write` takes it for one vehicle: the attitude
        ``(x, y, z, w)``, NaN to keep the target, the rate [rad/s], and the yaw lock bit.
        """


__all__ = ["GimbalController", "GimbalLaw"]
