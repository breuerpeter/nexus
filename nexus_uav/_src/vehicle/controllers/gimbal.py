"""The gimbal's controller: it holds a camera's attitude through the motors of three revolute joints.

The law is a cascade. The attitude error, the turn from the camera's attitude to the target as twice the vector
part of its quaternion, goes through a gain to a rate command in the camera's frame, which the law cuts to a
limit as a vector. The rate error goes onto each joint's axis, one dot product per joint, which is the Jacobian
transpose of a chain of revolute joints. A gain and an integral per motor then give each motor's current command,
from −1 to 1. The law reads each joint's axis and frames from the model, so it serves any chain of three revolute
joints, and it reads what a real gimbal's controller measures: the camera's rate, its attitude and the three joint
angles.

The camera's frame has x forward along the view, y right and z down, as a vehicle's body has. An attitude turns
the camera's frame into the North, East, Down frame, so the identity is a level camera that faces North. The
setpoint is a target attitude, a rate and a yaw lock bit. Roll and pitch always hold against the horizon. Yaw
holds against North with the lock, and against the vehicle's heading without it. The vehicle is the body the
first joint hangs from, and the law finds its heading from the camera's attitude and the joint angles.

:class:`GimbalLaw` runs the law over the gimbals of many vehicles at once, one kernel thread per vehicle. A
`Scope` of the vehicle declares one gimbal's controller with the ``NexusGimbalControllerAPI`` schema, and the
registry builds it as :class:`GimbalController`, a controller with a device stage, ``gimbal_law``, and a host
stage, ``gimbal_device``. The host stage is the gimbal's MAVLink end: the controller is a MAVLink gimbal device
on a port of its own, which a gimbal manager, an autopilot, sends its setpoints to.
"""

from __future__ import annotations

import errno
import math
import socket

import numpy as np
import warp as wp

from nexus_uav._src.core.errors import RunError
from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.schema import PoseTwist, joint_struct
from nexus_uav._src.core.signals import Signal
from nexus_uav._src.vehicle.controllers.gimbal_device import PORT, GimbalDevice

# What the plant writes for a revolute joint: one angle and one rate.
Revolute = joint_struct(1, 1)


@wp.func
def _heading(q: wp.quat) -> float:
    """The heading of the attitude ``q``: its turn about the vertical, the yaw of yaw, pitch, roll."""
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.func
def _turn(v: wp.vec3) -> wp.quat:
    """The turn about ``v`` by its length [rad]."""
    angle = wp.length(v)
    if angle > 0.0:
        return wp.quat_from_axis_angle(v / angle, angle)
    return wp.quat_identity()


@wp.func
def _hop(
    joint: int,
    angle: float,
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=wp.int32),
) -> wp.quat:
    """The turn from a joint's child body's frame to its parent body's, at the joint's ``angle``."""
    on_parent = wp.transform_get_rotation(joint_X_p[joint])
    on_child = wp.transform_get_rotation(joint_X_c[joint])
    return on_parent * wp.quat_from_axis_angle(joint_axis[joint_qd_start[joint]], angle) * wp.quat_inverse(on_child)


@wp.func
def _axis(
    joint: int,
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=wp.int32),
) -> wp.vec3:
    """A joint's axis in its child body's frame, which the joint's own angle doesn't change."""
    return wp.quat_rotate(wp.transform_get_rotation(joint_X_c[joint]), joint_axis[joint_qd_start[joint]])


@wp.kernel
def _law(
    rate: wp.array(dtype=wp.vec3),  # (vehicles,) the camera's angular rate in its own frame [rad/s]
    attitude: wp.array(dtype=wp.quat),  # (vehicles,) the camera's attitude
    angles: wp.array2d(dtype=float),  # (vehicles, 3) each joint's angle [rad]
    dt: float,  # the tick [s]
    joints: wp.array2d(dtype=wp.int32),  # (vehicles, 3) each gimbal's joints, from the vehicle to the camera
    joint_X_p: wp.array(dtype=wp.transform),  # the model's joint frames on each parent body
    joint_X_c: wp.array(dtype=wp.transform),  # the model's joint frames on each child body
    joint_axis: wp.array(dtype=wp.vec3),  # the model's joint axes, one per degree of freedom
    joint_qd_start: wp.array(dtype=wp.int32),  # each joint's first degree of freedom
    attitude_gain: float,
    rate_limit: float,
    rate_gain: wp.vec3,
    integral_gain: wp.vec3,
    setpoint_rate: wp.array(dtype=wp.vec3),  # (vehicles,) the rate that moves the target [rad/s]
    locked: wp.array(dtype=wp.int32),  # (vehicles,) the yaw lock bit
    target: wp.array(dtype=wp.quat),  # (vehicles,) state: the target, in the frame `frame` names
    frame: wp.array(dtype=wp.int32),  # (vehicles,) state: the lock bit the target's heading stands in
    integral: wp.array(dtype=wp.vec3),  # (vehicles,) state: each motor's integral
    controls: wp.array2d(dtype=float),  # (vehicles, 3) out: each motor's current command, −1 to 1
):
    """One step of the cascade for one vehicle: move the target by the setpoint's rate, then the attitude error
    to a rate command, the rate error onto each joint's axis, and a gain and an integral per motor.
    """
    v = wp.tid()
    q = attitude[v]
    # The camera's frame in each link's, down the chain, and so the vehicle's heading.
    h0 = _hop(joints[v, 0], angles[v, 0], joint_X_p, joint_X_c, joint_axis, joint_qd_start)
    h1 = _hop(joints[v, 1], angles[v, 1], joint_X_p, joint_X_c, joint_axis, joint_qd_start)
    h2 = _hop(joints[v, 2], angles[v, 2], joint_X_p, joint_X_c, joint_axis, joint_qd_start)
    heading = wp.vec3(0.0, 0.0, _heading(q * wp.quat_inverse(h0 * h1 * h2)))

    # A lock bit that changed since the target was last set: the target keeps its place and takes the new frame.
    t = target[v]
    if locked[v] != frame[v]:
        if locked[v] == 1:
            t = _turn(heading) * t
        else:
            t = _turn(-heading) * t
        frame[v] = locked[v]
    # The setpoint's rate moves the target: its z part about the vertical, its x and y parts about its own axes.
    r = setpoint_rate[v]
    t = wp.normalize(_turn(wp.vec3(0.0, 0.0, r[2] * dt)) * t * _turn(wp.vec3(r[0] * dt, r[1] * dt, 0.0)))
    target[v] = t
    world = t
    if locked[v] == 0:
        world = _turn(heading) * t

    # The attitude error, the short way round, to a rate command in the camera's frame, cut as a vector.
    e = wp.quat_inverse(q) * world
    sign = 2.0
    if e[3] < 0.0:
        sign = -2.0
    command = attitude_gain * sign * wp.vec3(e[0], e[1], e[2])
    # The setpoint's rate in the camera's frame: its z part from North, East, Down, its x and y parts from the
    # target's frame, which `e` turns into the camera's.
    command += wp.quat_rotate_inv(q, wp.vec3(0.0, 0.0, r[2])) + wp.quat_rotate(e, wp.vec3(r[0], r[1], 0.0))
    size = wp.length(command)
    if size > rate_limit:
        command *= rate_limit / size
    error = command - rate[v]

    # The rate error onto each joint's axis, in the camera's frame: the Jacobian transpose.
    a2 = _axis(joints[v, 2], joint_X_c, joint_axis, joint_qd_start)
    a1 = wp.quat_rotate_inv(h2, _axis(joints[v, 1], joint_X_c, joint_axis, joint_qd_start))
    a0 = wp.quat_rotate_inv(h2, wp.quat_rotate_inv(h1, _axis(joints[v, 0], joint_X_c, joint_axis, joint_qd_start)))
    on_joint = wp.vec3(wp.dot(a0, error), wp.dot(a1, error), wp.dot(a2, error))

    # A gain and an integral per motor, each within the command's range.
    held = integral[v]
    for k in range(3):
        held[k] = wp.clamp(held[k] + integral_gain[k] * on_joint[k] * dt, -1.0, 1.0)
        controls[v, k] = wp.clamp(rate_gain[k] * on_joint[k] + held[k], -1.0, 1.0)
    integral[v] = held


@wp.kernel
def _write(
    # (vehicles, 9) the written setpoint: the attitude, NaN to keep the target, the rate, the yaw lock bit, and
    # whether the attitude's heading is against North. One array, so a write costs one copy to the device.
    setpoint: wp.array2d(dtype=float),
    target: wp.array(dtype=wp.quat),
    frame: wp.array(dtype=wp.int32),
    setpoint_rate: wp.array(dtype=wp.vec3),
    locked: wp.array(dtype=wp.int32),
):
    """A written setpoint: the attitude replaces the target, in the frame the setpoint names, unless it's NaN.
    The law's step turns a target that stands in the other frame than the lock's into the lock's.
    """
    v = wp.tid()
    a = wp.quat(setpoint[v, 0], setpoint[v, 1], setpoint[v, 2], setpoint[v, 3])
    if not (wp.isnan(a[0]) or wp.isnan(a[1]) or wp.isnan(a[2]) or wp.isnan(a[3])):
        target[v] = wp.normalize(a)
        frame[v] = int(setpoint[v, 8])
    setpoint_rate[v] = wp.vec3(setpoint[v, 4], setpoint[v, 5], setpoint[v, 6])
    locked[v] = int(setpoint[v, 7])


class GimbalLaw:
    """The law over the gimbals of every vehicle of the model at once.

    Until something writes a setpoint, each gimbal holds its camera level, follows its vehicle's heading and has no
    rate.

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
        device = model.device
        self._model = model
        found = joints.numpy() if isinstance(joints, wp.array) else joints
        self._joints = wp.array(np.asarray(found, dtype=np.int32), dtype=wp.int32, device=device)
        self.vehicles = self._joints.shape[0]
        self._attitude_gain, self._rate_limit = float(attitude_gain), float(rate_limit)
        self._rate_gain = wp.vec3(*(float(x) for x in rate_gain))
        self._integral_gain = wp.vec3(*(float(x) for x in integral_gain))
        level = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.vehicles, 1))
        self._target = wp.array(level, dtype=wp.quat, device=device)
        self._frame = wp.zeros(self.vehicles, dtype=wp.int32, device=device)
        self._locked = wp.zeros(self.vehicles, dtype=wp.int32, device=device)
        self._rate = wp.zeros(self.vehicles, dtype=wp.vec3, device=device)
        self._integral = wp.zeros(self.vehicles, dtype=wp.vec3, device=device)

    def write(self, attitude, rate, locked, earth=None) -> None:
        """Write each vehicle's setpoint, once: an event, not a level the law reads again.

        Args:
            attitude: ``(vehicles, 4)`` the attitude that replaces the target, a quaternion ``(x, y, z, w)``. NaN
                leaves the target where the rate has taken it.
            rate: ``(vehicles, 3)`` the rate that moves the target each tick [rad/s]: its z part turns the target
                about the vertical, and its x and y parts about the target's own axes.
            locked: ``(vehicles,)`` the yaw lock bit: whether yaw holds against North, not the vehicle's heading.
            earth: ``(vehicles,)`` whether each attitude's heading is against North, not the vehicle's heading.
                ``None`` takes the frame the lock bit names. An attitude in the other frame than the lock's
                points the camera where it says at the next step, and the target then holds as the lock says.
        """
        device = self._target.device
        setpoint = np.empty((self.vehicles, 9), dtype=np.float32)
        setpoint[:, :4], setpoint[:, 4:7] = attitude, rate
        setpoint[:, 7] = locked
        setpoint[:, 8] = locked if earth is None else earth
        wp.launch(
            _write,
            dim=self.vehicles,
            inputs=(wp.array(setpoint, dtype=float, device=device),),
            outputs=(self._target, self._frame, self._rate, self._locked),
            device=device,
        )

    def step(self, rate, attitude, angles, dt, controls) -> None:
        """Run the law once and write each vehicle's three commands, from −1 to 1, into ``controls``.

        Args:
            rate: ``(vehicles,)`` the camera's angular rate in its own frame, a ``wp.vec3`` each [rad/s].
            attitude: ``(vehicles,)`` the camera's attitude, a ``wp.quat`` each.
            angles: ``(vehicles, 3)`` each joint's angle, in the order of ``joints`` [rad].
            dt: The tick [s].
            controls: ``(vehicles, 3)`` out: each joint motor's current command.
        """
        model = self._model
        wp.launch(
            _law,
            dim=self.vehicles,
            inputs=(
                rate,
                attitude,
                angles,
                float(dt),
                self._joints,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_qd_start,
                self._attitude_gain,
                self._rate_limit,
                self._rate_gain,
                self._integral_gain,
                self._rate,
                self._locked,
                self._target,
                self._frame,
                self._integral,
            ),
            outputs=(controls,),
            device=self._target.device,
        )


@wp.kernel
def _measure(
    camera: wp.array(dtype=PoseTwist),  # the camera body's pose and twist in the world's axes
    joint_0: wp.array(dtype=Revolute),
    joint_1: wp.array(dtype=Revolute),
    joint_2: wp.array(dtype=Revolute),
    rate: wp.array(dtype=wp.vec3),  # out: the camera's rate in its own frame
    attitude: wp.array(dtype=wp.quat),  # out: the camera's attitude against North, East, Down
    angles: wp.array2d(dtype=float),  # out: the three joint angles
    measured: wp.array(dtype=float),  # (7,) out: the attitude, then the rate, which the device's status reads
):
    """The plant's outputs as what a gimbal's controller measures. The world's x axis points North and its z
    axis up, so a half turn about x takes it to North, East, Down.
    """
    body = camera[0]
    w = wp.quat_rotate_inv(body.orientation, body.angular_velocity)
    q = wp.quat(1.0, 0.0, 0.0, 0.0) * body.orientation
    rate[0] = w
    attitude[0] = q
    for i in range(4):
        measured[i] = q[i]
    for i in range(3):
        measured[4 + i] = w[i]
    angles[0, 0] = joint_0[0].q[0]
    angles[0, 1] = joint_1[0].q[0]
    angles[0, 2] = joint_2[0].q[0]


_GAINS = {
    "attitude_gain": "nexus:attitudeGain",
    "rate_gain": "nexus:rateGain",
    "integral_gain": "nexus:integralGain",
}


def axis_limits(model, joints) -> dict:
    """The lowest and the highest angle the gimbal reaches about each camera axis [rad], from its joints' limits.

    Args:
        model: The finalized articulated model.
        joints: The gimbal's three revolute joints in the model, from the vehicle to the camera.

    Returns:
        ``roll``, ``pitch`` and ``yaw``, each a lowest and a highest angle, NaN for an axis whose joint turns a
        full turn or more, or that no joint turns about. A joint counts for the camera axis its own axis lies
        nearest to at zero joint angles, and its limits swap and change sign when it points the other way.
    """
    starts = model.joint_qd_start.numpy()
    axes, lower, upper = model.joint_axis.numpy(), model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    on_parent, on_child = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    limits = dict.fromkeys(("roll", "pitch", "yaw"), (math.nan, math.nan))
    to_camera = wp.quat_identity()  # from the frame of the joint's child body to the camera's, at zero angles
    for joint in reversed([int(j) for j in joints]):
        dof = int(starts[joint])
        child = wp.quat(*on_child[joint][3:])
        axis = np.array(wp.quat_rotate(to_camera, wp.quat_rotate(child, wp.vec3(*axes[dof]))))
        to_camera = to_camera * child * wp.quat_inverse(wp.quat(*on_parent[joint][3:]))
        about = int(np.argmax(np.abs(axis)))
        low, high = float(lower[dof]), float(upper[dof])
        if high - low >= 2.0 * math.pi:
            continue
        limits[("roll", "pitch", "yaw")[about]] = (low, high) if axis[about] > 0.0 else (-high, -low)
    return limits


def _chain(model, path: str, inputs: dict) -> list[int]:
    """The three joints the connections of the controller at `path` name, as their indices in the model, from the
    vehicle to the camera.

    Raises:
        ValueError: The connections don't name one camera and three joints, a joint is no revolute joint of the
            model, a joint doesn't hang from the one before it, or the camera's body isn't the one behind the last
            joint; the message names the controller's prim, the input and the prim at fault.
    """
    cameras, named = list(inputs.get("camera", ())), list(inputs.get("joints", ()))
    if len(cameras) != 1:
        raise ValueError(
            f"{path}: nexus:inputs:camera names {len(cameras)} prims, and a gimbal's controller reads one camera's body"
        )
    if len(named) != 3:
        raise ValueError(
            f"{path}: nexus:inputs:joints names {len(named)} joints, and a gimbal has three, in order from the "
            "vehicle to the camera"
        )
    labels = [str(label) for label in model.joint_label]
    dims, parents, children = model.joint_dof_dim.numpy(), model.joint_parent.numpy(), model.joint_child.numpy()
    joints: list[int] = []
    for joint in named:
        if joint not in labels:
            raise ValueError(f"{path}: nexus:inputs:joints names {joint}, which is no joint of the model")
        j = labels.index(joint)
        linear, angular = (int(n) for n in dims[j])
        if (linear, angular) != (0, 1):
            raise ValueError(
                f"{path}: nexus:inputs:joints names {joint}, which has {linear} linear and {angular} angular "
                "degrees of freedom, and a gimbal's joint is revolute"
            )
        if joints and int(parents[j]) != int(children[joints[-1]]):
            raise ValueError(
                f"{path}: nexus:inputs:joints names {joint}, which does not hang from the body behind "
                f"{labels[joints[-1]]}: the three joints form one chain"
            )
        joints.append(j)
    bodies = [str(label) for label in model.body_label]
    # The camera's target is its body today. A sensor on the camera later is a wire only, with no place in the chain.
    if cameras[0] in bodies and bodies.index(cameras[0]) != int(children[joints[-1]]):
        raise ValueError(
            f"{path}: nexus:inputs:camera names {cameras[0]}, which is not the body behind {named[-1]}, the last "
            "joint of nexus:inputs:joints"
        )
    return joints


# How often the device sends its heartbeat and its status, in nanoseconds of sim time: 1 Hz and 20 Hz.
_HEARTBEAT_PERIOD = 1_000_000_000
_STATUS_PERIOD = 50_000_000


class GimbalController(Component):
    """The controller of one camera gimbal, which a `Scope` of the vehicle declares with
    ``NexusGimbalControllerAPI``: the camera's pose and the three joints' angles to one current command per joint
    motor, entry ``i`` of its ``controls`` for joint ``i``.

    It's a MAVLink gimbal device too, :class:`~nexus_uav._src.vehicle.controllers.gimbal_device.GimbalDevice`, on
    port 13280 of the User Datagram Protocol (UDP) plus the run's instance. Its host stage reads the port without a
    wait, writes the law each setpoint that arrives, and answers the address its last datagram came from, its
    gimbal manager's: a heartbeat at 1 Hz and its status at 20 Hz of sim time, and its information on request. It
    sends nothing before a message arrives. Declared after the controller that flies the vehicle, its host stage
    sits beside that one's, and the tick gains no graph segment.

    Args:
        model: The finalized articulated model, in which the controller finds its joints.
        path: The controller's own prim, which its errors name.
        instance: The run's instance, which the device's port counts from.
        inputs: The prims its connections target, by input: ``camera``, the camera's body, and ``joints``, the
            three joints from the vehicle to the camera.
        attitude_gain: The rate command per attitude error [1/s], ``nexus:attitudeGain``.
        rate_limit: The size the law cuts the rate command to [rad/s], ``nexus:rateLimit``.
        rate_gain: ``(3,)`` each motor's command per rate error [s/rad], ``nexus:rateGain``.
        integral_gain: ``(3,)`` each motor's command per rate error per second [1/rad], ``nexus:integralGain``.

    Raises:
        ValueError: A gain is below 0, NaN or infinite, the rate limit isn't positive and finite, or the declaration isn't a camera's body
            behind a chain of three revolute joints; the message names the prim, and the attribute or the input.
        RunError: At ``connect``, another process holds the device's port; the message names the prim and the port.
    """

    needs_model = True
    needs_instance = True  # the device's port counts from the run's instance

    def __init__(
        self,
        *,
        model,
        path: str,
        inputs: dict,
        instance: int = 0,
        attitude_gain: float = 0.0,
        rate_limit: float = 0.0,
        rate_gain=(0.0, 0.0, 0.0),
        integral_gain=(0.0, 0.0, 0.0),
    ):
        gains = {"attitude_gain": (attitude_gain,), "rate_gain": rate_gain, "integral_gain": integral_gain}
        for keyword, values in gains.items():
            if not all(math.isfinite(float(x)) and float(x) >= 0.0 for x in values):
                shown = ", ".join(f"{float(x):g}" for x in values)
                raise ValueError(f"{path}: {_GAINS[keyword]} is {shown}, and a gain is finite and 0 or above")
        if not (math.isfinite(float(rate_limit)) and float(rate_limit) > 0.0):
            raise ValueError(
                f"{path}: nexus:rateLimit is {float(rate_limit):g}, and the limit of the rate command is finite and "
                "above 0"
            )
        device = model.device
        joints = _chain(model, path, inputs)
        self._law = GimbalLaw(
            model,
            [joints],
            attitude_gain=attitude_gain,
            rate_limit=rate_limit,
            rate_gain=rate_gain,
            integral_gain=integral_gain,
        )
        self.camera = Signal("camera", PoseTwist, shape=(1,))
        self.joints = Signal("joints", list[Revolute], shape=(1,))
        self.controls = Signal("controls", wp.float32, shape=(1, 3))
        # What the law reads each tick, written from the plant's outputs.
        self._rate = wp.zeros(1, dtype=wp.vec3, device=device)
        self._attitude = wp.zeros(1, dtype=wp.quat, device=device)
        self._angles = wp.zeros((1, 3), dtype=float, device=device)
        # The measured attitude and rate in one array, so the device's status costs one copy to the host.
        self._measured = wp.zeros(7, dtype=float, device=device)
        # The MAVLink end: the device, its socket, the address of its manager, and when it next sends.
        self._path = path
        self._port = PORT + int(instance)
        self._device = GimbalDevice(**axis_limits(model, joints))
        self._socket: socket.socket | None = None
        self._manager = None
        self._heartbeat_due = self._status_due = 0

    def connect(self, links) -> None:
        """Bind the device's port, for the run.

        Raises:
            RunError: Another process holds the port.
        """
        link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            link.bind(("0.0.0.0", self._port))
        except OSError as exc:
            link.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            raise RunError(
                f"{self._path}: UDP {self._port}, the gimbal device's port, is in use: another run or process on "
                "this machine holds it"
            ) from exc
        link.setblocking(False)
        self._socket = link

    def close(self) -> None:
        """Close the device's port. Idempotent."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def write(self, attitude, rate, locked, earth=None) -> None:
        """Write the setpoint, once, as :meth:`GimbalLaw.write` takes it for one vehicle: the attitude
        ``(x, y, z, w)``, NaN to keep the target, the rate [rad/s], the yaw lock bit, and the attitude's frame.
        """
        self._law.write(attitude, rate, locked, earth)

    def _step(self, tick) -> None:
        """The stage's work: the plant's outputs to the law's inputs, then the law, into the controls' buffer."""
        joints = self.joints.buffer
        wp.launch(
            _measure,
            dim=1,
            inputs=(self.camera.buffer, joints[0], joints[1], joints[2]),
            outputs=(self._rate, self._attitude, self._angles, self._measured),
        )
        self._law.step(self._rate, self._attitude, self._angles, tick.period, self.controls.buffer)

    def _link(self, tick) -> None:
        """The device's stage: take every datagram that waits, with no wait of its own, write the law each
        setpoint, answer the sender, and send the heartbeat and the status when each is due.
        """
        link = self._socket
        if link is None:
            return
        now = tick.t.time_ns
        while True:
            try:
                datagram, sender = link.recvfrom(65535)
            except BlockingIOError:
                break
            self._manager = sender
            setpoint, replies = self._device.handle(datagram, now // 1_000_000)
            if setpoint is not None:
                self._law.write([setpoint.attitude], [setpoint.rate], [setpoint.locked], [setpoint.earth])
            for reply in replies:
                link.sendto(reply, sender)
        if self._device.system is None:
            return
        if now >= self._heartbeat_due:
            link.sendto(self._device.heartbeat(), self._manager)
            self._heartbeat_due = _next(self._heartbeat_due, _HEARTBEAT_PERIOD, now)
        if now >= self._status_due:
            # What the law measured at its last step, one tick back.
            measured = self._measured.numpy()
            link.sendto(self._device.status(now // 1_000_000, measured[:4], measured[4:]), self._manager)
            self._status_due = _next(self._status_due, _STATUS_PERIOD, now)

    def stages(self) -> list[Stage]:
        """A host stage, ``gimbal_device``, the MAVLink end, then a device stage, ``gimbal_law``, reading the
        camera's pose and the joints, and writing the controls.
        """
        return [
            Stage("gimbal_device", "host", self._link),
            Stage("gimbal_law", "device", self._step, reads=(self.camera, self.joints), writes=(self.controls,)),
        ]


def _next(due: int, period: int, now: int) -> int:
    """When a message that was due at `due` and went out at `now` is next due [ns]: the first goes out at once,
    and each later one a whole number of periods after it, as a sensor's samples do.
    """
    first = due if due else now
    return first + ((now - first) // period + 1) * period


__all__ = ["GimbalController", "GimbalLaw", "axis_limits"]
