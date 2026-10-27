"""The base of every RTX sensor: a host sensor over a prim the vehicle file authored, rendered by the run's render
stage.

The prims are body children the vehicle's Universal Scene Description (USD) file authors, and each
declares its sensor with an applied schema. The render stage, :mod:`.stage`, renders in the loop's process: no
vehicle names it, the sensor's class requires it by the name ``ovrtx``, and the build makes it once. The loop hands
the sensor the render stage at ``connect``, where the sensor reads its prim from the vehicle's stage, its mount and
a camera its optics, and asks the render stage for a render product of its own. Each sensor decimates to its
declared rate on sim time. At a due tick the render stage renders the tick's poses, and the sensor writes the frame
to its output signal, stamped with the tick's sim time, and logs it here on the host.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_uav._src.core import logger
from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.signals import Signal

from .. import rate as due_rule

PEER = "ovrtx"  # the name an RTX sensor's class requires the render stage by


class RtxMountedSensor(Component):
    """Shared base for RTX sensors mounted on a vehicle body: a ``Sensor`` whose work is one host
    stage, over a prim authored in the vehicle USD as a body child.

    Its frames come from the render stage, so its stage runs once per tick as a host stage, outside any captured
    graph, self-decimating to ``rate``. The mount is rigid: the render stage poses the body, and the prim rides it
    through the vehicle's own hierarchy.

    Args:
        model: Newton's model, whose body labels are the paths of the vehicle's rigid bodies: the sensor
            rides the body its prim sits under.
        path: The path of the sensor's prim on the vehicle's stage.
        seed: The sensor's noise seed, which an RTX sensor doesn't draw on.
        rate: The sensor's physical rate, hertz.
        out: The host signal each frame goes to, which its stage declares it writes.

    Raises:
        ValueError: The prim's parent is no rigid body of the model, or the rate is negative; the message names
            the prim.
    """

    KIND = "rtx"
    output = ""  # what the render stage hands it: "color", "radiance_depth" or "points"
    requires = (PEER,)  # what the build makes, once, for a vehicle that declares this sensor: the render stage

    def __init__(self, model, path: str, *, seed: int = 0, rate: float, out: Signal):
        from pxr import Sdf

        from .stage import render_path

        bodies = [str(label) for label in model.body_label]
        parent = str(Sdf.Path(path).GetParentPath())
        if parent not in bodies:
            raise ValueError(
                f"{path}: the prim's parent {parent} isn't a rigid body of the vehicle; put the sensor's prim "
                "under the body it rides"
            )
        self.body = bodies.index(parent)
        self.model = model  # the model the render stage binds its bodies from
        self.prim_path = str(path)  # on the vehicle's stage, where the sensor reads its prim at connect
        # On the render stage, where the renderer draws the prim: the vehicle's root prim, its default prim
        # by convention, sits at the render stage's vehicle root.
        self.path = render_path(path, str(Sdf.Path(path).GetPrefixes()[0]))
        self.name = self.path.rsplit("/", 1)[-1].lower()  # for example 'fpvcam' -> sim/vehicle/sensors/fpvcam
        self.rate = due_rule.checked_rate(rate, path)
        self.out = out
        self._link = None
        self._logger = None
        self._local = None  # the authored mount, a row-vector Gf.Matrix4d, read at connect
        self.mount: np.ndarray | None = None
        self._start: int | None = None  # the sim time of the first frame, nanoseconds
        self._taken = 0  # how many frames the sensor has taken
        self._now = 0.0  # the tick in progress: an emit logs at the time its frame shows, then restores it
        logger.info(f"{type(self).__name__}: {self.path} @{self.rate:.0f}Hz -> sensors/{self.name} (body[{self.body}])")

    def connect(self, links) -> None:
        """Take the render stage, read the sensor's prim from the vehicle's stage, the authored mount, a rigid
        transform in the body's axes, and ask the render stage for the sensor's render product. In debug mode a non-camera sensor logs its coordinate frame
        once, statically, at its own row ``frame``, with the fixed mount and the body's frame as its parent:
        the body's entity gets a ``Transform3D`` every tick, so the frame rides the body pose and nothing
        re-logs per sample. The frame has a row of its own because a lidar's points sit at the sensor's
        entity in world coordinates. Cameras carry their own Pinhole frustum instead.

        Raises:
            ValueError: The vehicle's stage holds no prim at the sensor's path; the message names it.
        """
        from pxr import UsdGeom

        self._link = links[PEER]
        prim = self._link.vehicle_stage.GetPrimAtPath(self.prim_path)
        if not prim:
            raise ValueError(f"{self.prim_path}: the vehicle's stage holds no prim at this path")
        self._local = UsdGeom.Xformable(prim).GetLocalTransformation()
        self.mount = np.array(self._local, dtype=np.float64)
        self._read_prim(prim)
        if self._logger is not None and getattr(self._logger, "debug", False) and self.KIND != "cameras":
            try:
                loc = self._local  # row-vector Gf: the fixed mount transform relative to the body
                self._logger.log_static_frame(
                    "frame",
                    [loc[3][0], loc[3][1], loc[3][2]],
                    [[loc[i][j] for j in range(3)] for i in range(3)],
                    label=self.name,
                )
            except Exception as exc:
                logger.warning(f"{type(self).__name__} {self.name}: static frame logging unavailable: {exc!r}")
        self._link.add(self)

    def _read_prim(self, prim) -> None:
        """Read what the sensor's prim authors beside its mount; a camera reads its optics. Nothing by default."""

    def set_logger(self, logger_) -> None:
        """Component-owned logging: the loop hands over the Logger scoped to this sensor, and None = off."""
        self._logger = logger_

    def due(self, now_ns: int) -> bool:
        """Whether a frame is due at sim time ``now_ns``, in nanoseconds; a due frame counts toward the next
        one's due time. Every sensor's rule, :func:`~nexus_uav._src.vehicle.sensors.rate.due`, decides.
        """
        elapsed = 0 if self._start is None else now_ns - self._start
        if not due_rule.due(wp.int64(elapsed), wp.int32(self._taken), wp.float64(self.rate)):
            return False
        if self._start is None:
            self._start = now_ns
        self._taken += 1
        return True

    def stages(self) -> list[Stage]:
        """One host stage, named after the sensor, over :meth:`sample`, which writes the output signal."""
        return [Stage(self.name, "host", lambda tick: self.sample(tick.state, tick.t), writes=(self.out,))]

    def sample(self, state, t) -> None:
        """Host-stage sample: the render stage renders the tick's poses for the due sensors and hands each its
        frame.

        Decimation is on sim time, since ``rate`` is the sensor's physical rate: a 30 Hz camera
        samples 30x per simulated second whatever the Real Time Factor (RTF).
        """
        self._now = float(t.sim_time)
        self._link.tick(t, state)

    def emit(self, arrays: dict, t_shown: float) -> None:
        """Write one frame to the output signal and log it; ``t_shown`` is the sim time the frame shows."""
        raise NotImplementedError
