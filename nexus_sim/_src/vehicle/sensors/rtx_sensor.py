"""The base of every RTX sensor: a host sensor over a prim the vehicle file authored, rendered by the Kit peer.

The prims are body children the vehicle's Universal Scene Description (USD) file authors, and each
declares its sensor with an applied schema. The Kit peer is a required peer: no vehicle names it, the
sensor's class requires it, and the build starts it once. The loop hands the sensor the peer's link at
``connect``, and the sensor reads its prim from the link's vehicle stage there: its mount, and a camera
its optics. Each sensor decimates to its declared rate on sim time. At a due tick the render link sends
the poses and the loop flies on; the frame comes back on a later tick, stamped with the sim time it
shows, and the sensor writes it to its output signal and logs it here on the host.
"""

from __future__ import annotations

import numpy as np

from nexus_sim._src.core import logger
from nexus_sim._src.core.interfaces import Component, Stage
from nexus_sim._src.core.signals import Signal

PEER = "kit"  # the name an RTX sensor's class requires the Kit peer by


class RtxMountedSensor(Component):
    """Shared base for RTX sensors mounted on a vehicle body: a ``Sensor`` whose work is one host
    stage, over a prim authored in the vehicle USD as a body child.

    Its frames come from the Kit peer over a socket, so its stage runs once per tick as a host
    stage, outside any captured graph, self-decimating to ``rate``. The mount is rigid: the link poses the
    sensor prim with W = L_authored * W_body, in row-vector form, from the same state the body poses come
    from, since a Fabric world matrix has no hierarchy.

    Args:
        model: Newton's model, whose body labels are the paths of the vehicle's rigid bodies: the sensor
            rides the body its prim sits under.
        path: The path of the sensor's prim on the vehicle's stage.
        seed: The sensor's noise seed, which an RTX sensor doesn't draw on.
        rate: The sensor's physical rate, hertz.
        out: The host signal each frame goes to, which its stage declares it writes.

    Raises:
        ValueError: The prim's parent is no rigid body of the model; the message names the prim.
    """

    KIND = "rtx"
    output = ""  # what the peer returns for it: "color", "radiance_depth" or "points"
    requires = (PEER,)  # the peers the build starts, once, for a vehicle that declares this sensor

    def __init__(self, model, path: str, *, seed: int = 0, rate: float, out: Signal):
        from pxr import Sdf

        from .rtx_stage import render_path

        bodies = [str(label) for label in model.body_label]
        parent = str(Sdf.Path(path).GetParentPath())
        if parent not in bodies:
            raise ValueError(
                f"{path}: the prim's parent {parent} isn't a rigid body of the vehicle; put the sensor's prim "
                "under the body it rides"
            )
        self.body = bodies.index(parent)
        self.prim_path = str(path)  # on the vehicle's stage, where the sensor reads its prim at connect
        # On the render stage, where the peer renders the prim: the vehicle's root prim, its default prim
        # by convention, sits at the render stage's vehicle root.
        self.path = render_path(path, str(Sdf.Path(path).GetPrefixes()[0]))
        self.name = self.path.rsplit("/", 1)[-1].lower()  # for example 'fpvcam' -> sim/vehicle/sensors/fpvcam
        self.rate = float(rate)
        self.out = out
        self._link = None
        self._logger = None
        self._local = None  # the authored mount, a row-vector Gf.Matrix4d, read at connect
        self.mount: np.ndarray | None = None
        self._next_due: float | None = None
        self._now = 0.0  # the tick in progress: an emit logs at the time its frame shows, then restores it
        logger.info(f"{type(self).__name__}: {self.path} @{self.rate:.0f}Hz -> sensors/{self.name} (body[{self.body}])")

    def connect(self, links) -> None:
        """Take the Kit link and read the sensor's prim from the link's vehicle stage: the authored mount,
        a rigid transform in the body's axes. In debug mode a non-camera sensor logs its coordinate frame
        once, statically, at its own row ``frame``, with the fixed mount and the body's frame as its parent:
        the body's entity gets a ``Transform3D`` every tick, so the frame rides the body pose and nothing
        re-logs per sample. The frame has a row of its own because a lidar's points sit at the sensor's
        entity in world coordinates. Cameras carry their own Pinhole frustum instead.

        Raises:
            ValueError: The link's vehicle stage holds no prim at the sensor's path; the message names it.
        """
        from pxr import UsdGeom

        self._link = links[PEER]
        prim = self._link.stage.GetPrimAtPath(self.prim_path)
        if not prim:
            raise ValueError(f"{self.prim_path}: the Kit link's vehicle stage holds no prim at this path")
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

    def _read_prim(self, prim) -> None:
        """Read what the sensor's prim authors beside its mount; a camera reads its optics. Nothing by default."""

    def set_logger(self, logger_) -> None:
        """Component-owned logging: the loop hands over the Logger scoped to this sensor, and None = off."""
        self._logger = logger_

    def take_due(self, now: float) -> bool:
        """Whether a frame is due at sim time ``now``; a due frame advances the due time.

        Absolute due-time accumulation, not last+period quantized to the tick grid: a 24 Hz camera
        on 4 ms ticks would land every 44 ms = 22.7 fps; accumulation alternates 40/44 ms = 24.0.
        """
        if self._next_due is not None and now < self._next_due:
            return False
        period = 1.0 / self.rate
        base = self._next_due if self._next_due is not None else now
        self._next_due = max(base + period, now + 0.5 * period)  # never burst after a stall
        return True

    def stages(self) -> list[Stage]:
        """One host stage, named after the sensor, over :meth:`sample`, which writes the output signal."""
        return [Stage(self.name, "host", lambda tick: self.sample(tick.state, tick.t), writes=(self.out,))]

    def sample(self, state, t) -> None:
        """Host-stage sample: the link sends the due sensors' frame and hands back the one before.

        Decimation is on sim time, since ``rate`` is the sensor's physical rate: a 30 Hz camera
        samples 30x per simulated second whatever the Real Time Factor (RTF).
        """
        self._now = float(t.sim_time)
        self._link.tick(t, state)

    def emit(self, arrays: dict, t_shown: float) -> None:
        """Write one frame to the output signal and log it; ``t_shown`` is the sim time the frame shows."""
        raise NotImplementedError
