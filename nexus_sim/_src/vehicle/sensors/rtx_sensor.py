"""The base of every RTX sensor: a host sensor over a prim the vehicle file authored, rendered by the Kit peer.

The prims are body children the vehicle's Universal Scene Description (USD) file authors, and each
declares its sensor with an applied schema. The Kit peer is a required peer: no vehicle names it, the
sensor's class requires it, and the build starts it once. Each sensor decimates to its
declared rate on sim time. At a due tick the render link sends the poses and the loop flies
on; the frame comes back on a later tick, stamped with the sim time it shows, and the sensor logs
it here on the host.
"""

from __future__ import annotations

import numpy as np

from nexus_sim._src.core import logger
from nexus_sim._src.core.interfaces import Stage


class RtxMountedSensor:
    """Shared base for RTX sensors mounted on a vehicle body: a ``Sensor`` whose work is one host
    stage, over a prim authored in the vehicle USD as a body child.

    Its frames come from the Kit peer over a socket, so its stage runs once per tick as a host
    stage, outside any captured graph, self-decimating to ``rate``. The mount is rigid: the link poses the sensor prim with
    W = L_authored * W_body, in row-vector form, from the same state the body poses come from, since a
    Fabric world matrix has no hierarchy.

    Args:
        run: The run's values: the sensor's prim on the vehicle's own stage, the model body it rides,
            and the render link, :class:`~nexus_sim._src.rendering.KitRenderer`, whose ``tick`` sampling calls.
        rate: The sensor's physical rate, hertz.
    """

    KIND = "rtx"
    output = ""  # what the peer returns for it: "color", "radiance_depth" or "points"
    requires = ("kit",)  # the peers the build starts, once, for a vehicle that declares this sensor

    def __init__(self, run, *, rate: float):
        from pxr import UsdGeom

        from .rtx_stage import render_path

        prim = run.prim
        path = render_path(prim.GetPath(), prim.GetStage().GetDefaultPrim().GetPath())
        body = run.body
        self._link = run.link
        self._logger = None
        self.path = path  # on the render stage, where the peer renders the prim
        self.name = path.rsplit("/", 1)[-1].lower()  # for example 'fpvcam' -> sim/vehicle/sensors/fpvcam
        self.rate = float(rate)
        self.body = int(body)
        self._local = UsdGeom.Xformable(prim).GetLocalTransformation()  # authored mount, a row-vector Gf.Matrix4d
        self.mount = np.array(self._local, dtype=np.float64)
        self._next_due: float | None = None
        self._now = 0.0  # the tick in progress: an emit logs at the time its frame shows, then restores it
        logger.info(f"{type(self).__name__}: {path} @{self.rate:.0f}Hz -> sensors/{self.name} (body[{body}])")

    def set_logger(self, logger_) -> None:
        """Component-owned logging: the orchestrator hands over the Logger scoped to this sensor,
        and None = off.

        In debug mode, non-camera sensors log their coordinate frame once, statically, at their own row
        ``frame``, with the fixed local mount transform and the body's frame as its parent: the body's
        entity gets a ``Transform3D`` every logged tick, so the frame rides the body pose and nothing
        re-logs per sample. The frame has a row of its own because a lidar's points sit at the sensor's
        entity in world coordinates. Cameras carry their own Pinhole frustum instead.
        """
        self._logger = logger_
        if logger_ is not None and getattr(logger_, "debug", False) and self.KIND != "cameras":
            try:
                loc = self._local  # row-vector Gf: the fixed mount transform relative to the body
                logger_.log_static_frame(
                    "frame",
                    [loc[3][0], loc[3][1], loc[3][2]],
                    [[loc[i][j] for j in range(3)] for i in range(3)],
                    label=self.name,
                )
            except Exception as exc:
                logger.warning(f"{type(self).__name__} {self.name}: static frame logging unavailable: {exc!r}")

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
        """One host stage, named after the sensor, over :meth:`sample`."""
        return [Stage(self.name, "host", lambda tick: self.sample(tick.state, tick.t, tick.meas))]

    def sample(self, state, t, out) -> None:
        """Host-stage sample: the link sends the due sensors' frame and hands back the one before.

        Decimation is on sim time, since ``rate`` is the sensor's physical rate: a 30 Hz camera
        samples 30x per simulated second whatever the Real Time Factor (RTF).
        """
        self._now = float(t.sim_time)
        self._link.tick(t, state)

    def emit(self, arrays: dict, t_shown: float) -> None:
        """Log one frame; ``t_shown`` is the sim time the frame shows."""
        raise NotImplementedError
