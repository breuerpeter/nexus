"""The base of every RTX sensor: a host sensor over a prim the vehicle file authored, rendered by the Kit peer.

The prims are body children the vehicle's Universal Scene Description (USD) file authors, found
with no flags. Each sensor decimates to its
authored physical rate on sim time. At a due tick the render link sends the poses and the loop flies
on; the frame comes back on a later tick, stamped with the sim time it shows, and the sensor logs
and streams it here on the host.
"""

from __future__ import annotations

import numpy as np

from nexus._src.core import logger


def _prim_sensor_attr(prim, name: str, fallback, kind: str):
    """Read an authored ``sensor:<name>`` custom attr off an RTX prim; the vehicle USD is the
    single authority for its sensors' render params, resolution and rate. Falls back, with a warning,
    only for legacy assets authored before the attrs existed; author the attr on the prim to
    silence it.
    """
    attr = prim.GetAttribute(f"sensor:{name}")
    if attr and attr.HasAuthoredValue():
        return attr.Get()
    logger.warning(
        f"{kind} {prim.GetPath()}: no authored sensor:{name}, using the fallback {fallback!r} "
        "(re-author the vehicle USD; the asset is the authority for its sensors)"
    )
    return fallback


class RtxMountedSensor:
    """Shared base for RTX sensors mounted on a vehicle body: a host-rate ``Sensor`` over a prim
    authored in the vehicle USD as a body child.

    Sampled at the host seam of every loop, self-decimating to ``rate_hz``, never vetoing the
    captured strategy, per ``host_rate``. The mount is rigid: the link poses the sensor prim with
    W = L_authored * W_body, in row-vector form, from the same state the body poses come from, since a
    Fabric world matrix has no hierarchy.

    Args:
        link: The render link, :class:`~nexus._src.rendering.KitRenderer`: sampling calls its ``tick``.
        prim: The sensor's prim on the vehicle's own stage.
        path: The prim's path on the render stage, where the peer renders it.
        body: The model body index the sensor rides.
        rate_hz: The sensor's physical rate.
    """

    host_rate = True  # host-bound + low-rate: sampled at the host seam, excluded from the capture gate
    KIND = "rtx"
    output = ""  # what the peer returns for it: "color", "radiance_depth" or "points"

    def __init__(self, link, prim, *, path: str, body: int, rate_hz: float):
        from pxr import UsdGeom

        self._link = link
        self._logger = None
        self.path = path
        self.name = path.rsplit("/", 1)[-1].lower()  # for example 'fpvcam' -> cameras/fpvcam
        self.rate_hz = float(rate_hz)
        self.body = int(body)
        self._local = UsdGeom.Xformable(prim).GetLocalTransformation()  # authored mount, a row-vector Gf.Matrix4d
        self.mount = np.array(self._local, dtype=np.float64)
        self._next_due: float | None = None
        self._now = 0.0  # the tick in progress: an emit logs at the time its frame shows, then restores it
        logger.info(f"{type(self).__name__}: {path} @{self.rate_hz:.0f}Hz -> {self.KIND}/{self.name} (body[{body}])")

    def set_logger(self, logger_) -> None:
        """Component-owned logging seam; the orchestrator hands the Logger over, and None = off.

        In debug mode, non-camera sensors log their coordinate frame once, statically, under their USD
        prim path, a child of the body prim, with the fixed local mount transform: the debug scene logs
        a per-tick ``Transform3D`` on each body entity, so the child frame inherits the body pose and
        nothing re-logs per sample. Cameras carry their own Pinhole frustum instead.
        """
        self._logger = logger_
        if logger_ is not None and getattr(logger_, "debug", False) and self.KIND != "cameras":
            try:
                loc = self._local  # row-vector Gf: the fixed mount transform relative to the body
                logger_.log_static_frame(
                    self.path,
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
        period = 1.0 / self.rate_hz
        base = self._next_due if self._next_due is not None else now
        self._next_due = max(base + period, now + 0.5 * period)  # never burst after a stall
        return True

    def sample(self, state, env, t, out) -> None:
        """Host-seam sample: the link sends the due sensors' frame and hands back the one before.

        Decimation is on sim time, since ``rate_hz`` is the sensor's physical rate: a 30 Hz camera
        samples 30x per simulated second whatever the Real Time Factor (RTF).
        """
        self._now = float(t.sim_time)
        self._link.tick(t, state)

    def emit(self, arrays: dict, t_shown: float) -> None:
        """Log and stream one frame; ``t_shown`` is the sim time the frame shows."""
        raise NotImplementedError

    def close(self) -> None:
        pass
