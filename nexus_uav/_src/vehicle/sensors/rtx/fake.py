"""The render stage's fake: a stand-in that renders nothing and hands each due RTX sensor a synthetic output.

It keeps the render stage's side of the contract, :meth:`add` at a sensor's ``connect`` and :meth:`tick` from its host
stage, and answers each due sensor at once with a blank output at the size it declares, stamped with the tick's sim
time. So the RTX sensors and the loop run as they do over the renderer, and a test proves their side with no GPU and
no renderer. It proves nothing about the rendering.

A test sends the render stage to this class with a row of its own registry, ``ovrtx``.
"""

from __future__ import annotations

import numpy as np

from .stage import NAME


def _outputs(sensor, points: np.ndarray) -> dict:
    """A synthetic output for `sensor`: each array the render stage hands it, blank, and `points` for a lidar's scan."""
    h, w = int(sensor.height), int(sensor.width)
    if sensor.output == "color":
        return {"color": np.zeros((h, w, 3), dtype=np.uint8)}
    if sensor.output == "radiance_depth":
        return {"radiance": np.zeros((h, w), dtype=np.float32), "depth": np.ones((h, w), dtype=np.float32)}
    return {"points": points}


class RenderStageFake:
    """A stand-in for one run's render stage, built and started as the render stage is.

    A test sends the render stage to this class with a row of its own registry, ``ovrtx``, and to one that answers
    otherwise with a row for :meth:`configured`.

    Args:
        files: The scene description files the run reads; the fake reads none of them.
        vehicle_file: The vehicle's root file, whose stage a sensor reads its prim from at ``connect``.
        points: The world points every lidar scan holds, one ``(x, y, z)`` each; none by default.
        **run: The run's other values, which the fake needs none of.
    """

    name = NAME
    clock = "on_request"

    def __init__(self, files: list = (), *, vehicle_file: str, points=(), **run) -> None:
        self.vehicle_file = str(vehicle_file)
        self.points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        self.vehicle_stage = None
        self.sensors: list = []
        self._last_t: float | None = None
        self._up = False

    @classmethod
    def configured(cls, **answers) -> type:
        """A subclass of the fake that answers as `answers` say, the keyword ``points``, for a registry to map
        ``ovrtx`` to.
        """

        class Configured(cls):
            def __init__(self, files: list = (), **run) -> None:
                super().__init__(files, **{**run, **answers})

        return Configured

    def start(self) -> None:
        from pxr import Usd

        self.vehicle_stage = Usd.Stage.Open(self.vehicle_file)
        self._up = True

    def alive(self) -> bool:
        return self._up

    def stop(self) -> None:
        self._up = False

    def add(self, sensor) -> None:
        """Take `sensor`, which the fake answers at each of its due ticks."""
        self.sensors.append(sensor)

    def tick(self, t, state) -> None:
        """At a tick where any sensor is due: hand each due sensor its synthetic output, stamped with the tick's sim
        time. A later call at the same sim time returns at once.
        """
        now = float(t.sim_time)
        if now == self._last_t:
            return
        self._last_t = now
        for sensor in [s for s in self.sensors if s.due(t.time_ns)]:
            sensor.emit(_outputs(sensor, self.points), now)


__all__ = ["RenderStageFake"]
