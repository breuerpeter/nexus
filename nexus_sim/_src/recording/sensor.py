"""The sensors' histories: each sensor's device buffer → ``SensorSample``.

A sensor already fills a persistent device buffer, ``_out``, each tick. :class:`SensorRecorder` is a
mixin that makes it recordable: ``set_recorder`` registers a ``vehicle/sensors/<name>`` history and
``record_wp`` copies ``_out`` into it inside the captured graph, device-only, no D2H. The host reads it
via ``sim.sensors["imu"]`` → a :class:`SensorSample` whose ``fields`` map the sensor's documented
layout, for example the Inertial Measurement Unit (IMU) ``xacc``/…/``zgyro``/quat. The Global
Positioning System (GPS) sensor uses a ``float64`` history so lat/lon survive. #250 moves this into
the Recorder, which then records every signal itself.

The history's key ends in the flat instance name, ``vehicle/sensors/imu``, and the registering class
rides along as ``source`` metadata: a future redundant setup registers sibling instances,
``vehicle/sensors/imu_bosch`` / ``vehicle/sensors/imu_murata``, never a nested kind level; the
Recorder's duplicate guard makes a name collision loud instead of silently sharing one buffer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(slots=True)
class SensorSample:
    """One sensor measurement: its named fields at sim time ``t``, in the sensor's documented layout."""

    t: float
    """Sim time of the sample, in seconds (the lockstep clock: counter × dt)."""
    fields: dict[str, float]
    """Field name → value, per the sensor's layout (e.g. ``{"xacc": .., "zgyro": .., "qw": ..}``)."""


@wp.kernel
def record_sensor(
    src: wp.array(dtype=float),
    n: int,
    dt: float,
    staging: int,
    buf: wp.array(dtype=float, ndim=2),
    counter: wp.array(dtype=int),
):
    c = counter[0]  # advances per graph replay: on-device, not a frozen kernel arg
    s = c % staging  # the staging slot
    buf[s, 0] = dt * wp.float32(c)
    for k in range(n):
        buf[s, 1 + k] = src[k]
    counter[0] = c + 1


@wp.kernel
def record_sensor_f64(
    src: wp.array(dtype=wp.float64),
    n: int,
    dt: float,
    staging: int,
    buf: wp.array(dtype=wp.float64, ndim=2),
    counter: wp.array(dtype=int),
):
    c = counter[0]
    s = c % staging
    buf[s, 0] = wp.float64(dt) * wp.float64(c)
    for k in range(n):
        buf[s, 1 + k] = src[k]
    counter[0] = c + 1


def make_decode_sensor(fields: tuple[str, ...]) -> Callable[[np.ndarray], SensorSample]:
    """Build the row→:class:`SensorSample` decode mapping the row's values onto the sensor's field names."""

    def decode(r) -> SensorSample:
        return SensorSample(t=float(r[0]), fields={f: float(r[1 + i]) for i, f in enumerate(fields)})

    return decode


class SensorRecorder:
    """Mixin that makes a sensor recordable, the read-side twin of logging.

    The sensor declares ``name``, the ``sim.sensors`` key, which is the flat instance name, and
    ``fields``, its ``_out`` layout; it already owns the device buffer ``_out``. ``set_recorder``
    registers the history, ``float64`` when ``_out`` is, for example GPS, with the per-field schema +
    the implementing class as ``source``, and ``record_wp`` copies ``_out`` into it each tick inside
    the captured graph.
    """

    name: str
    fields: tuple[str, ...]

    def set_recorder(self, recorder) -> None:
        f64 = self._out.dtype == wp.float64
        self._sensor_ch = recorder.history(
            f"vehicle/sensors/{self.name}",
            width=1 + len(self.fields),
            decode=make_decode_sensor(self.fields),
            dtype=wp.float64 if f64 else float,
            fields=tuple((f, 1) for f in self.fields),
            source=type(self).__name__,
        )
        self._sensor_kernel = record_sensor_f64 if f64 else record_sensor

    def record_wp(self) -> None:
        ch = getattr(self, "_sensor_ch", None)
        if ch is None:  # no Recorder handed over
            return
        wp.launch(
            self._sensor_kernel, dim=1, inputs=(self._out, len(self.fields), ch.dt, ch.staging, ch.buf, ch.counter)
        )
