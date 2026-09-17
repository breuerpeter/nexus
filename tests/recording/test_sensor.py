"""Sensor-measurement channels: the SensorRecorder mixin copies a sensor's device ``_out`` into a
``sensors/<name>`` channel each tick, capturable; the host reads a SensorSample by field name.
Global Positioning System (GPS) uses a float64 channel so lat/lon survive.
"""

import numpy as np
import warp as wp

from nexus._src.recording import Recorder, SensorSample
from nexus._src.recording.sensor import SensorRecorder


class _FakeImu(SensorRecorder):
    name = "imu"
    fields = ("ax", "ay", "az")

    def __init__(self):
        self._out = wp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=float)


class _FakeGps(SensorRecorder):
    name = "gps"
    fields = ("lat", "lon")

    def __init__(self):
        self._out = wp.array(np.array([47.3977, 8.5456], dtype=np.float64), dtype=wp.float64)


def test_sensor_records_fields_by_name():
    rec = Recorder(dt=0.004)
    s = _FakeImu()
    s.set_recorder(rec)
    s.record_wp()
    s.record_wp()  # two ticks
    ch = rec.channels["sensors/imu"]
    samp = ch.latest()
    assert isinstance(samp, SensorSample)
    assert samp.fields == {"ax": 1.0, "ay": 2.0, "az": 3.0}
    assert round(samp.t, 6) == 0.004  # second row → counter=1 → t = dt × 1
    assert len(ch.history()) == 2


def test_sensor_f64_channel_preserves_gps_precision():
    rec = Recorder(dt=0.004)
    s = _FakeGps()
    s.set_recorder(rec)
    s.record_wp()
    ch = rec.channels["sensors/gps"]
    assert ch.dtype == wp.float64  # not f32: lat/lon would lose ~metres in f32
    f = ch.latest().fields
    assert abs(f["lat"] - 47.3977) < 1e-9
    assert abs(f["lon"] - 8.5456) < 1e-9


def test_sensor_declares_schema_and_source():
    rec = Recorder(dt=0.004)
    s = _FakeGps()
    s.set_recorder(rec)
    s.record_wp()
    ch = rec.channels["sensors/gps"]
    assert ch.fields == (("lat", 1), ("lon", 1))  # per-field schema, auto-derived from the sensor
    assert ch.source == "_FakeGps"  # the registering impl class: kind metadata for tools
    arrays = ch.history_arrays()
    assert arrays["lat"].dtype == np.float64  # vectorized readback keeps the channel dtype
    assert abs(arrays["lat"][0] - 47.3977) < 1e-9
    assert arrays["lon"].shape == (1,)  # width-1 quantities come back 1-D
