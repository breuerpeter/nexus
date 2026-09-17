"""StateSensor passes the live state through ``Measurement.state``, the ground-truth
state-feedback seam a privileged controller reads. A pure passthrough: no device work.
"""

from nexus._src.core.schema import EnvSample, Measurement, SimTime
from nexus._src.vehicle.sensors import StateSensor


def test_state_sensor_passes_view_through():
    view = object()  # any state-like; StateSensor just forwards it
    meas = Measurement()
    assert meas.state is None
    StateSensor().sample(view, EnvSample(), SimTime(0.0, 0), meas)
    assert meas.state is view
