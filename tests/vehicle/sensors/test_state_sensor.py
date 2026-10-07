"""StateSensor passes the live state through ``Measurement.state``, the ground-truth
state a privileged controller reads. A pure passthrough: no device work.
"""

from nexus_sim._src.core.schema import Measurement, SimTime
from nexus_sim._src.vehicle.sensors import StateSensor


def test_state_sensor_passes_view_through():
    view = object()  # any state-like; StateSensor just forwards it
    meas = Measurement()
    assert meas.state is None
    StateSensor().sample(view, SimTime(0.0, 0), meas)
    assert meas.state is view
