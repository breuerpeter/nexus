"""The Recorder: every row a run records, read on demand by the host and written in blocks by the Logger.

The read-side twin of logging, ``_src/logging``. Logging hands each component a ``Logger`` and the
component emits its live rows to it: Rerun, host-side. Recording keeps a :class:`History` per body, per
joint and per sensor: a device kernel writes the row each tick into a small staging buffer, so the
write joins the captured CUDA graph with no host readback, and each time the buffer fills the rows
drain onto host blocks that grow with the run. The host, a script, a test or a CI gate, reads a history
on demand, and the Logger writes each drained block into the recording.

A history's key is its instance's path below the process root, the path its series take in a
recording, and the access surface mirrors the keys. ``Sim.physics`` is a name-keyed view over the
per-body histories, ``vehicle/body/<label>``, and the per-joint histories, ``vehicle/joints/<label>``,
that the Recorder registers from the plant: ``sim.physics["body_frd"]`` → :class:`BodyState`,
``sim.physics["rotor_1_joint"]`` → :class:`JointState`. ``Sim.sensors`` is the same over the sensor
histories, ``vehicle/sensors/<name>``, with flat instance names; the registering class rides along as
``source``.
"""

from .recorder import Histories, History, Recorder
from .sensor import SensorSample
from .state import BodyState, JointState

__all__ = ["BodyState", "Histories", "History", "JointState", "Recorder", "SensorSample"]
