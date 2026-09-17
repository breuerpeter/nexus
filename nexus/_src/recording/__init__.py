"""The observation sink: component-owned, capturable taps the host reads on demand.

The read-side twin of the logging seam, ``_src/logging``. Logging hands each component a ``Logger``
and the component emits its quantities to it: rerun, host-side, throttled. Observation is the same
shape to a different sink: each recordable component gets a :class:`Recorder`, registers a named
:class:`RecordChannel`, and writes its quantities into the channel's DEVICE ring buffer each tick via a
capturable kernel, so the tap joins the captured CUDA graph: no host readback in the hot loop, no
throttle. The host, a test, a benchmark or a CI gate, reads a channel on demand with one locked D2H.

Channel keys are component-kind-prefixed, and the access surface mirrors them. ``Sim.physics`` is a
name-keyed view over the per-body channels, ``physics/body/<label>``, and the per-joint channels,
``physics/joint/<label>``, that physics registers: ``sim.physics["body_frd"]`` → :class:`BodyState`,
``sim.physics["rotor_1_joint"]`` → :class:`JointState`. ``Sim.sensors`` is the same over the sensor
channels, ``sensors/<name>``, with flat instance names; the registering class rides along as ``source``
metadata.
"""

from .recorder import ChannelMap, RecordChannel, Recorder
from .sensor import SensorSample
from .state import BodyState, JointState

__all__ = ["BodyState", "ChannelMap", "JointState", "RecordChannel", "Recorder", "SensorSample"]
