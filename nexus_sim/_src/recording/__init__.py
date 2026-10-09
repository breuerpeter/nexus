"""The Recorder: every row a run records, read on demand by the host and written in blocks by the Logger.

The read-side twin of logging, ``_src/logging``. Logging hands each component a ``Logger`` and the
component logs live what has no fixed width: Rerun, host-side. Recording keeps a :class:`History` per
body, per joint and per device signal a component writes: a device kernel writes the row each tick into
a small staging buffer, so the write joins the captured CUDA graph with no host readback, and each time
the buffer fills the rows drain onto host blocks that grow with the run. The host, a script, a test or a
CI gate, reads a history on demand, and the Logger writes each drained block into the recording. No
component records itself: the Recorder taps the plant's state and each signal's buffer.

A history's key is its writer's path below the process root, the path its rows take in a recording, then
the signal's name for a signal, and the access surface mirrors the keys. ``Sim.physics`` is a name-keyed
view over the per-body histories, ``vehicle/body/<label>``, and the per-joint histories,
``vehicle/joints/<label>``, that the Recorder registers from the plant: ``sim.physics["body_frd"]`` →
:class:`BodyState`, ``sim.physics["rotor_1_joint"]`` → :class:`JointState`. ``Sim.sensors`` is the same
over the sensors' histories, ``vehicle/sensors/<name>/<signal>``, by instance name: a row reads as a record
of ``t`` then the quantities the signal's type declares, and the writing class rides along as ``source``.
"""

from .recorder import Histories, History, Recorder
from .state import BodyState, JointState

__all__ = ["BodyState", "Histories", "History", "JointState", "Recorder"]
