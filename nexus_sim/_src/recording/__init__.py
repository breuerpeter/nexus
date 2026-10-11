"""The Recorder: every row a run records, read on demand by the host and written in blocks by the Logger.

The read-side twin of logging, ``_src/logging``. Logging hands each component a ``Logger`` and the
component logs live what has no fixed width: Rerun, host-side. Recording keeps a :class:`History` per
device signal a component writes, the plant's bodies and joints among them: a device kernel writes the
row each tick into a small staging buffer, so the write joins the captured CUDA graph with no host
readback, and each time the buffer fills the rows drain onto host blocks that grow with the run. The
host, a script, a test or a CI gate, reads a history on demand, and the Logger writes each drained block
into the recording. No component records itself: the Recorder taps each signal's buffer.

A history's key is its writer's path below the process root, the path its rows take in a recording, then
the signal's name, and a script reads a history by that key, indexing the run: the plant's,
``sim["vehicle/body/<label>"]`` and ``sim["vehicle/joints/<label>"]``, and every other component's,
``sim["vehicle/sensors/<name>/<signal>"]``. Every history reads one way: ``latest()`` is one record of
``t`` then the quantities the signal's type declares, ``history()`` one structured array of them, and
``arrays()`` one array per quantity, with the writing class along as ``source``.
"""

from .recorder import History, Recorder

__all__ = ["History", "Recorder"]
