"""The stand-in project's sensor class, which `StandInSensorAPI` builds, and its output's type."""

import numpy as np

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.signals import DeviceType, Signal


class Gain(DeviceType):
    """The stand-in sensor's output, a signal type the stand-in project defines: its gain, one 32-bit float."""

    dtype = "float32"


class StandInSensor:
    """A sensor that keeps the seed and the tick the run hands it, and writes its gain where a reader reads it.

    Its stage writes the gain as the signal `gain`, of the project's own type `Gain`, and into the
    `Measurement`.

    Args:
        run: The run's values for this sensor; it reads `seed` and `dt`.
        gain: The schema's `nexus:gain`.
    """

    def __init__(self, run, gain: float = 1.0):
        self.seed = run.seed
        self.dt = run.dt
        self.gain = float(gain)
        self.out = Signal("gain", Gain, shape=(1,))

    def stages(self) -> list[Stage]:
        """One host stage, which writes the gain where a reader reads it."""
        return [Stage("stand_in", "host", self._sample, writes=(self.out,))]

    def _sample(self, tick) -> None:
        self.out.write(np.array([self.gain], dtype=np.float32))
        tick.meas.eph = self.gain
