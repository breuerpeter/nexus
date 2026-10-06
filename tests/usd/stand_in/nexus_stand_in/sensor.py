"""The stand-in project's sensor class, which `StandInSensorAPI` builds."""

from nexus_sim._src.core.interfaces import Stage


class StandInSensor:
    """A sensor that keeps the seed and the tick the run hands it, and writes its gain into the `Measurement`.

    Args:
        run: The run's values for this sensor; it reads `seed` and `dt`.
        gain: The schema's `nexus:gain`.
    """

    def __init__(self, run, gain: float = 1.0):
        self.seed = run.seed
        self.dt = run.dt
        self.gain = float(gain)

    def stages(self) -> list[Stage]:
        """One host stage, which writes the gain where a controller reads it."""
        return [Stage("stand_in", "host", self._sample)]

    def _sample(self, tick) -> None:
        tick.meas.eph = self.gain
