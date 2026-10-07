"""A guidance names only its own rows: ``waypoints/wp_<i>`` and ``reference``. The loop scopes its
logger to the guidance's path, which ``tests/core/test_orchestrator_guidance.py`` proves. A stand-in
sink takes the place of that scoped logger, and each test drives the guidance through its stages, the
seam the loop calls.
"""

import pytest

pytest.importorskip("warp")

from nexus_sim._src.core.interfaces import Stage, Tick
from nexus_sim._src.core.schema import PoseTwist, SimTime
from nexus_sim._src.core.signals import Signal, wire
from nexus_sim._src.core.stages import Bound
from nexus_sim._src.guidance import MissionGuidance, TrackingGuidance

START = (0.0, 0.0, 2.0)


class _Estimator:
    """Stands in for the estimator: it writes the estimate the guidance reads, which each tick sets."""

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1, 13))

    def stages(self):
        return [Stage("estimate", "device", lambda tick: None, writes=(self.estimate,))]


class _Reference:
    """A planned reference, the shape a tracking guidance hands its controller."""

    duration = 4.0

    def set_start(self, p0):
        pass

    def reference_path(self):
        return [START, (2.0, 0.5, 3.5)]


class _Sink:
    """The scoped logger a guidance logs through: it keeps the name of every row."""

    def __init__(self):
        self.rows = []

    def log_points(self, name, positions, **style):
        self.rows.append(name)

    def log_strip(self, name, points, **style):
        self.rows.append(name)


def _wired(guidance):
    """The guidance with its estimate and setpoint wired, as the loop wires them before any stage runs."""
    estimator = _Estimator()
    wire(
        [
            Bound(stage, c, role)
            for c, role in ((estimator, "estimator"), (guidance, "guidance"))
            for stage in c.stages()
        ]
    )
    return guidance


def _tick(guidance, pos, sim_time):
    """Run the guidance's stages once, as the loop does on one tick."""
    guidance.estimate.write([[*pos, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # the vehicle at `pos`
    tick = Tick(state=None, t=SimTime(sim_time, 0), dt=0.004, meas=None)
    for stage in guidance.stages():
        stage.run(tick)


def test_a_guidance_names_only_its_own_rows():
    """A guidance names only its own rows, its waypoints and its tracked reference, and no path."""
    sink = _Sink()
    mission = _wired(MissionGuidance())
    mission.set_logger(sink)
    mission.set_mission([(1.0, 0.0, 2.0), (2.0, 0.0, 2.0), (3.0, 0.0, 2.0)])
    _tick(mission, START, 0.004)
    tracking = _wired(TrackingGuidance(planner=lambda waypoints: _Reference()))
    tracking.set_logger(sink)
    tracking.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
    _tick(tracking, START, 0.004)
    assert sorted(set(sink.rows)) == ["reference", "waypoints/wp_0", "waypoints/wp_1", "waypoints/wp_2"]
