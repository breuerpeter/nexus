"""A guidance names only its own rows: ``waypoints/wp_<i>`` and ``reference``. The loop scopes its
logger to the guidance's path, which ``tests/core/test_orchestrator_guidance.py`` proves. A stand-in
sink takes the place of that scoped logger, and each test drives the guidance through its stages,
which the loop calls.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Tick
from nexus_sim._src.core.schema import SimTime
from nexus_sim._src.core.signals import wire
from nexus_sim._src.core.stages import Bound
from nexus_sim._src.guidance import MissionGuidance, TrackingGuidance

START = (0.0, 0.0, 2.0)


class _State:
    """The physics state a guidance reads: one body at a set position."""

    def __init__(self, pos):
        self.pos = pos

    @property
    def body_q(self):
        return wp.array(np.array([[*self.pos, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


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
    """The guidance with its setpoint wired, as the loop wires it before any stage runs."""
    wire([Bound(stage, guidance, "guidance") for stage in guidance.stages()])
    return guidance


def _tick(guidance, pos, sim_time):
    """Run the guidance's stages once, as the loop does on one tick."""
    tick = Tick(state=_State(pos), t=SimTime(sim_time, 0), dt=0.004, meas=None)
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
