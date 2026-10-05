"""A guidance logs its mission markers and its tracked reference under ``guidance/``, the path of
the component that logs them. A stand-in sink takes the place of the recording, and each test drives
the guidance through its stages, the seam the loop calls.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus._src.core.interfaces import Tick
from nexus._src.core.schema import SimTime
from nexus._src.guidance import MissionGuidance, TrackingGuidance

START = (0.0, 0.0, 2.0)


class _Controller:
    def accept_setpoint(self, sp):
        pass


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
    """The recording sink a guidance logs through: it keeps the entity path of every row."""

    def __init__(self):
        self.entities = []

    def log_points(self, entity, positions, **style):
        self.entities.append(entity)

    def log_strip(self, entity, points, **style):
        self.entities.append(entity)


def _tick(guidance, pos, sim_time):
    """Run the guidance's stages once, as the loop does on one tick."""
    tick = Tick(state=_State(pos), t=SimTime(sim_time, 0), dt=0.004, meas=None)
    for stage in guidance.stages():
        stage.run(tick)


def test_the_mission_markers_and_the_tracked_reference_sit_under_guidance_in_the_recording():
    """The mission markers and the tracked reference sit under `guidance/` in the recording."""
    sink = _Sink()
    mission = MissionGuidance(_Controller())
    mission.set_logger(sink)
    mission.set_mission([(1.0, 0.0, 2.0), (2.0, 0.0, 2.0), (3.0, 0.0, 2.0)])
    _tick(mission, START, 0.004)
    tracking = TrackingGuidance(_Controller(), planner=lambda waypoints: _Reference())
    tracking.set_logger(sink)
    tracking.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
    _tick(tracking, START, 0.004)
    assert sorted(set(sink.entities)) == [
        "guidance/reference",
        "guidance/waypoints/wp_0",
        "guidance/waypoints/wp_1",
        "guidance/waypoints/wp_2",
    ]
