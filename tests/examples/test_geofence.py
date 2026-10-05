"""GeofenceGuidance, the guidance demonstrator of the goto policy flight: a mission guidance that
ends the run when the vehicle leaves its box. Each test drives the guidance through its stages, the
seam the loop calls, with a stand-in controller, state and recording sink.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus._src.core.interfaces import Tick
from nexus._src.core.schema import SimTime
from nexus.examples.controllers.policy.goto.geofence import GeofenceGuidance

BOUNDS = ((-3.0, -3.0, 0.2), (3.0, 3.0, 4.0))
INSIDE = (0.0, 0.0, 2.0)
OUTSIDE = (4.0, 0.0, 2.0)
MISSION = [(1.0, 0.0, 2.0), (2.0, 0.0, 2.0)]
RED = (255, 0, 0)


class _Controller:
    """A controller that takes setpoints, and keeps each one it receives."""

    def __init__(self):
        self.setpoints = []

    def accept_setpoint(self, sp):
        self.setpoints.append(sp)


class _State:
    """The physics state a guidance reads: one body at a set position."""

    def __init__(self, pos):
        self.pos = pos

    @property
    def body_q(self):
        return wp.array(np.array([[*self.pos, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


class _Sink:
    """The recording sink a guidance logs through: it keeps each row's entity path, points and color."""

    def __init__(self):
        self.rows = []

    def _keep(self, entity, points, color):
        self.rows.append(
            (entity, [tuple(float(v) for v in p) for p in points], None if color is None else tuple(color))
        )

    def log_points(self, entity, positions, *, colors=None, **style):
        self._keep(entity, positions, colors)

    def log_strip(self, entity, points, *, color=None, **style):
        self._keep(entity, points, color)


def _tick(guidance, pos, sim_time):
    """Run the guidance's stages once, as the loop does on one tick."""
    tick = Tick(state=_State(pos), t=SimTime(sim_time, 0), dt=0.004, meas=None)
    for stage in guidance.stages():
        stage.run(tick)


def test_a_geofence_guidance_ends_the_run_when_the_vehicle_leaves_its_box():
    """A geofence guidance ends the run when the vehicle leaves its box."""
    stops = []
    guidance = GeofenceGuidance(_Controller(), bounds=BOUNDS, stop=lambda: stops.append(1))
    guidance.set_mission(MISSION)
    _tick(guidance, INSIDE, 0.25)
    _tick(guidance, OUTSIDE, 0.5)
    assert (len(stops), guidance.breached_at, tuple(guidance.breach_pos)) == (1, 0.5, OUTSIDE)


def test_after_a_breach_the_mission_no_longer_advances():
    """After a breach the mission no longer advances."""
    controller = _Controller()
    guidance = GeofenceGuidance(controller, bounds=BOUNDS, reached_m=0.3)
    guidance.set_mission(MISSION)
    _tick(guidance, OUTSIDE, 0.5)
    _tick(guidance, MISSION[0], 0.75)  # within reach of the active goal
    assert [tuple(sp.pos) for sp in controller.setpoints] == [MISSION[0]]


def test_a_new_mission_clears_an_earlier_breach():
    """A new mission clears an earlier breach."""
    controller = _Controller()
    guidance = GeofenceGuidance(controller, bounds=BOUNDS)
    guidance.set_mission(MISSION)
    _tick(guidance, OUTSIDE, 0.5)
    guidance.set_mission([(0.0, 1.0, 2.0)])
    _tick(guidance, INSIDE, 0.75)
    cleared = (guidance.breached_at, guidance.breach_pos, tuple(controller.setpoints[-1].pos))
    assert cleared == (None, None, (0.0, 1.0, 2.0))


def test_the_recording_shows_the_fence_and_a_breach_as_a_red_marker():
    """The recording shows the fence, and a breach as a red marker."""
    sink = _Sink()
    guidance = GeofenceGuidance(_Controller(), bounds=BOUNDS)
    guidance.set_logger(sink)
    guidance.set_mission(MISSION)
    _tick(guidance, INSIDE, 0.25)
    fence_at_start = any(entity.startswith("guidance/fence") for entity, _, _ in sink.rows)
    _tick(guidance, OUTSIDE, 0.5)
    breach = [(points, color) for entity, points, color in sink.rows if entity == "guidance/breach"]
    assert (fence_at_start, breach) == (True, [([OUTSIDE], RED)])
