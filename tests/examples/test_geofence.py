"""GeofenceGuidance, the guidance demonstrator of the goto policy flight: a mission guidance that
ends the run when the vehicle leaves its box. Each test drives the guidance through its stages, the
seam the loop calls, with a stand-in state and recording sink, and reads what the stage wrote: the goal
in its setpoint signal, wired as the loop wires it, and whether the tick's mission is over.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Tick
from nexus_sim._src.core.schema import SimTime
from nexus_sim._src.core.signals import wire
from nexus_sim._src.core.stages import Bound
from nexus_sim.examples.controllers.policy.goto.geofence import GeofenceGuidance

BOUNDS = ((-3.0, -3.0, 0.2), (3.0, 3.0, 4.0))
INSIDE = (0.0, 0.0, 2.0)
OUTSIDE = (4.0, 0.0, 2.0)
MISSION = [(1.0, 0.0, 2.0), (2.0, 0.0, 2.0)]
RED = (255, 0, 0)


class _State:
    """The physics state a guidance reads: one body at a set position."""

    def __init__(self, pos):
        self.pos = pos

    @property
    def body_q(self):
        return wp.array(np.array([[*self.pos, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


class _Sink:
    """The scoped logger a guidance logs through: it keeps each row's name, points and color."""

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


def _wired(guidance):
    """The guidance with its setpoint wired, as the loop wires it before any stage runs."""
    wire([Bound(stage, guidance, "guidance") for stage in guidance.stages()])
    return guidance


def _goal(guidance) -> tuple[float, float, float]:
    """The goal the guidance's setpoint holds."""
    return tuple(float(v) for v in guidance.setpoint.read()[0])


def _tick(guidance, pos, sim_time) -> Tick:
    """Run the guidance's stages once, as the loop does on one tick, and return that tick."""
    tick = Tick(state=_State(pos), t=SimTime(sim_time, 0), dt=0.004, meas=None)
    for stage in guidance.stages():
        stage.run(tick)
    return tick


def test_a_geofence_guidance_ends_the_run_when_the_vehicle_leaves_its_box():
    """A geofence guidance ends the run when the vehicle leaves its box: its stage marks that tick done,
    which the loop ends the run on.
    """
    guidance = _wired(GeofenceGuidance(bounds=BOUNDS))
    guidance.set_mission(MISSION)
    inside = _tick(guidance, INSIDE, 0.25)
    outside = _tick(guidance, OUTSIDE, 0.5)
    ended = (inside.done, outside.done, guidance.breached_at, tuple(guidance.breach_pos))
    assert ended == (False, True, 0.5, OUTSIDE)


def test_after_a_breach_the_mission_no_longer_advances():
    """After a breach the mission no longer advances."""
    guidance = _wired(GeofenceGuidance(bounds=BOUNDS, reached_m=0.3))
    guidance.set_mission(MISSION)
    _tick(guidance, OUTSIDE, 0.5)
    _tick(guidance, MISSION[0], 0.75)  # within reach of the active goal
    assert (_goal(guidance), guidance.reached) == (MISSION[0], 0)


def test_a_new_mission_clears_an_earlier_breach():
    """A new mission clears an earlier breach."""
    guidance = _wired(GeofenceGuidance(bounds=BOUNDS))
    guidance.set_mission(MISSION)
    _tick(guidance, OUTSIDE, 0.5)
    guidance.set_mission([(0.0, 1.0, 2.0)])
    tick = _tick(guidance, INSIDE, 0.75)
    cleared = (guidance.breached_at, guidance.breach_pos, _goal(guidance), tick.done)
    assert cleared == (None, None, (0.0, 1.0, 2.0), False)


def test_the_recording_shows_the_fence_and_a_breach_as_a_red_marker():
    """The recording shows the fence, and a breach as a red marker."""
    sink = _Sink()
    guidance = _wired(GeofenceGuidance(bounds=BOUNDS))
    guidance.set_logger(sink)
    guidance.set_mission(MISSION)
    _tick(guidance, INSIDE, 0.25)
    fence_at_start = any(name.startswith("fence/") for name, _, _ in sink.rows)
    _tick(guidance, OUTSIDE, 0.5)
    breach = [(points, color) for name, points, color in sink.rows if name == "breach"]
    assert (fence_at_start, breach) == (True, [([OUTSIDE], RED)])
