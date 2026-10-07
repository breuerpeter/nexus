"""The sim clock during the preroll, while a controller's peer dials in."""

import pytest

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import Controls, SimTime

pytestmark = pytest.mark.usefixtures("warp_cpu")  # Python stand-ins run stage by stage, never as a graph


class _Clock:
    dt = 0.004

    def __init__(self):
        self._t = SimTime()

    def now(self):
        return self._t

    def advance(self):
        self._t = SimTime(self._t.sim_time + self.dt, self._t.step_index + 1)
        return self._t

    def throttle(self):
        pass


class _Physics:
    def reset(self):
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        return state

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]


class _Actuator:
    def forces(self, controls, state):
        pass

    def stages(self):
        return [Stage("forces", "device", lambda tick: self.forces(None, tick.state))]


class _LatePeerController:
    """A controller with a peer that dials in on the preroll's hundredth try and answers the first
    message it gets. It keeps the stamp of every message that reached the peer. Its work is the PX4
    shape, a ``read`` and an ``exchange`` host stage.
    """

    host_boundary = True

    def __init__(self):
        self._tries = 0
        self.stamps = []

    @property
    def attached(self):
        return self._tries >= 100

    def connect(self):
        pass

    def exchange(self, meas, t, timeout):
        reached = self.attached  # a message sent before the peer dials in reaches nobody
        self._tries += 1
        if not reached:
            return None
        self.stamps.append(t.sim_time)
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        pass

    def stages(self):
        def exchange(tick):
            return self.exchange(tick.meas, tick.t, None) is not None

        return [Stage("read", "host", lambda tick: None), Stage("exchange", "host", exchange)]


def test_a_peer_that_dials_in_late_gets_its_first_stamp_near_zero():
    """A controller with a host stage and a peer holds the sim clock until the peer attaches, and its
    first stamp lands near zero: the first message the peer gets carries a stamp under 10 ms.
    """
    controller = _LatePeerController()
    orch = Orchestrator(
        clock=_Clock(),
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=[],
        controller=controller,
    )
    orch.step()
    assert controller.stamps[0] < 0.010
