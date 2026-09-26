"""The sim clock during the preroll, while a controller's peer dials in."""

from nexus._src.core.orchestrator import Orchestrator
from nexus._src.core.schema import Controls, SimTime


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


class _Env:
    def sample(self, pos, t):
        return None


class _Physics:
    def reset(self):
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, env, dt):
        return state


class _Actuator:
    def forces(self, controls, state, env):
        pass


class _LatePeerController:
    """A host-boundary controller whose peer dials in on the preroll's hundredth try and answers the
    first message it gets. It keeps the stamp of every message that reached the peer.
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


def test_a_peer_that_dials_in_late_gets_its_first_stamp_near_zero():
    """A peer that dials in late gets its first stamp near zero: the preroll holds the sim clock until
    the controller reports its link attached, so the first message the peer gets carries a stamp
    under 10 ms.
    """
    controller = _LatePeerController()
    orch = Orchestrator(
        clock=_Clock(),
        environment=_Env(),
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=[],
        controller=controller,
    )
    orch.step()
    assert controller.stamps[0] < 0.010
