"""The loop hands a component the links of the peers its class requires at ``connect``, keyed by the peer's name,
and fails a component that requires a peer the run holds no link to. Python stand-ins, no physics.
"""

import pytest

from nexus_sim._src.core.interfaces import Component, Stage
from nexus_sim._src.core.loop import Loop
from nexus_sim._src.core.schema import SimTime

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

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: None),
            Stage("step", "device", lambda tick: None),
            Stage("state", "device", lambda tick: None),
        ]


class _Controller(Component):
    """A stand-in controller with no peer: its exchange answers at once."""

    def stages(self):
        return [Stage("exchange", "host", lambda tick: True)]


class _Sensor(Component):
    """A stand-in sensor whose class requires the peer `kit`: it keeps the links its connect took."""

    requires = ("kit",)

    def __init__(self):
        self.links = None

    def connect(self, links):
        self.links = dict(links)

    def stages(self):
        return [Stage("sample", "host", lambda tick: None)]


def _loop(sensor, **kw):
    return Loop(clock=_Clock(), physics=_Physics(), sensors=[sensor], controller=_Controller(), **kw)


def test_a_component_takes_the_link_of_each_peer_it_requires_at_connect():
    """A component takes the link of each peer its class requires at connect, keyed by the peer's name.

    Given a run that holds a link under `kit` and another under `px4_sitl`, and a sensor whose class requires
    `kit`, when the run starts, then the sensor's connect took the `kit` link alone.
    """
    sensor, kit = _Sensor(), object()
    loop = _loop(sensor, links={"kit": kit, "px4_sitl": object()})
    loop.step()
    loop.close()

    assert sensor.links == {"kit": kit}


def test_a_component_that_requires_a_peer_the_run_holds_no_link_to_fails_naming_both():
    """A component that requires a peer the run holds no link to fails the run, and the message names the
    component and the peer.

    Given a run that holds no link and a sensor whose class requires `kit`, when the run starts, then it fails
    and the message names the sensor and `kit`.
    """
    loop = _loop(_Sensor())

    with pytest.raises(ValueError, match=r"_Sensor.*kit"):
        loop.step()
