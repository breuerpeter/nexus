"""One contract serves every role, and the loop and ``Sim`` read its members from every component that states
them, #44.

A real build of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, with a stand-in
sensor that waits on its peer and a stand-in controller, hosted by ``Sim``. Skipped without newton or pxr.
"""

from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import Controls
from nexus_sim._src.core.stages import peer_stages
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

DT = 0.004  # the fixture's 250 Hz tick


class _Lifecycle:
    """What the loop did to a component through the contract: each connect and close, in order."""

    def __init__(self):
        self.lifecycle: list[str] = []

    def connect(self):
        self.lifecycle.append("connect")

    def close(self):
        self.lifecycle.append("close")


class _WaitingSensor(_Lifecycle):
    """A stand-in sensor that waits on its peer: attached once its host stage has run three times. Each run notes
    the tick's sim time, and its artifact is its peer's log.
    """

    instances: ClassVar[list] = []

    def __init__(self, run, **kwargs):
        super().__init__()
        self.times: list[float] = []
        type(self).instances.append(self)

    @property
    def attached(self) -> bool:
        return len(self.times) >= 3

    def artifacts(self) -> dict:
        return {"sensor_log": "sensor.log"}

    def stages(self):
        def poll(tick):
            self.times.append(tick.t.sim_time)

        return [Stage("poll", "host", poll)]


class _AttachedController(_Lifecycle):
    """A stand-in controller attached from the start, which answers at once with zero commands. Its artifact is
    its log.
    """

    instances: ClassVar[list] = []
    attached = True

    def __init__(self, **kwargs):
        super().__init__()
        type(self).instances.append(self)

    def artifacts(self) -> dict:
        return {"controller_log": "controller.log"}

    def stages(self):
        return peer_stages(self)

    def exchange(self, t, timeout=None):
        return Controls(command=np.zeros(4))


def test_the_loop_and_sim_read_the_contracts_members_from_every_component_that_states_them(tmp_path):
    """One contract serves every role, and the loop and `Sim` read its members from every component that states
    them: connect, close, the start held for each that waits on its peer, and artifacts collected from each.

    Given a run on the fixture vehicle whose stand-in sensor and stand-in controller each state `connect`,
    `close`, `attached` and `artifacts`, when the run starts and ends, then each connected once and closed once,
    the clock held at 0 until both were attached, so the sensor's first three polls see 0 s and its fourth one
    tick, and `sim.artifacts()` holds both entries.
    """
    _WaitingSensor.instances.clear()
    _AttachedController.instances.clear()
    registry = sv.components(NexusImuAPI=_WaitingSensor, NexusPx4API=_AttachedController)
    loop = sv.build(sv.vehicle(tmp_path, sv.prim("Imu", "NexusImuAPI")), components=registry)
    with nx.Sim.from_orchestrator(loop) as sim:
        sim.start(timeout=5.0)
        artifacts = sim.artifacts()
    (sensor,), (controller,) = _WaitingSensor.instances, _AttachedController.instances

    read = (
        sensor.lifecycle,
        controller.lifecycle,
        sensor.times[:4],
        artifacts.keys() >= {"sensor_log", "controller_log"},
    )
    assert read == (["connect", "close"], ["connect", "close"], [0.0, 0.0, 0.0, DT], True)
