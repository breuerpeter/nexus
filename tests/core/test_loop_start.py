"""The start of a run and its first tick, #229: the start runs the ring without the clock's stage and the
physics substeps once over the settled state at sim time zero, and each tick's physics step opens it, so
every stage of a tick reads the state that step left. Stand-in components at the loop's roles, on the Warp
CPU device, where every stage runs as plain Python; each keeps, in one journal shared by the run, its role,
the step index of the tick it ran on and the position the plant held when it ran.
"""

import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.clock import Clock
from nexus_sim._src.core.interfaces import Component, Stage
from nexus_sim._src.core.loop import Loop
from nexus_sim._src.core.schema import PoseTwist
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.recording import Recorder

DT = 0.004
# The plant's position: settled, then after each physics step. It falls 1 m a step.
PATH = [(0.0, 0.0, 10.0), (0.0, 0.0, 9.0), (0.0, 0.0, 8.0), (0.0, 0.0, 7.0), (0.0, 0.0, 6.0)]
ROLES = ("sensor", "estimator", "guidance", "controller")


class _State:
    """The physics state: one body, at the position the path holds."""

    def __init__(self, pos):
        self.pos = pos


class _Physics:
    """Physics whose ``step`` moves the body to the next point of `PATH`, and holds the last."""

    def __init__(self):
        self._steps = 0
        self.state = _State(PATH[0])

    def reset(self):
        return self.state

    def step(self, state):
        self._steps += 1
        state.pos = PATH[min(self._steps, len(PATH) - 1)]

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: None),
            Stage("step", "device", lambda tick: self.step(tick.state)),
            Stage("state", "device", lambda tick: None),
        ]


def _keep(journal: list, role: str, tick) -> None:
    journal.append((role, tick.t.step_index, tuple(tick.state.pos)))


class _Actuator(Component):
    def stages(self):
        return [Stage("forces", "device", lambda tick: None)]


class _Sensor(Component):
    def __init__(self, journal):
        self._journal = journal

    def stages(self):
        return [Stage("sample", "device", lambda tick: _keep(self._journal, "sensor", tick))]


class _Estimator(Component):
    """An estimator whose stage writes the plant's position to the estimate."""

    def __init__(self, journal):
        self._journal = journal
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))

    def stages(self):
        def estimate(tick):
            _keep(self._journal, "estimator", tick)
            self.estimate.write([(tick.state.pos, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])

        return [Stage("estimate", "device", estimate, writes=(self.estimate,))]


class _Guidance(Component):
    """A guidance whose device stage reads the estimate and writes a setpoint."""

    def __init__(self, journal):
        self._journal = journal
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))

    def stages(self):
        return [
            Stage(
                "guidance",
                "device",
                lambda tick: _keep(self._journal, "guidance", tick),
                reads=(self.estimate,),
                writes=(self.setpoint,),
            )
        ]


class _Controller(Component):
    """A controller with no peer, whose one stage is a host stage or a device stage, as `kind` says. It reads
    the setpoint when `guided`.
    """

    def __init__(self, journal, kind: str, *, guided: bool = True):
        self._journal = journal
        self._kind = kind
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))
        self._reads = (self.setpoint,) if guided else ()

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        def act(tick):
            _keep(self._journal, "controller", tick)
            return True

        return [Stage("act", self._kind, act, reads=self._reads)]


class _LatePeerController(Component):
    """A controller whose peer dials in on the third poll of its host stage and answers the first message it gets."""

    def __init__(self):
        self._polls = 0

    @property
    def attached(self):
        return self._polls >= 3

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        def exchange(tick):
            reached = self.attached  # a message sent before the peer dials in reaches nobody
            self._polls += 1
            return reached

        return [Stage("exchange", "host", exchange)]


def _loop(journal, controller, *, guided: bool = True, recorder=None) -> tuple[Loop, Clock]:
    clock = Clock(DT)
    loop = Loop(
        clock=clock,
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=[_Sensor(journal)],
        estimator=_Estimator(journal),
        guidance=_Guidance(journal) if guided else None,
        controller=controller,
        recorder=recorder,
    )
    return loop, clock


def test_the_start_runs_each_stage_once_in_ring_order_at_sim_time_zero_over_the_settled_state():
    """The start runs the ring without the clock's stage and the physics substeps, once, in ring order, at sim
    time zero, over the settled state.

    Given stand-ins of a sensor, an estimator, a guidance and a controller with a host stage, each logging its
    stage's runs with the tick's sim time and the position it read, when the run takes its first tick, then
    before that tick each ran once, in ring order, at `0.0` and at the settled position, and the record stage
    wrote one row at `0.0`.
    """
    journal = []
    recorder = Recorder()
    with wp.ScopedDevice("cpu"):
        loop, _ = _loop(journal, _Controller(journal, "host"), recorder=recorder)
        loop.step()
        loop.close()
    rows = next(h for key, h in recorder.histories.items() if key.endswith("/estimate")).arrays()["t"]

    assert ([entry for entry in journal if entry[1] == 0], rows.tolist()) == (
        [(role, 0, PATH[0]) for role in ROLES],
        [0.0, 0.004],
    )


@pytest.mark.parametrize("controller", ["host", "device", "late peer"])
def test_the_first_tick_is_step_one_for_every_controller(controller):
    """The first tick is step one for every controller.

    Given a run at a 4 ms tick, once with a controller that has a host stage, once with one whose stages are all
    device stages and once with one whose peer attaches late, when each run takes its first tick, then that
    tick's sim time is `0.004` and its step index 1.
    """
    journal = []
    with wp.ScopedDevice("cpu"):
        if controller == "late peer":
            stand_in = _LatePeerController()
        else:
            stand_in = _Controller(journal, controller, guided=False)
        loop, clock = _loop(journal, stand_in, guided=False)
        loop.step()
        loop.close()

    assert (clock.now().sim_time, clock.now().step_index) == (0.004, 1)


@pytest.mark.parametrize("controller", ["host", "device"])
def test_each_ticks_stages_read_the_state_that_ticks_physics_step_left(controller):
    """Each tick's sensors, estimator, guidance and controller read the state that tick's physics step left.

    Given a falling plant and stand-ins of each role that log the position they read, once on a ring of device
    stages only and once with a controller host stage, when the run steps 3 ticks, then on each tick every role
    read the position the plant holds after that tick's physics step.
    """
    journal = []
    with wp.ScopedDevice("cpu"):
        loop, _ = _loop(journal, _Controller(journal, controller))
        for _ in range(3):
            loop.step()
        loop.close()

    assert [entry for entry in journal if entry[1] >= 1] == [(role, k, PATH[k]) for k in (1, 2, 3) for role in ROLES]
