"""Sim lifecycle: drives the orchestrator on the CALLER's thread, proxies observation, tears down.

Uses a fake orchestrator, a monkeypatched build_from_launch, so the test needs no real
assets / Newton physics / PX4: it exercises the Sim wiring, not the sim.
"""

import threading

import pytest
import warp as wp

import nexus._src.api.sim as sim_mod
from nexus._src.recording import BodyState
from nexus._src.recording.state import BODY_WIDTH, decode_body, record_body


class _FakePhysics:
    base_body = "body_frd"  # the discovered airframe label Sim reads for the default vehicle entity


class _FakeOrch:
    """A step-driven orchestrator stand-in, the shape ``Orchestrator.step()`` presents: setup runs
    inside the first ``step()``, each call advances one tick and records the airframe row, and
    ``run()`` is ``while self.step(): pass``. Every ``step()`` records the thread it ran on.
    """

    STEPS = 3  # ticks before the run reports it has ended

    def __init__(self):
        self.sensors = []
        self._stop = False
        self.physics = _FakePhysics()
        self.preroll_timeout = 2.0  # Sim.start(timeout=) overrides this before the first step
        self._rec = None
        self.steps = 0
        self.threads = []  # the thread each step() ran on, the affinity this issue is about
        self.closed = False
        self.logs_closed = False

    def attach_recorder(self, recorder):
        # mimic the orchestrator handing physics the Recorder: register the airframe body channel
        self._rec = recorder.channel("physics/body/body_frd", width=BODY_WIDTH, decode=decode_body)

    def step(self):
        self.threads.append(threading.current_thread())
        if self._stop or self.steps >= self.STEPS:
            return False
        self.steps += 1
        if self._rec is not None:  # the physics tap snapshots the airframe, as the real loop does per tick
            v = _FakeView()
            wp.launch(
                record_body,
                dim=1,
                inputs=(v.body_q, v.body_qd, 0, self._rec.dt, self._rec.maxlen, self._rec.buf, self._rec.counter),
            )
            wp.synchronize()
        return True

    def run(self):
        while self.step():
            pass

    def close(self):  # the step-driven teardown seam, which closes the tick generator
        self.closed = True

    def _close_logs(self):  # the logging-teardown seam for a Sim entered but never driven
        self.logs_closed = True

    def stop(self):
        self._stop = True


class _T:
    sim_time = 0.123


class _FakeView:
    import numpy as _np
    import warp as _wp

    # Device arrays the physics tap reads: body_q = [p, q-xyzw], body_qd = [lin(0:3), ang(3:6)].
    body_q = _wp.array(_np.array([[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0]], dtype=_np.float32), dtype=_wp.transform)
    body_qd = _wp.array(_np.zeros((1, 6), dtype=_np.float32), dtype=_wp.spatial_vector)

    def positions(self):
        return self._np.array([[0.0, 0.0, 4.0]])

    def orientations(self):
        return self._np.array([[0.0, 0.0, 0.0, 1.0]])

    def linear_velocities(self):
        return self._np.zeros((1, 3))

    def angular_velocities(self):
        return self._np.zeros((1, 3))


def test_sim_start_drives_setup_observes_and_stops(monkeypatch):
    fake = _FakeOrch()
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)

    with sim_mod.Sim("astro_max_base", control="px4-sitl", device="cpu") as sim:
        sim.start(timeout=30.0)
        assert fake.preroll_timeout == 30.0  # start(timeout=) caps the wait for the peer
        assert fake.steps == 1  # start() returns after one full control tick
        assert fake.threads == [threading.main_thread()]  # driven here, not on a sim thread
        assert fake._rec is not None  # observe=True attached a Recorder + physics channel
        st = sim.physics[sim.base_body].latest()  # name-keyed: the discovered airframe entity
        assert isinstance(st, BodyState)
        assert st.altitude_m == 4.0
    # __exit__ -> stop() ended the run and closed the tick generator
    assert fake._stop
    assert fake.closed


def test_sim_start_raises_if_lockstep_never_established(monkeypatch):
    class _NeverLockstep(_FakeOrch):
        def step(self):
            # The real _ticks() catches the preroll ConnectionError, runs its teardown and stops:
            # the first step() reports the run has already ended: PX4 never connected on :4560.
            self.threads.append(threading.current_thread())
            return False

    fake = _NeverLockstep()
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)
    with pytest.raises(RuntimeError):
        with sim_mod.Sim("astro_max_base") as sim:
            sim.start(timeout=30.0)


def test_sim_px4_spawns_no_sim_thread(monkeypatch):
    """No more daemon: driving PX4 must not start a `newton-sim` thread at any point.

    That's what keeps the Kit thread rule, issue #40, true by construction: nothing left in `Sim` can
    reach `RtxFrame` from off the thread that built it.
    """
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: _FakeOrch())
    assert not any(t.name == "newton-sim" for t in threading.enumerate())
    with sim_mod.Sim("astro_max_base", control="px4-sitl", device="cpu") as sim:
        sim.start(timeout=30.0)
        assert not any(t.name == "newton-sim" for t in threading.enumerate())
    assert not any(t.name == "newton-sim" for t in threading.enumerate())


def test_sim_passes_device_into_launch(monkeypatch):
    captured = {}

    def _cap(launch, **kw):
        captured["device"] = launch.runtime.device
        return _FakeOrch()

    monkeypatch.setattr(sim_mod, "build_from_launch", _cap)
    with sim_mod.Sim("astro_max_base", device="cuda:0") as sim:
        sim.start(timeout=30.0)
    assert captured["device"] == "cuda:0"


class _FakeController:
    def __init__(self):
        self.setpoints = []

    def accept_setpoint(self, sp):
        self.setpoints.append(sp)


class _InProcessOrch:
    """An in-process orchestrator, whose controller has accept_setpoint: run() is synchronous, fires on_tick."""

    def __init__(self):
        self.sensors = []
        self.controller = _FakeController()
        self.physics = _FakePhysics()
        self.logger = None
        self.on_tick = None
        self.run_stats = {}
        self.ran = False
        self._stop = False

    def _close_logs(self):  # the orchestrator's logging-teardown seam, a no-op: this mock has no Logger
        pass

    def add_loggable(self, component):  # the orchestrator's loggable-registration seam, a no-op in the mock
        pass

    def attach_recorder(self, recorder):  # the observation seam: register the airframe channel Sim caches
        recorder.channel("physics/body/body_frd", width=BODY_WIDTH, decode=decode_body)

    def close(self):  # the step/run teardown seam, a no-op in this mock
        pass

    def run(self):
        self.ran = True
        # one host-seam tick at the goal, so the operator advances/ends, then stamp run_stats
        if self.on_tick is not None:
            self.on_tick(_FakeView(), _T(), 1)
        self.run_stats = {"control_steps": 1, "rtf": 1.0}

    def step(self):  # advance 3 ticks then report the run ended, mirroring the generator's StopIteration
        self._steps = getattr(self, "_steps", 0) + 1
        return self._steps < 3

    def stop(self):
        self._stop = True


def test_sim_in_process_wires_operator_controller_and_runs():
    from nexus._src.operator import InProcessOperator

    fake = _InProcessOrch()

    # A self-assembled in-process orchestrator enters via from_orchestrator, the examples' entry.
    with sim_mod.Sim.from_orchestrator(fake) as sim:
        assert isinstance(sim.operator, InProcessOperator)  # operator constructed over the controller
        assert sim.controller is fake.controller  # thin surface, which has accept_setpoint
        assert fake.on_tick is not None  # operator wired to the host seam
        sim.operator.set_mission([(0.0, 0.0, 4.0)])  # the fake view sits at z=4 → reached immediately
        assert fake.controller.setpoints  # accept_setpoint commanded the first goal
        sim.run()
        assert fake.ran  # ran synchronously, no thread
        assert sim.results() == {"control_steps": 1, "rtf": 1.0}
    assert fake._stop  # __exit__ -> stop()


def test_sim_step_drives_inprocess_tick_by_tick():
    fake = _InProcessOrch()
    with sim_mod.Sim.from_orchestrator(fake) as sim:
        assert sim.step() is True  # advances one tick, delegating to orch.step()
        assert sim.step() is True
        assert sim.step() is False  # the run ended, where the generator would StopIteration
    assert fake._stop  # __exit__ -> stop(), which closes the tick generator


def test_sim_step_drives_px4_on_the_calling_thread(monkeypatch):
    fake = _FakeOrch()
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)
    with sim_mod.Sim("astro_max_base", control="px4-sitl", device="cpu") as sim:
        assert sim.step() is True  # a host-boundary sim steps the same as any other
        assert sim.step() is True
        assert fake.threads == [threading.main_thread(), threading.main_thread()]


class _FakeOffboard:
    """Px4Offboard stand-in: `open()` is non-blocking and `connected` only turns True after the sim
    has stepped `CONNECT_STEPS` times, standing in for PX4, whose clock is the sim's under lockstep.
    """

    CONNECT_STEPS = 3

    def __init__(self, orch):
        self.orch = orch  # the fake orchestrator whose step count gates `connected`
        self.opened = self.closed = False

    def open(self):
        self.opened = True
        return self

    @property
    def connected(self):
        return self.orch.steps >= self.CONNECT_STEPS

    def close(self):
        self.closed = True


def test_sim_px4_controller_is_none_and_operator_is_px4offboard(monkeypatch):
    fake = _FakeOrch()
    fake.STEPS = 99  # let it keep stepping while the link connects
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)

    import nexus._src.operator as op_mod

    # The stub needs no live PX4 on :14540, see _FakeOffboard.
    monkeypatch.setattr(op_mod, "Px4Offboard", lambda *a, **k: _FakeOffboard(fake), raising=False)
    with sim_mod.Sim("astro_max_base", control="px4-sitl") as sim:
        sim.start(timeout=30.0)
        assert sim.controller is None  # PX4 owns its mission externally, so the thin surface is None
        op = sim.operator  # PX4: lazily constructs + connects a Px4Offboard on :14540
        assert isinstance(op, _FakeOffboard) and op.opened
        assert sim.operator is op  # cached
    assert op.closed  # closed on sim teardown


def test_sim_operator_steps_the_sim_while_the_px4_link_connects(monkeypatch):
    """*The* regression the isaacsim cell caught: PX4's clock is the sim's under lockstep, so the
    operator's connect has to step the sim. A caller that merely slept here would stop the sim and
    PX4 would never send the heartbeat the caller waits for.
    """
    fake = _FakeOrch()
    fake.STEPS = 99
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)

    import nexus._src.operator as op_mod

    monkeypatch.setattr(op_mod, "Px4Offboard", lambda *a, **k: _FakeOffboard(fake), raising=False)
    with sim_mod.Sim("astro_max_base", control="px4-sitl", device="cpu") as sim:
        sim.start(timeout=30.0)
        assert fake.steps == 1  # start() flew exactly one tick; the link isn't up yet
        op = sim.operator
        assert op.connected
        assert fake.steps == _FakeOffboard.CONNECT_STEPS  # the property stepped until PX4 answered
        assert fake.threads[-1] is threading.main_thread()  # …on the caller's thread, no daemon


def test_sim_operator_raises_if_the_run_ends_before_px4_answers(monkeypatch):
    fake = _FakeOrch()  # the default of 3 steps, and the stub never connects
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)

    class _NeverConnects(_FakeOffboard):
        CONNECT_STEPS = 10_000

    import nexus._src.operator as op_mod

    monkeypatch.setattr(op_mod, "Px4Offboard", lambda *a, **k: _NeverConnects(fake), raising=False)
    with sim_mod.Sim("astro_max_base", control="px4-sitl", device="cpu") as sim:
        sim.start(timeout=30.0)
        with pytest.raises(RuntimeError, match="operator link"):
            _ = sim.operator
