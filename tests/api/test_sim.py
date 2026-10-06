"""Sim lifecycle: drives the orchestrator on the CALLER's thread, proxies observation, tears down.

Uses a fake orchestrator, a monkeypatched build_from_launch, so the test needs no real
assets / Newton physics / PX4: it exercises the Sim wiring, not the sim.
"""

import threading

import numpy as np
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
        self._rec = recorder.channel("vehicle/body/body_frd", width=BODY_WIDTH, decode=decode_body)

    def step(self):
        self.threads.append(threading.current_thread())
        if self._stop or self.steps >= self.STEPS:
            return False
        self.steps += 1
        if self._rec is not None:  # the physics tap snapshots the airframe, as the real loop does per tick
            device = self._rec.buf.device  # record where the channel lives, whatever the default device is now
            v = _FakeView(device)
            wp.launch(
                record_body,
                dim=1,
                inputs=(v.body_q, v.body_qd, 0, self._rec.dt, self._rec.maxlen, self._rec.buf, self._rec.counter),
                device=device,
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
    def __init__(self, device=None):
        # Device arrays the physics tap reads: body_q = [p, q-xyzw], body_qd = [lin(0:3), ang(3:6)].
        q = np.array([[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.body_q = wp.array(q, dtype=wp.transform, device=device)
        self.body_qd = wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.spatial_vector, device=device)

    def positions(self):
        return np.array([[0.0, 0.0, 4.0]])

    def orientations(self):
        return np.array([[0.0, 0.0, 0.0, 1.0]])

    def linear_velocities(self):
        return np.zeros((1, 3))

    def angular_velocities(self):
        return np.zeros((1, 3))


def test_sim_start_drives_setup_observes_and_stops(monkeypatch):
    fake = _FakeOrch()
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)

    with sim_mod.Sim("astro_max_base", scene="empty", device="cpu") as sim:
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
        with sim_mod.Sim("astro_max_base", scene="empty") as sim:
            sim.start(timeout=30.0)


def test_sim_px4_spawns_no_sim_thread(monkeypatch):
    """No more daemon: driving PX4 must not start a `newton-sim` thread at any point.

    That's what keeps the Kit thread rule, issue #40, true by construction: nothing left in `Sim` can
    reach `RtxFrame` from off the thread that built it.
    """
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: _FakeOrch())
    assert not any(t.name == "newton-sim" for t in threading.enumerate())
    with sim_mod.Sim("astro_max_base", scene="empty", device="cpu") as sim:
        sim.start(timeout=30.0)
        assert not any(t.name == "newton-sim" for t in threading.enumerate())
    assert not any(t.name == "newton-sim" for t in threading.enumerate())


def test_sim_passes_device_into_launch(monkeypatch):
    captured = {}

    def _cap(launch, **kw):
        captured["device"] = launch.runtime.device
        return _FakeOrch()

    monkeypatch.setattr(sim_mod, "build_from_launch", _cap)
    with sim_mod.Sim("astro_max_base", scene="empty", device="cuda") as sim:
        sim.start(timeout=30.0)
    assert captured["device"] == "cuda"


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

    def add_loggable(self, component, path):  # the orchestrator's loggable-registration seam, a no-op in the mock
        pass

    def attach_recorder(self, recorder):  # the observation seam: register the airframe channel Sim caches
        recorder.channel("vehicle/body/body_frd", width=BODY_WIDTH, decode=decode_body)

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
    with sim_mod.Sim("astro_max_base", scene="empty", device="cpu") as sim:
        assert sim.step() is True  # a host-boundary sim steps the same as any other
        assert sim.step() is True
        assert fake.threads == [threading.main_thread(), threading.main_thread()]


def test_a_run_whose_controller_takes_no_setpoint_has_no_operator_and_the_error_names_the_port_map(monkeypatch):
    """A run whose controller takes no setpoint, as PX4's does, has no controller surface and no
    operator: the autopilot owns its mission in its own process, and a script commands it over a
    link it opens itself.

    Given a `Sim` over a controller with no setpoint surface, when a script reads `sim.controller`
    and `sim.operator`, then the controller is `None` and the operator raises `RuntimeError` that
    names `sim.ports`.
    """
    fake = _FakeOrch()
    monkeypatch.setattr(sim_mod, "build_from_launch", lambda launch, **kw: fake)

    with sim_mod.Sim("astro_max_base", scene="empty", device="cpu") as sim:
        sim.start(timeout=30.0)
        controller = sim.controller
        with pytest.raises(RuntimeError, match=r"no operator.*sim\.ports"):
            _ = sim.operator

    assert controller is None


def test_sim_takes_no_control_argument():
    """`Sim` takes no control argument: the vehicle's Universal Scene Description (USD) file declares its controller.

    Given `Sim("astro_max_base", scene="empty", control="px4-sitl")`, when constructed, then it raises `TypeError`.
    """
    with pytest.raises(TypeError, match="control"):
        sim_mod.Sim("astro_max_base", scene="empty", control="px4-sitl")


def test_a_sim_that_names_no_vehicle_fails_at_construction():
    """A `Sim` that names no vehicle fails at construction, and the error names the missing argument.

    Given the bundled catalog, when a caller builds `Sim(scene="empty")`, then it raises an error that names
    the missing `vehicle`.
    """
    with pytest.raises((TypeError, ValueError), match="vehicle"):
        sim_mod.Sim(scene="empty")


def test_a_sim_that_names_no_scene_fails_at_construction():
    """A `Sim` that names no scene fails at construction, and the error names the missing argument.

    Given the bundled catalog, when a caller builds `Sim(vehicle="astro_max_base")`, then it raises an error
    that names the missing `scene`.
    """
    with pytest.raises((TypeError, ValueError), match="scene"):
        sim_mod.Sim(vehicle="astro_max_base")
