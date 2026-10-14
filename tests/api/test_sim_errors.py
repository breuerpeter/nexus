"""A script separates an error from a failure by one public exception, ``nexus_uav.RunError``, #321.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, through ``Sim``,
with the PX4 Software In The Loop (SITL) peer sent to a stand-in of the fake PX4: one that never dials in, one that
exits during preroll, or one a test stops past lockstep. Each stand-in names its own error lines through ``errors``.
Skipped without newton or pxr.
"""

import time
from typing import ClassVar

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_uav as nx
from nexus_uav._src.peers.kit.fake import KitFake
from nexus_uav._src.peers.kit.runner import KitPeerError
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from tests.peers.kit.conftest import daemon  # noqa: F401
from tests.recording._flight import rows
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.usefixtures("warp_cpu"), pytest.mark.xdist_group("px4_ports")]

# The line PX4 prints when its tree lacks the vehicle's airframe, as a stand-in's own error line.
UNKNOWN_MODEL = "ERROR [init] Unknown model none_astro_max"
# The base body's position history in the recording.
POSITION = "/sim/vehicle/body/body/position"


def _raised(call) -> BaseException | None:
    """The exception `call` raises, or `None` when it returns."""
    try:
        call()
    except Exception as exc:  # the test reads which error the call raised
        return exc
    return None


def _caught_as_run_error(exc: BaseException | None) -> bool:
    """Whether `except nexus_uav.RunError` catches `exc`."""
    return exc is not None and isinstance(exc, getattr(nx, "RunError", ()))


class _Console(Px4Fake):
    """A stand-in for PX4 whose console names `UNKNOWN_MODEL` among other lines, by its path and by its error lines."""

    def artifacts(self) -> dict:
        return {"px4_log": "/runs/px4.log"}

    def errors(self) -> list[str]:
        return [UNKNOWN_MODEL]


class _Silent(_Console):
    """A PX4 that boots but never dials in: it stays up and sends nothing."""

    def start(self) -> None:
        pass

    def alive(self) -> bool:
        return True


class _Exited(_Console):
    """A PX4 that exits during its boot, before it dials in."""

    def start(self) -> None:
        pass

    def alive(self) -> bool:
        return False


class _Lost(_Console):
    """The fake PX4, each started instance kept in `started`, so a test stops it past lockstep."""

    started: ClassVar[list] = []

    def start(self) -> None:
        type(self).started.append(self)
        super().start()


def _sim(tmp_path, peer: type, **kw) -> nx.Sim:
    """A run of the fixture vehicle on the PX4 controller, with the PX4 SITL peer sent to `peer`."""
    vehicle = sv.vehicle(tmp_path, px4=True)
    return nx.Sim(vehicle, scene=sv.SCENE, device="cpu", registry=nx.Registry({"NexusPx4SitlAPI": peer}), **kw)


def _lost_mid_run(sim: nx.Sim) -> None:
    """Start `sim` to lockstep, then stop its fake PX4, as a PX4 that exits mid-flight."""
    sim.start(timeout=10.0)
    _Lost.started[-1].stop()


def test_a_run_whose_peer_never_answers_raises_the_public_run_error_from_start_and_tears_down(tmp_path, monkeypatch):
    """A run whose peer never answers raises `nexus_uav.RunError` from `Sim.start()`, and the run's teardown has run.

    Given the fixture vehicle on a fake PX4 that never dials in and a recorded run, when `Sim.start(timeout=0.2)` runs
    under `except nexus_uav.RunError`, then the handler catches it and the run has closed the recording: its file holds
    the base body's rows.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    with _sim(tmp_path, _Silent, log=True) as sim:
        error = _raised(lambda: sim.start(timeout=0.2))
        rrd = sim.artifacts()["rrd"]

    assert (_caught_as_run_error(error), rows(rrd, POSITION) > 0) == (True, True), error


def test_the_run_errors_message_carries_the_peers_own_error_lines_and_its_logs_path(tmp_path):
    """The run error's message carries the peer's own error lines, not only its log's path.

    Given a fake PX4 that never dials in and whose error lines hold PX4's `Unknown model` line, when
    `Sim.start(timeout=0.2)` fails, then the message holds that line and the log's path.
    """
    with _sim(tmp_path, _Silent) as sim:
        error = _raised(lambda: sim.start(timeout=0.2))

    assert (UNKNOWN_MODEL in str(error), "/runs/px4.log" in str(error)) == (True, True), error


def test_a_peer_that_exits_during_preroll_fails_the_run_at_once(tmp_path):
    """A peer whose process exits during preroll fails the run at once, not after `preroll_timeout`.

    Given a fake PX4 that exits during preroll with an error line, when `Sim.start(timeout=60)` runs, then it raises
    `RunError` holding that line within 5 s of wall time.
    """
    with _sim(tmp_path, _Exited) as sim:
        t0 = time.monotonic()
        error = _raised(lambda: sim.start(timeout=60.0))
        took = time.monotonic() - t0

    assert (_caught_as_run_error(error), UNKNOWN_MODEL in str(error), took < 5.0) == (True, True, True), (
        error,
        took,
    )


class _KitFailsToStart(KitFake):
    """A Kit render peer whose container fails to start, as the Kit runner reports it."""

    def start(self) -> None:
        raise KitPeerError("starting the Kit container failed: no such image")


def test_a_run_whose_kit_render_peer_fails_to_start_raises_the_public_run_error(tmp_path, daemon):  # noqa: F811
    """A run whose Kit render peer fails to start raises `nexus_uav.RunError`.

    Given the fixture vehicle with a camera and a Kit fake whose start fails, when the run starts under
    `except nexus_uav.RunError`, then the handler catches it.
    """
    camera = sv.prim("Cam", "NexusCameraAPI", kind="Camera")
    vehicle = sv.vehicle(tmp_path, camera)
    registry = sv.components(kit=_KitFailsToStart)

    def start():
        with nx.Sim(vehicle, scene=sv.SCENE, device="cpu", registry=registry) as sim:
            sim.start(timeout=5.0)

    error = _raised(start)

    assert _caught_as_run_error(error), error


def test_a_peer_lost_mid_run_raises_the_run_error_from_step_and_tears_down(tmp_path, monkeypatch):
    """A peer lost mid-run raises `RunError` from `step()`.

    Given the fixture vehicle on the fake PX4 past lockstep, when the fake stops and the script steps, then a
    `step()` raises `RunError` naming the peer, and the run has closed the recording: its file holds the base
    body's rows.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    with _sim(tmp_path, _Lost, log=True) as sim:
        _lost_mid_run(sim)

        def step_on():
            while sim.step():
                pass

        error = _raised(step_on)
        rrd = sim.artifacts()["rrd"]

    named = "px4" in str(error).lower()
    assert (_caught_as_run_error(error), named, rows(rrd, POSITION) > 0) == (True, True, True), error


def test_a_peer_lost_mid_run_raises_the_run_error_from_run(tmp_path):
    """A peer lost mid-run raises `RunError` from `run()`.

    Given the fixture vehicle on the fake PX4 past lockstep, when the fake stops and the script calls `sim.run()`,
    then `run()` raises `RunError` naming the peer.
    """
    with _sim(tmp_path, _Lost) as sim:
        _lost_mid_run(sim)
        error = _raised(sim.run)

    assert (_caught_as_run_error(error), "px4" in str(error).lower()) == (True, True), error


def test_a_peer_lost_mid_run_raises_the_run_error_from_wait_until_not_a_timeout(tmp_path):
    """A peer lost mid-run raises `RunError` from `wait_until`, not `TimeoutError`.

    Given the fixture vehicle on the fake PX4 past lockstep, when the fake stops during
    `sim.wait_until(lambda: False, 60.0)`, then it raises `RunError`, not `TimeoutError`.
    """
    with _sim(tmp_path, _Lost) as sim:
        _lost_mid_run(sim)
        error = _raised(lambda: sim.wait_until(lambda: False, 60.0))

    assert (_caught_as_run_error(error), isinstance(error, TimeoutError)) == (True, False), error


def test_a_peer_lost_mid_run_raises_the_run_error_from_sleep_not_a_quiet_return(tmp_path):
    """A peer lost mid-run raises `RunError` from `sleep`, not a quiet return.

    Given the fixture vehicle on the fake PX4 past lockstep, when the fake stops during `sim.sleep(60.0)`, then it
    raises `RunError`.
    """
    with _sim(tmp_path, _Lost) as sim:
        _lost_mid_run(sim)
        error = _raised(lambda: sim.sleep(60.0))

    assert _caught_as_run_error(error), error


def test_the_mid_run_errors_message_carries_the_peers_own_error_lines(tmp_path):
    """The mid-run error's message carries the peer's own error lines.

    Given a fake PX4 that stops answering past lockstep and whose error lines hold an error line, when the script
    steps, then the `RunError` message holds that line.
    """
    with _sim(tmp_path, _Lost) as sim:
        _lost_mid_run(sim)

        def step_on():
            while sim.step():
                pass

        error = _raised(step_on)

    assert UNKNOWN_MODEL in str(error), error


def test_a_condition_not_met_in_time_on_a_live_run_raises_a_timeout(tmp_path):
    """A condition not met in time on a live run still raises `TimeoutError`.

    Given the fixture vehicle on the fake PX4, when `sim.wait_until(lambda: False, 0.5)` runs, then it raises
    `TimeoutError` once 0.5 s of sim time has passed.
    """
    with _sim(tmp_path, Px4Fake) as sim:
        sim.start(timeout=10.0)
        t0 = sim.time
        error = _raised(lambda: sim.wait_until(lambda: False, 0.5))
        waited = sim.time - t0

    assert (type(error), waited >= 0.5) == (TimeoutError, True), (error, waited)


def _ended(tmp_path) -> nx.Sim:
    """A run of the fixture vehicle on the stand-in controller that ends after 10 ticks, by `max_steps`."""
    return nx.Sim(sv.vehicle(tmp_path), scene=sv.SCENE, device="cpu", registry=sv.components(), max_steps=10)


def test_a_sleep_past_the_end_of_a_run_by_max_steps_returns(tmp_path):
    """A run that ends by `max_steps` still ends with no error: `sleep` returns.

    Given the fixture vehicle on stand-ins with `max_steps=10`, when the script calls `sleep(60.0)`, then it returns.
    """
    with _ended(tmp_path) as sim:
        error = _raised(lambda: sim.sleep(60.0))

    assert error is None, error


def test_the_step_past_the_end_of_a_run_by_max_steps_returns_false(tmp_path):
    """A run that ends by `max_steps` still ends with no error: the step past the end returns `False`.

    Given the fixture vehicle on stand-ins with `max_steps=10`, when the script steps 11 times, then the first ten
    return `True` and the eleventh `False`.
    """
    with _ended(tmp_path) as sim:
        stepped = [sim.step() for _ in range(11)]

    assert stepped == [True] * 10 + [False]


def test_a_wait_past_the_end_of_a_run_by_max_steps_raises_a_timeout(tmp_path):
    """A run that ends by `max_steps` still ends with no error of its own: `wait_until` raises `TimeoutError`.

    Given the fixture vehicle on stand-ins with `max_steps=10`, when the script calls `wait_until(lambda: False,
    60.0)`, then it raises `TimeoutError`.
    """
    with _ended(tmp_path) as sim:
        error = _raised(lambda: sim.wait_until(lambda: False, 60.0))

    assert type(error) is TimeoutError, error
