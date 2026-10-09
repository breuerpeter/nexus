"""A short recorded flight of a one-body vehicle on the Warp CPU backend, for the recording tests.

:func:`build` wires the run: the loop on real NVIDIA Newton physics, a Logger that writes an ``.rrd``,
and a Recorder whose staging buffer holds ``staging`` rows, so a test crosses a drain in a few ticks. A
stand-in controller commands nothing, so the body falls under gravity alone. :func:`rows` and
:func:`times` read one entity of a written ``.rrd`` through Rerun's reader.

Run as a script, ``python _flight.py <dir> <mode>``, it flies with no step cap in ``<dir>``, prints
``ready`` after 50 ticks, and steps on until a signal ends it, so a test sends it one and reads
what the ``.rrd`` kept. On a Ctrl-C it prints ``history <n>``, the rows the base body's history holds,
once the run has torn down. The ``staged`` mode gives the Recorder a staging buffer of 8 rows, and the
``run`` mode leaves the loop its own. In the ``slow-write`` mode the first write Rerun makes after
``ready`` takes two seconds and prints ``writing`` first, so a test can land a second Ctrl-C inside it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DT = 0.004  # 250 Hz control ticks
READY_TICKS = 50  # the ticks the script flies before it prints ready
# The base body's recorded position series, and its pose, as #47 places them.
POSITION = "/sim/vehicle/body/body/series/position"
POSE = "/sim/vehicle/body"


def author_usd(path: str) -> None:
    """Write a one-body vehicle, a box collider with a mass, to ``path``."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Cube.Define(stage, "/Vehicle/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.5)
    stage.GetRootLayer().Save()


class Actuator:
    """Writes no forces: the body flies under gravity alone."""

    def stages(self):
        from nexus_sim._src.core.interfaces import Stage

        return [Stage("forces", "device", lambda tick: None)]


class Controller:
    """One device stage that commands nothing. ``act`` runs each tick, so a test can make it raise."""

    def __init__(self, act=None):
        self._act = act or (lambda tick: None)

    def connect(self) -> None:
        pass

    def stages(self):
        from nexus_sim._src.core.interfaces import Stage

        return [Stage("act", "device", self._act)]

    def close(self) -> None:
        pass


def build(
    tmp,
    *,
    staging: int | None = 8,
    max_steps: int | None = None,
    controller=None,
    peers=(),
    usd: str | None = None,
    renderer=None,
):
    """Build the run in ``tmp`` and return it with the path of its ``.rrd``.

    ``staging`` is the rows the Recorder's staging buffer holds; ``None`` leaves the loop its own
    Recorder. ``controller`` replaces the stand-in, ``peers`` are the peers the loop stops, ``usd`` a
    vehicle in place of the one-body one, and ``renderer`` the loop's renderer, which it closes first
    at teardown.
    """
    import newton

    from nexus_sim._src.core.clock import Clock
    from nexus_sim._src.core.orchestrator import Orchestrator
    from nexus_sim._src.logging import Logger
    from nexus_sim._src.physics.physics import NewtonPhysics
    from nexus_sim._src.physics.vehicle import VehicleUsd
    from nexus_sim._src.recording import Recorder

    if usd is None:
        usd = str(Path(tmp) / "mini.usda")
        author_usd(usd)
    cfg = {"physics": {"dt": DT, "solver": "semi_implicit", "contacts": False, "spawn": {"pos": [0.0, 0.0, 5.0]}}}
    rrd = str(Path(tmp) / "flight.rrd")
    mb = newton.ModelBuilder()
    VehicleUsd({"usd_path": usd}).build(mb)
    model = mb.finalize()
    recorder = {} if staging is None else {"recorder": Recorder(dt=DT, staging=staging)}
    orch = Orchestrator(
        clock=Clock(DT),
        physics=NewtonPhysics(model=model, cfg=cfg),
        actuator=Actuator(),
        sensors=[],
        controller=controller or Controller(),
        logger=Logger(model, serve=False, record_to_rrd=rrd),
        renderer=renderer,
        peers=list(peers),
        max_steps=max_steps,
        **recorder,
    )
    return orch, rrd


def _chunks(rrd: str, entity: str):
    """The timed chunks of ``entity`` in the ``.rrd`` at ``rrd``, as record batches: a static row, such
    as a series' legend, isn't a row of the history.
    """
    from rerun.experimental import RrdReader

    for chunk in RrdReader(rrd).stream():
        if chunk.entity_path == entity and not chunk.is_static:
            yield chunk.to_record_batch()


def rows(rrd: str, entity: str) -> int:
    """The timed rows ``entity`` holds in the ``.rrd`` at ``rrd``, over every chunk."""
    return sum(batch.num_rows for batch in _chunks(rrd, entity))


def times(rrd: str, entity: str) -> list[float]:
    """The time of each row ``entity`` holds, in seconds, in file order."""
    import pyarrow as pa

    return [ns / 1e9 for batch in _chunks(rrd, entity) for ns in batch.column("time").cast(pa.int64()).to_pylist()]


class Flight:
    """What a flight run as a script left: its ``.rrd``, its exit code, and the rows the base body's
    history held at its Ctrl-C, ``None`` when none landed.
    """

    def __init__(self, rrd: str, returncode: int, history: int | None):
        self.rrd = rrd
        self.returncode = returncode
        self.history = history


def _lines(proc, timeout: float):
    """Yield the script's stdout lines as they come, or stop after ``timeout`` seconds of silence.

    Reads the pipe itself, in chunks, since a line reader would buffer past the line it returns and
    leave the pipe empty for the wait that follows.
    """
    import os
    import select

    fd = proc.stdout.fileno()
    pending = b""
    while True:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return
        chunk = os.read(fd, 4096)
        if not chunk:
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.decode().strip()


def _until(proc, line: str, timeout: float) -> list[str]:
    """The lines the script printed up to and including ``line``, which it must print within ``timeout``."""
    seen = []
    for got in _lines(proc, timeout):
        seen.append(got)
        if got == line:
            return seen
    raise AssertionError(f"the flight never printed {line!r}; it printed {seen}")


def run_until_ready(tmp, sig: int, *, mode: str = "run", then: int | None = None, timeout: float = 10.0) -> Flight:
    """Fly the script in ``tmp`` and send it ``sig`` once it prints ``ready``.

    ``mode`` is ``run``, a Recorder of the loop's own, ``staged``, a staging buffer of 8 rows, or
    ``slow-write``. ``then`` is a second signal, sent once the script prints ``writing``, which it does
    in the ``slow-write`` mode. The script must exit within ``timeout`` seconds of the last signal, or
    the test ends it. Returns what it left.
    """
    stderr = Path(tmp) / "flight.err"
    with stderr.open("w") as err:
        proc = subprocess.Popen(
            [sys.executable, __file__, str(tmp), mode], stdout=subprocess.PIPE, stderr=err, text=True
        )
    lines: list[str] = []
    try:
        try:
            lines += _until(proc, "ready", 120.0)
        except AssertionError as exc:
            raise AssertionError(f"{exc}; its stderr ends with: {stderr.read_text()[-2000:]}") from None
        proc.send_signal(sig)
        if then is not None:
            lines += _until(proc, "writing", timeout)
            proc.send_signal(then)
        out, _ = proc.communicate(timeout=timeout)
        lines += out.splitlines()
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AssertionError(f"the flight did not exit within {timeout} s of its signal") from None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    words = " ".join(lines).split()
    history = [int(words[i + 1]) for i, word in enumerate(words) if word == "history"]
    return Flight(str(Path(tmp) / "flight.rrd"), proc.returncode, history[0] if history else None)


def _slow_first_write_after(stopping: dict) -> None:
    """Make the first columnar write Rerun takes after ``stopping`` holds ``now`` print ``writing`` and take 2 s."""
    import rerun as rr

    send = rr.send_columns

    def slow(*args, **kwargs):
        if stopping.pop("now", False):
            print("writing", flush=True)
            time.sleep(2.0)
        return send(*args, **kwargs)

    rr.send_columns = slow


def _fly(tmp: str, mode: str) -> None:
    import warp as wp

    import nexus_sim as nx

    stopping: dict = {}
    if mode == "slow-write":
        _slow_first_write_after(stopping)
    with wp.ScopedDevice("cpu"):
        orch, _ = build(tmp, staging=8 if mode == "staged" else None)
        try:
            with nx.Sim.from_orchestrator(orch) as sim:
                try:
                    n = 0
                    while sim.step():
                        n += 1
                        if n == READY_TICKS:
                            stopping["now"] = True  # the next write Rerun takes is the teardown's
                            print("ready", flush=True)
                except KeyboardInterrupt:  # the run has torn down on its way here
                    print(f"history {len(sim.physics[sim.base_body].history())}", flush=True)
                    raise
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    _fly(sys.argv[1], sys.argv[2])
