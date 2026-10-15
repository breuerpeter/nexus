"""The parameters a PX4 Software In The Loop (SITL) run boots with: its airframe's, and none an earlier run set.

PX4 saves a parameter set at run time in its instance's folder of the tree. Each test flies an earlier run in a
child process that sets ``COM_DISARM_PRFLT`` to 0 over the offboard link, then a run of its own through
``nx.Sim`` that reads the parameter back, so the tests need docker and a PX4 tree carrying the airframe the sim
flies; they skip without one.
"""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does: whichever import runs
# first sets it.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus_uav as nx
from nexus_uav._src.peers.px4_sitl import checkout

# The tree the run would fly, with no fetch: the checkout the pin names, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
# A mark, not a skip at import, so a run that deselects `px4_sitl` lists none of these tests.
# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [
    pytest.mark.px4_sitl,
    pytest.mark.xdist_group("px4_ports"),
    pytest.mark.skipif(
        _TREE is None
        or not list((_TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes").glob("*_none_astro_max")),
        reason="needs docker and a PX4 tree on this machine, the pinned checkout or commit, carrying the none_astro_max airframe",
    ),
]

_PARAM = "COM_DISARM_PRFLT"
_AIRFRAME_VALUE = 10.0  # the parameter's default, which neither the airframe nor its base changes

# The earlier run: it sets the parameter, flies on while PX4 saves it, and prints its instance's system id.
# With `hold`, it then waits until the test ends it with `SIGKILL`, so the run never reaches its teardown.
_EARLIER_RUN = textwrap.dedent(
    f"""
    import sys, time
    import nexus_uav as nx
    from nexus_uav.px4 import OffboardClient

    with nx.Sim("astro_max_base", scene="empty") as sim:
        sim.start(timeout=120.0)
        with OffboardClient.connect(sim, 30.0) as op:
            for _ in range(3):  # fire and forget, so send it more than once
                op.param_set("{_PARAM}", 0.0)
                sim.sleep(1.0)
            sim.sleep(2.0)  # PX4 saves a changed parameter a moment after the change
        print("system_id", sim.ports["offboard"]["system_id"], flush=True)
        if sys.argv[1:] == ["hold"]:
            time.sleep(3600)
    """
)


def _earlier_run(*, killed: bool) -> int:
    """Fly the earlier run in a child process, ended by its teardown or, with ``killed``, by ``SIGKILL``.

    Returns:
        The system id of the instance it flew.
    """
    args = [sys.executable, "-c", _EARLIER_RUN] + (["hold"] if killed else [])
    child = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    try:
        # Warp prints its banner to stdout too, so the id is the line the run marks.
        system_id = next(int(line.split()[1]) for line in child.stdout if line.startswith("system_id "))
        if killed:
            child.send_signal(signal.SIGKILL)
        assert child.wait(timeout=60) in (0, -signal.SIGKILL)
        return system_id
    finally:
        child.kill()
        child.stdout.close()


def _boot_value(earlier_system_id: int) -> float:
    """Start a run, read the parameter PX4 booted with, then set it back to its airframe value, so a tree the
    earlier run left it in serves the next run as it found it.
    """
    with nx.Sim("astro_max_base", scene="empty") as sim:
        sim.start(timeout=120.0)
        link = sim.ports["offboard"]
        if link["system_id"] != earlier_system_id:
            pytest.fail(
                f"this run flew system {link['system_id']}, the earlier run {earlier_system_id}: not one instance"
            )
        mav = mavutil.mavlink_connection(f"udpin:0.0.0.0:{link['port']}", source_system=255, source_component=240)
        try:
            booted = _param(
                sim,
                mav,
                link["system_id"],
                lambda: mav.mav.param_request_read_send(link["system_id"], 1, _PARAM.encode(), -1),
            )
            _param(
                sim,
                mav,
                link["system_id"],
                lambda: mav.mav.param_set_send(
                    link["system_id"], 1, _PARAM.encode(), _AIRFRAME_VALUE, mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                ),
                until=_AIRFRAME_VALUE,
            )
            sim.sleep(2.0)  # PX4 saves a changed parameter a moment after the change
        finally:
            mav.close()
    return booted


def _param(sim, mav, system_id: int, send, *, until: float | None = None) -> float:
    """Step the sim, sending once a sim second, until PX4 reports the parameter, at ``until`` when given."""
    seen: list[float] = []
    next_send = sim.time

    def reported() -> bool:
        nonlocal next_send
        if sim.time >= next_send:
            next_send = sim.time + 1.0
            send()
        while (msg := mav.recv_match(type="PARAM_VALUE", blocking=False)) is not None:
            if msg.get_srcSystem() == system_id and msg.param_id == _PARAM:
                seen.append(msg.param_value)
        return bool(seen) and (until is None or seen[-1] == until)

    sim.wait_until(reported, sim_timeout=30.0)
    return seen[-1]


def test_a_run_boots_px4_with_no_parameter_an_earlier_run_set():
    """A run's PX4 boots with no parameter an earlier run set on the same instance of the tree.

    Given a run that sets `COM_DISARM_PRFLT` to 0 over the offboard link and ends, when a second run starts PX4
    on the same instance of the same tree, then the second PX4 reports `COM_DISARM_PRFLT` at its airframe value,
    10.0.
    """
    assert _boot_value(_earlier_run(killed=False)) == _AIRFRAME_VALUE


def test_a_run_boots_px4_with_no_parameter_a_killed_run_set():
    """A run's PX4 boots with no parameter a run killed before its teardown set.

    Given a run that sets `COM_DISARM_PRFLT` to 0 and whose process is then killed with `SIGKILL`, when a new
    run starts PX4 on the same instance of the same tree, then the new PX4 reports `COM_DISARM_PRFLT` at 10.0.
    """
    assert _boot_value(_earlier_run(killed=True)) == _AIRFRAME_VALUE
