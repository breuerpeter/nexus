"""The state a PX4 Software In The Loop (SITL) run boots with: its airframe's parameters and no mission, and
nothing an earlier run left.

PX4 saves a parameter set at run time, and a mission it accepts, in its working folder. Each test flies an
earlier run in a child process that sets ``COM_DISARM_PRFLT`` to 0, or uploads a mission, over the offboard
link, then a run of its own through ``nx.Sim`` that reads PX4's state back, so the tests need docker and a PX4
tree carrying the airframe the sim flies; they skip without one.
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

# The earlier run: with `param` it sets the parameter, with `mission` it uploads the mission example's plan at
# the plan's home, and PX4 stores either a moment later. Then it prints its instance's system id. With `hold`,
# it then waits until the test ends it with `SIGKILL`, so the run never reaches its teardown.
_EARLIER_RUN = textwrap.dedent(
    f"""
    import sys, time
    import nexus_uav as nx
    from nexus_uav.examples._lib.origin import origin_layer
    from nexus_uav.examples.controllers.px4.mission import PLAN
    from nexus_uav.px4 import OffboardClient, read_plan

    mission = "mission" in sys.argv[1:]
    layer = origin_layer(*read_plan(PLAN).home) if mission else None
    with nx.Sim("astro_max_base", scene="empty", layer=layer) as sim:
        sim.start(timeout=120.0)
        with OffboardClient.connect(sim, 30.0) as op:
            if mission:
                op.upload_mission(PLAN)
                sim.wait_until(op.mission_uploaded, sim_timeout=60.0)
            else:
                for _ in range(3):  # fire and forget, so send it more than once
                    op.param_set("{_PARAM}", 0.0)
                    sim.sleep(1.0)
            sim.sleep(2.0)  # PX4 stores a change a moment after it
        print("system_id", sim.ports["offboard"]["system_id"], flush=True)
        if "hold" in sys.argv[1:]:
            time.sleep(3600)
    """
)


def _earlier_run(*args: str) -> int:
    """Fly the earlier run in a child process, ended by its teardown or, with ``hold``, by ``SIGKILL``.

    Returns:
        The system id of the instance it flew.
    """
    child = subprocess.Popen([sys.executable, "-c", _EARLIER_RUN, *args], stdout=subprocess.PIPE, text=True)
    try:
        # Warp prints its banner to stdout too, so the id is the line the run marks.
        system_id = next(int(line.split()[1]) for line in child.stdout if line.startswith("system_id "))
        if "hold" in args:
            child.send_signal(signal.SIGKILL)
        assert child.wait(timeout=60) in (0, -signal.SIGKILL)
        return system_id
    finally:
        child.kill()
        child.stdout.close()


def _next_run(earlier_system_id: int, read):
    """Start a run on the earlier run's instance, and return what ``read(sim, mav, system_id)`` reads from PX4.

    ``read`` also puts back what it reads, so a tree that keeps PX4's state serves the next run as the test found it.
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
            return read(sim, mav, link["system_id"])
        finally:
            mav.close()


def _reply(sim, mav, system_id: int, send, kind: str, matches=lambda msg: True):
    """Step the sim, sending once a sim second, until PX4 replies with a ``kind`` message that ``matches``."""
    seen = []
    next_send = sim.time

    def replied() -> bool:
        nonlocal next_send
        if sim.time >= next_send:
            next_send = sim.time + 1.0
            send()
        while (msg := mav.recv_match(type=kind, blocking=False)) is not None:
            if msg.get_srcSystem() == system_id and matches(msg):
                seen.append(msg)
        return bool(seen)

    sim.wait_until(replied, sim_timeout=30.0)
    return seen[-1]


def _booted_param(sim, mav, system_id: int) -> float:
    """The parameter PX4 booted with, then set back to its airframe value."""
    name = _PARAM.encode()
    booted = _reply(
        sim,
        mav,
        system_id,
        lambda: mav.mav.param_request_read_send(system_id, 1, name, -1),
        "PARAM_VALUE",
        lambda msg: msg.param_id == _PARAM,
    ).param_value
    _reply(
        sim,
        mav,
        system_id,
        lambda: mav.mav.param_set_send(system_id, 1, name, _AIRFRAME_VALUE, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
        "PARAM_VALUE",
        lambda msg: msg.param_id == _PARAM and msg.param_value == _AIRFRAME_VALUE,
    )
    sim.sleep(2.0)  # PX4 stores a change a moment after it
    return booted


def _booted_mission(sim, mav, system_id: int) -> int:
    """The count of the mission PX4 booted with, then cleared."""
    mission = mavutil.mavlink.MAV_MISSION_TYPE_MISSION
    count = _reply(
        sim, mav, system_id, lambda: mav.mav.mission_request_list_send(system_id, 1, mission), "MISSION_COUNT"
    ).count
    _reply(sim, mav, system_id, lambda: mav.mav.mission_clear_all_send(system_id, 1, mission), "MISSION_ACK")
    sim.sleep(2.0)  # PX4 stores a change a moment after it
    return count


def test_a_run_boots_px4_with_no_parameter_an_earlier_run_set():
    """A run's PX4 boots with no parameter an earlier run set on the same instance of the tree.

    Given a run that sets `COM_DISARM_PRFLT` to 0 over the offboard link and ends, when a second run starts PX4
    on the same instance of the same tree, then the second PX4 reports `COM_DISARM_PRFLT` at its airframe value,
    10.0.
    """
    assert _next_run(_earlier_run("param"), _booted_param) == _AIRFRAME_VALUE


def test_a_run_boots_px4_with_no_parameter_a_killed_run_set():
    """A run's PX4 boots with no parameter a run killed before its teardown set.

    Given a run that sets `COM_DISARM_PRFLT` to 0 and whose process is then killed with `SIGKILL`, when a new
    run starts PX4 on the same instance of the same tree, then the new PX4 reports `COM_DISARM_PRFLT` at 10.0.
    """
    assert _next_run(_earlier_run("param", "hold"), _booted_param) == _AIRFRAME_VALUE


def test_a_run_boots_px4_with_no_mission_an_earlier_run_stored():
    """A run's PX4 boots with no mission an earlier run stored on the same instance of the tree.

    Given a run that uploads `box.plan` over the offboard link, PX4 accepts it, and the run ends, when a second
    run starts PX4 on the same instance of the same tree, then the second PX4 reports a mission of 0 items.
    """
    assert _next_run(_earlier_run("mission"), _booted_mission) == 0
