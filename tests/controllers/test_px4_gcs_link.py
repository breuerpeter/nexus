"""What PX4 concludes about a Ground Control Station (GCS) in a recording run, now that nothing in
the sim heartbeats to it.

Each module fixture flies one recording run through ``na.Sim``, which builds and launches PX4 from
``$PX4_DIR``, so the tests need docker and a PX4 checkout carrying the airframe the sim flies; they
skip without one. The runs pace at real time, ``rtf=1.0``, so a heartbeat sent once a wall-clock
second also arrives once a second of PX4's lockstep time, as it does when a person flies.
"""

import os
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does: whichever import runs
# first sets it, and the controller's HIL_GPS needs the common dialect's fields.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus as na
from nexus._src.vehicle.controllers.px4.sitl import px4_dir

_AIRFRAMES = px4_dir() / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes"
if not list(_AIRFRAMES.glob("*_none_astro_max")):
    pytest.skip(
        f"needs docker and a PX4 checkout at $PX4_DIR ({px4_dir()}) carrying the none_astro_max airframe",
        allow_module_level=True,
    )

_OPERATOR_LINK = "udpin:0.0.0.0:14540"  # PX4's offboard and onboard API instance sends here
_GCS_LINK = "udpin:0.0.0.0:14550"  # PX4's GCS instance, :18570, sends here, where QGroundControl listens
_DATALINK_TIMEOUT_S = 10.0  # COM_DL_LOSS_T at its default: how long PX4 waits before it counts a GCS as lost
_SET_MODE = mavutil.mavlink.MAV_CMD_DO_SET_MODE
_ARM = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
_HOLD = (4, 3)  # PX4's custom main and sub mode for Hold, the loiter sub mode of its auto mode


class _Peer:
    """A MAVLink peer on one of PX4's links, identified as a GCS. Its pump thread keeps what PX4 sends and,
    while ``heartbeat`` stays true, heartbeats once a second, the way QGroundControl does. Without the
    heartbeat, PX4's link-loss logic never counts it.
    """

    def __init__(self, conn: str, *, heartbeat: bool):
        self._mav = mavutil.mavlink_connection(
            conn, source_system=255, source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER
        )
        self.heartbeat = heartbeat
        self.px4 = None  # the last HEARTBEAT PX4 sent
        self.statustexts: list[str] = []
        self.params: dict[str, int] = {}  # the INT32 parameters PX4 reported, by name
        self.acks: dict[int, int] = {}  # the last COMMAND_ACK result, by command
        self._lock = threading.Lock()  # guards every send on the shared connection
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, name="px4-peer", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._mav.close()

    def command(self, command: int, *params: float) -> None:
        with self._lock:
            self._mav.mav.command_long_send(1, 1, command, 0, *params, *[0.0] * (7 - len(params)))

    def param_request(self, name: str) -> None:
        with self._lock:
            self._mav.mav.param_request_read_send(1, 1, name.encode(), -1)

    def param_set_int(self, name: str, value: int) -> None:
        # PX4 reads an INT32 parameter from the bits of PARAM_SET's float field, not from its value.
        bits = struct.unpack("<f", struct.pack("<i", value))[0]
        with self._lock:
            self._mav.mav.param_set_send(1, 1, name.encode(), bits, mavutil.mavlink.MAV_PARAM_TYPE_INT32)

    def _pump(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.is_set():
            if self.heartbeat and time.monotonic() - last_heartbeat >= 1.0:
                last_heartbeat = time.monotonic()
                with self._lock:
                    self._mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
            msg = self._mav.recv_match(blocking=True, timeout=0.1)
            if msg is None or msg.get_srcSystem() != 1:
                continue
            kind = msg.get_type()
            if kind == "HEARTBEAT" and msg.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
                self.px4 = msg
            elif kind == "STATUSTEXT":
                self.statustexts.append(msg.text if isinstance(msg.text, str) else msg.text.decode(errors="ignore"))
            elif kind == "PARAM_VALUE" and msg.param_type == mavutil.mavlink.MAV_PARAM_TYPE_INT32:
                self.params[msg.param_id] = struct.unpack("<i", struct.pack("<f", msg.param_value))[0]
            elif kind == "COMMAND_ACK":
                self.acks[msg.command] = msg.result


def _mode(heartbeat) -> tuple[int, int]:
    """PX4's custom main and sub mode, from a HEARTBEAT."""
    return (heartbeat.custom_mode >> 16) & 0xFF, (heartbeat.custom_mode >> 24) & 0xFF


def _fly(sim, seconds: float) -> None:
    """Step the sim for ``seconds`` of sim time. Unlike ``sim.sleep``, it raises when the run ends first."""
    until = sim.physics[sim.base_body].latest().t + seconds
    sim.wait_until(lambda: sim.physics[sim.base_body].latest().t >= until, sim_timeout=seconds + 1.0)


def _retry_until(sim, done, send, *, sim_timeout: float) -> None:
    """Step the sim until ``done()``, sending again once a second until then, since PX4 can drop a command
    that races its state machine.
    """
    last = 0.0

    def settled() -> bool:
        nonlocal last
        if done():
            return True
        if time.monotonic() - last >= 1.0:
            last = time.monotonic()
            send()
        return False

    sim.wait_until(settled, sim_timeout=sim_timeout)


def _set_param(sim, link: _Peer, name: str, value: int) -> None:
    """Set an INT32 PX4 parameter, stepping the sim until PX4 reports the new value."""
    _retry_until(sim, lambda: link.params.get(name) == value, lambda: link.param_set_int(name, value), sim_timeout=30.0)


@pytest.fixture(scope="module")
def run_with_no_gcs():
    """A recording run with no GCS and no operator.

    A command link on the operator port, which never heartbeats, puts PX4 in Hold, where a command arms it
    without sticks, and waits until PX4 reports it could arm and has been up past the datalink timeout. Then
    it makes a GCS required, ``NAV_DLL_ACT`` 2, asks PX4 to arm, and puts the parameter back.
    """
    with na.Sim(control="px4-sitl", log=True, rtf=1.0) as sim:
        sim.start(timeout=120.0)
        link = _Peer(_OPERATOR_LINK, heartbeat=False)
        try:
            sim.wait_until(lambda: link.px4 is not None, sim_timeout=60.0)
            _retry_until(
                sim, lambda: _mode(link.px4) == _HOLD, lambda: link.command(_SET_MODE, 1, *_HOLD), sim_timeout=120.0
            )
            sim.wait_until(lambda: link.px4.system_status == mavutil.mavlink.MAV_STATE_STANDBY, sim_timeout=120.0)
            sim.wait_until(lambda: sim.physics[sim.base_body].latest().t > _DATALINK_TIMEOUT_S, sim_timeout=60.0)
            _retry_until(
                sim, lambda: "NAV_DLL_ACT" in link.params, lambda: link.param_request("NAV_DLL_ACT"), sim_timeout=30.0
            )
            found = link.params["NAV_DLL_ACT"]
            _set_param(sim, link, "NAV_DLL_ACT", 2)
            try:
                _retry_until(sim, lambda: _ARM in link.acks, lambda: link.command(_ARM, 1.0), sim_timeout=30.0)
            finally:
                # PX4 saves each parameter change in the checkout, and the next PX4 run reads it back.
                _set_param(sim, link, "NAV_DLL_ACT", found)
            _fly(sim, 2.0)  # PX4 writes a changed parameter back a moment after the change
        finally:
            link.close()
    return SimpleNamespace(arm_result=link.acks[_ARM], **sim.artifacts())


@pytest.fixture(scope="module")
def run_where_the_gcs_leaves():
    """A recording run with a GCS on its usual port. The GCS heartbeats once a second and sends a mode
    request that names no mode, which PX4 answers with a status message. Then it goes quiet for longer than
    the datalink timeout, the way a GCS device does when it drops off the network.
    """
    with na.Sim(control="px4-sitl", log=True, rtf=1.0) as sim:
        sim.start(timeout=120.0)
        gcs = _Peer(_GCS_LINK, heartbeat=True)
        try:
            sim.wait_until(lambda: gcs.px4 is not None, sim_timeout=60.0)
            _fly(sim, 3.0)  # PX4 sends status messages on this link once a GCS heartbeat reached it
            gcs.command(_SET_MODE, 0)  # base mode 0, which PX4 answers with "Unsupported base mode"
            _fly(sim, 3.0)
            gcs.heartbeat = False
            _fly(sim, _DATALINK_TIMEOUT_S + 5.0)
        finally:
            gcs.close()
    return SimpleNamespace(gcs_statustexts=gcs.statustexts, **sim.artifacts())


def test_dropping_the_real_gcs_trips_px4s_link_loss_handling(run_where_the_gcs_leaves):
    """Dropping the real GCS trips PX4's link-loss handling.

    PX4 says "Connection to ground station lost" once no GCS heartbeat has reached it for the datalink
    timeout, and the run keeps PX4's console log as an artifact.
    """
    log = Path(run_where_the_gcs_leaves.px4_log).read_text(errors="ignore")
    assert "Connection to ground station lost" in log


def test_a_run_with_no_gcs_at_all_never_reports_one(run_with_no_gcs):
    """A run with no GCS at all never reports one.

    With a GCS required, PX4 answers the arm request with a temporary rejection while it counts the GCS as
    lost. PX4 could arm until the run made a GCS required, so nothing else refuses it.
    """
    assert run_with_no_gcs.arm_result == mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED


def test_a_real_gcs_still_receives_statustext_on_its_own_link(run_where_the_gcs_leaves):
    """A real GCS still receives STATUSTEXT on its own link.

    PX4 answers the GCS's mode request with "Unsupported base mode", and the GCS link carries the status
    message, as it carries every one to QGroundControl or a Pilot Pro.
    """
    texts = run_where_the_gcs_leaves.gcs_statustexts
    assert any("Unsupported base mode" in text for text in texts)
