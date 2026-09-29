"""Px4MavlinkController: the loop face of the PX4 peer, architecture.md §2 and §6.

Its work is two host stages, ``read`` and ``exchange``: ``exchange(measurement, t)`` is the blocking
lockstep that paces the loop: opens a tcpin TCP server on :4560, which PX4 dials into as client with no HEARTBEAT,
serializes the typed Measurement into HIL_SENSOR, HIL_GPS and HIL_STATE_QUATERNION with the
encoders lifted verbatim from the bridge, then blocks on HIL_ACTUATOR_CONTROLS with a run-ending
timeout. newton-sensors already derives the Measurement in
the Forward-Right-Down (FRD) body frame; this layer only encodes wire units and moves bytes.

The autopilot itself is a peer of the run, not of this controller: the build starts the PX4
Software In The Loop (SITL) container, :class:`~nexus._src.peers.px4_sitl.runner.Px4Sitl`,
for a managed PX4 peer, and starts nothing for an external one, an autopilot started elsewhere
that dials in. The build makes this controller the same way for both: it takes the run's
Hardware In The Loop (HIL) port and PX4's system id, listens, and speaks to whatever dials in.
"""

from __future__ import annotations

import glob
import os

import numpy as np

# Set the MAVLink dialect before importing mavutil; this matches the bridge.
os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"

from pymavlink import mavutil

from nexus._src.core import logger
from nexus._src.core.schema import Controls
from nexus._src.core.stages import peer_stages
from nexus._src.peers.px4_sitl import HIL_PORT

# Where this controller looks for PX4's ULog, mirroring the recorder's
# ~/.cache/nexus/logs convention for the .rrd. PX4 writes the .ulg itself, since it
# owns the bytes; a SITL deployment points PX4's log dir here, for example by symlinking
# the build rootfs/log, so the controller can surface it as an artifact alongside
# the .rrd. When the dir is missing, because PX4 logs elsewhere or logging is off,
# ulog_path is None.
PX4_ULOG_DIR = os.path.expanduser("~/.cache/nexus/px4-ulog")


class Px4MavlinkController:
    def __init__(
        self,
        *,
        ip: str = "0.0.0.0",
        port: int = HIL_PORT,
        sysid: int = 1,
        compid: int = 200,
        gps_rate_hz: float = 10.0,
        ulog_dir: str | None = None,
        target_system: int = 1,
    ):
        """Configure the HIL link; nothing listens until ``connect``.

        Args:
            ip: The address the HIL server binds.
            port: The TCP port the HIL server listens on, the run's ``peers.px4.hil_port``: PX4 dials it.
            sysid: This end's MAVLink system id.
            compid: This end's MAVLink component id.
            gps_rate_hz: How often ``HIL_GPS`` and the ground-truth state go out.
            ulog_dir: Where PX4's ULog lands, ``PX4_ULOG_DIR`` by default.
            target_system: PX4's MAVLink system id, the run's ``peers.px4.system_id``: instance + 1.
        """
        self.ip = ip
        self.port = port
        self.sysid = sysid
        self.compid = compid
        self.target_system = target_system
        self.gps_interval = 1.0 / gps_rate_hz
        self._last_gps = 0.0
        self.gps_fix_type = 3
        self.mav = None
        self.proto = None
        self._ulog_dir = ulog_dir if ulog_dir is not None else PX4_ULOG_DIR

    @property
    def ulog_path(self) -> str | None:
        """The PX4 ULog for this run, if whoever ran PX4 pointed it at this controller's log
        dir, PX4_ULOG_DIR. Newest ``*.ulg`` found there: with SDLOG_MODE=2, which logs from
        boot to shutdown, that's exactly one file per run, so there is nothing
        to disambiguate. None if the dir is missing or empty.
        """
        if not os.path.isdir(self._ulog_dir):
            return None
        ulogs = glob.glob(os.path.join(self._ulog_dir, "**", "*.ulg"), recursive=True)
        if not ulogs:
            return None
        return max(ulogs, key=os.path.getmtime)

    @property
    def attached(self) -> bool:
        """Whether PX4 has dialed in: the HIL link accepted its connection.

        The accept is lazy, on the first receive, so this turns true during the exchange that meets
        PX4's connection. The preroll holds the sim clock until then, so PX4's first sensor stamp is
        near zero even when PX4 took long to dial in.
        """
        return self.mav is not None and self.mav.port is not None

    def artifacts(self) -> dict:
        """This controller's contribution to Sim.artifacts(): the PX4 ULog. The console log is the
        peer's own artifact. Keeps the PX4-specific keys out of the generic Sim API.
        """
        return {"ulog": self.ulog_path}

    def connect(self) -> None:
        conn_string = f"tcpin:{self.ip}:{self.port}"
        logger.info(f"Waiting for PX4 connection on {conn_string}...")
        self.mav = mavutil.mavlink_connection(conn_string, source_system=self.sysid, source_component=self.compid)
        self.proto = self.mav.mav
        self.proto.srcSystem = self.sysid
        self.proto.srcComponent = self.compid
        self.mav.target_system = self.target_system
        self.mav.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        logger.info(
            f"MAVLink source {self.sysid}/{self.compid} -> target {self.target_system}/{self.mav.target_component}"
        )
        # mavlink_connection("tcpin:") binds and *listens* in the constructor, and the accept is lazy,
        # on the first recv. The peer started at build, so PX4 might already be dialing: it retries that
        # connect forever, so the only real constraint is that it comes up inside the sim's preroll window.

    def stages(self):
        """The ``read`` and ``exchange`` host stages: the MAVLink lockstep round-trip blocks on the
        peer, so it runs between graph replays.
        """
        return peer_stages(self)

    def exchange(self, meas, t, timeout):
        time_usec = t.time_usec

        # HIL_SENSOR every tick.
        self.proto.hil_sensor_send(
            time_usec,
            meas.xacc,
            meas.yacc,
            meas.zacc,
            meas.xgyro,
            meas.ygyro,
            meas.zgyro,
            meas.xmag,
            meas.ymag,
            meas.zmag,
            meas.abs_pressure,
            0.0,
            meas.pressure_alt,
            meas.temperature,
            0x1FFF,
            0,
        )

        # HIL_GPS, plus the ground-truth HIL_STATE_QUATERNION, at the Global Positioning System (GPS) sub-rate.
        if meas.gps_valid and (t.sim_time - self._last_gps >= self.gps_interval):
            self._last_gps = t.sim_time
            lat = int(meas.lat_deg * 1e7)
            lon = int(meas.lon_deg * 1e7)
            alt = int(meas.alt_m * 1000)
            vn = int(meas.vn * 100)
            ve = int(meas.ve * 100)
            vd = int(meas.vd * 100)
            vel = int(meas.ground_speed * 100)
            fix_type, eph, epv, sats = meas.fix_type, 100, 100, 10
            if self.gps_fix_type < 2:
                fix_type, eph, sats = 0, 9999, 0
            self.proto.hil_gps_send(
                time_usec, fix_type, lat, lon, alt, eph, epv, vel, vn, ve, vd, 65535, sats, id=0, yaw=0
            )
            xacc_mg = int(meas.xacc * 1000 / 9.81)
            yacc_mg = int(meas.yacc * 1000 / 9.81)
            zacc_mg = int(meas.zacc * 1000 / 9.81)
            self.proto.hil_state_quaternion_send(
                time_usec,
                list(meas.quat_wxyz),
                meas.rollspeed,
                meas.pitchspeed,
                meas.yawspeed,
                lat,
                lon,
                alt,
                vn,
                ve,
                vd,
                0,
                0,
                xacc_mg,
                yacc_mg,
                zacc_mg,
            )

        # Block for actuator controls, the lockstep. None on timeout -> run ends.
        msg = self.mav.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=True, timeout=timeout)
        if msg is None:
            return None
        return Controls(command=np.asarray(msg.controls, dtype=np.float32))

    def close(self) -> None:
        # The peer is the run's to stop; this end only closes its socket. It tolerates failure, the same
        # as every other teardown step: the orchestrator still has to flush the recording.
        try:
            if self.mav is not None:
                self.mav.close()
        except Exception:
            pass
