"""Px4MavlinkController: the single host and marshalling boundary, architecture.md §2 and §6.

Implements ``Controller.exchange(measurement, t) -> Controls`` as the blocking lockstep that paces
the loop: opens a tcpin TCP server on :4560, which PX4 dials into as client with no HEARTBEAT,
serializes the typed Measurement into HIL_SENSOR, HIL_GPS and HIL_STATE_QUATERNION with the
encoders lifted verbatim from the bridge, then blocks on HIL_ACTUATOR_CONTROLS with a run-ending
timeout. newton-sensors already derives the Measurement in
the Forward-Right-Down (FRD) body frame; this layer only encodes wire units and moves bytes.

It also owns the PX4 Software In The Loop (SITL) peer end to end: ``prepare`` builds it,
``connect`` launches it, ``close`` stops it. This controller is the one thing that already knows it
talks to a PX4, since it holds the Hardware In The Loop (HIL) socket and the ULog artifact, so the
autopilot's lifecycle belongs here, and reaches every run through the
single ``build_from_launch`` seam.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np

# Set the MAVLink dialect before importing mavutil; this matches the bridge.
os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"

from pymavlink import mavutil

from nexus._src.core import logger
from nexus._src.core.schema import Controls
from nexus._src.vehicle.controllers.px4.sitl import CONTAINER, PX4_LOG_DIR, Px4Sitl, build_px4_sitl

# Where this controller looks for PX4's ULog, mirroring the recorder's
# ~/.cache/nexus/logs convention for the .rrd. PX4 writes the .ulg itself, since it
# owns the bytes; a SITL deployment points PX4's log dir here, for example by symlinking
# the build rootfs/log, so the controller can surface it as an artifact alongside
# the .rrd. When the dir is missing, because PX4 logs elsewhere or logging is off,
# ulog_path is None.
PX4_ULOG_DIR = os.path.expanduser("~/.cache/nexus/px4-ulog")


class Px4MavlinkController:
    # The host boundary, architecture.md §5: ``exchange`` is a blocking MAVLink lockstep round-trip,
    # so it can't join a CUDA graph. The captured strategy records only the device region and runs
    # this seam, read -> exchange -> write_controls, between replays; see Orchestrator._loop_captured_host_exchange.
    host_boundary = True

    def __init__(
        self,
        *,
        ip: str = "0.0.0.0",
        port: int = 4560,
        sysid: int = 1,
        compid: int = 200,
        gps_rate_hz: float = 10.0,
        ulog_dir: str | None = None,
        airframe: str = "astro_max",
    ):
        self.ip = ip
        self.port = port
        self.sysid = sysid
        self.compid = compid
        self.gps_interval = 1.0 / gps_rate_hz
        self._last_gps = 0.0
        self.gps_fix_type = 3
        self.mav = None
        self.proto = None
        self._ulog_dir = ulog_dir if ulog_dir is not None else PX4_ULOG_DIR
        # The PX4 peer this controller flies against. Constructing it starts nothing; `prepare`
        # builds it and `connect` launches it. `airframe` is the registry's make target, and this
        # line prepends `none_`, PX4's "no simulator" or external-HIL prefix.
        os.makedirs(PX4_LOG_DIR, exist_ok=True)
        self._sitl = Px4Sitl(
            container=CONTAINER,
            log_path=os.path.join(PX4_LOG_DIR, f"px4-{time.strftime('%Y%m%d-%H%M%S')}.log"),
            airframe=f"none_{airframe}",
        )

    def prepare(self) -> None:
        """Get the PX4 peer ready before the run starts: check the prerequisites and build PX4 SITL.

        Blocking, and called from ``build_from_launch`` on the caller's thread: the launch's own
        ``make`` must fit inside the 30 s preroll window, GH #39, which an incremental no-op does
        and a cold build never does, so the build can't happen any later than this.

        Raises:
            RuntimeError: No PX4 checkout at ``$PX4_DIR``, or the docker daemon is unreachable.
        """
        logger.info("building PX4 SITL (incremental; a cold build takes minutes) …")
        build_px4_sitl()

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

    def artifacts(self) -> dict:
        """This controller's contribution to Sim.artifacts(): the PX4 ULog and the PX4 console log
        of the container it started. Keeps the PX4-specific keys out of the generic Sim API.
        """
        return {"ulog": self.ulog_path, "px4_log": self._sitl.log_path}

    def connect(self) -> None:
        conn_string = f"tcpin:{self.ip}:{self.port}"
        logger.info(f"Waiting for PX4 connection on {conn_string}...")
        self.mav = mavutil.mavlink_connection(conn_string, source_system=self.sysid, source_component=self.compid)
        self.proto = self.mav.mav
        self.proto.srcSystem = self.sysid
        self.proto.srcComponent = self.compid
        self.mav.target_system = 1
        self.mav.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        logger.info(f"MAVLink source {self.sysid}/{self.compid} -> target 1/{self.mav.target_component}")
        # PX4 last, and the ordering is what makes this safe, not a timing margin: the preceding
        # mavlink_connection("tcpin:") binds and *listens* in the constructor, and the accept
        # is lazy, on the first recv, so :4560 is already up when the launch asks PX4 to dial in. PX4
        # retries that connect forever, so the only real constraint is that it comes up inside the
        # sim's preroll window.
        logger.info(f"launching PX4 SITL ({self._sitl.airframe}) -> {self._sitl.log_path}")
        self._sitl.launch()

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
        # The peer dies with the run that started it, but this step tolerates failure, the same as
        # every other step here. A daemon that has gone away must not cost the rest of the teardown:
        # the MAVLink socket still has to close, and the orchestrator still has to flush the
        # recording, which is exactly what a failing run needs to have kept.
        try:
            self._sitl.stop()
        except Exception as exc:
            logger.warning(f"PX4 container teardown failed ({exc})")
        try:
            if self.mav is not None:
                self.mav.close()
        except Exception:
            pass
