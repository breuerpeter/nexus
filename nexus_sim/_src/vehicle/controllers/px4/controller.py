"""Px4MavlinkController: the component that speaks the PX4 peer's link.

Its work is two host stages, ``truth`` and ``exchange``: ``exchange(t, timeout)`` is the blocking lockstep that
paces the loop: opens a tcpin TCP server on :4560, which PX4 dials into as client with no HEARTBEAT, serializes the
samples of the Inertial Measurement Unit (IMU), the magnetometer, the barometer and the
Global Positioning System (GPS), which its stage reads as signals, into HIL_SENSOR and HIL_GPS, and the base
body's true state, which ``truth`` reads, into HIL_STATE_QUATERNION, then blocks on HIL_ACTUATOR_CONTROLS with a
run-ending timeout. The sensors already give their samples in their own axes; this layer only encodes wire units
and moves bytes.

The autopilot itself is a peer of the run, not of this controller: the build starts the PX4
Software In The Loop (SITL) container, :class:`~nexus_sim._src.peers.px4_sitl.runner.Px4Sitl`,
when the vehicle declares that peer, and starts nothing when a layer drops the declaration, for an
autopilot started elsewhere that dials in. The build makes this controller the same way for both: it takes the run's
Hardware In The Loop (HIL) port and PX4's system id, listens, and speaks to whatever dials in.
"""

from __future__ import annotations

import errno
import glob
import math
import os

import numpy as np

# Set the MAVLink 2 common dialect before importing mavutil, which reads it at import.
os.environ["MAVLINK20"] = "1"
os.environ["MAVLINK_DIALECT"] = "common"

from pymavlink import mavutil

from nexus_sim._src import transform
from nexus_sim._src.core import logger
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import BaroSample, Controls, GpsSample, ImuSample, MagSample
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.core.stages import peer_stages
from nexus_sim._src.peers.px4_sitl import HIL_PORT

# Where this controller looks for PX4's ULog, mirroring the recorder's
# ~/.cache/nexus/logs convention for the .rrd. PX4 writes the .ulg itself, since it
# owns the bytes; a SITL deployment points PX4's log dir here, for example by symlinking
# the build rootfs/log, so the controller can surface it as an artifact alongside
# the .rrd. When the dir is missing, because PX4 logs elsewhere or logging is off,
# ulog_path is None.
PX4_ULOG_DIR = os.path.expanduser("~/.cache/nexus/px4-ulog")

# What PX4 receives for a sensor the vehicle doesn't declare, as the sample its signal holds: a sample with no
# time, which is never new, so HIL_SENSOR marks none of its fields updated and no HIL_GPS goes out. Its values
# only fill HIL_SENSOR: an IMU and a magnetometer that read zero, and a barometer at 1013.25 hPa, 0 m and
# 25 degrees Celsius.
_NO_IMU = [(math.nan, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
_NO_MAG = [(math.nan, (0.0, 0.0, 0.0))]
_NO_BARO = [(math.nan, 1013.25, 0.0, 25.0)]
_NO_GPS = [(math.nan, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0)]

# The bits of HIL_SENSOR's fields_updated that each sensor's sample fills, as MAVLink's HIL_SENSOR_UPDATED_FLAGS
# number them: the IMU's accelerometer and gyroscope, the magnetometer, and the barometer's absolute and
# differential pressure, pressure altitude and temperature. With all four new, the mask is 0x1FFF.
_IMU_FIELDS = 0x003F
_MAG_FIELDS = 0x01C0
_BARO_FIELDS = 0x1E00


def _listener(port: int) -> int | None:
    """The process that listens on TCP `port` on this machine, from Linux's `/proc`; `None` when it finds none."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as rows:
                next(rows)
                for row in rows:
                    fields = row.split()
                    if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:  # 0A: LISTEN
                        inodes.add(f"socket:[{fields[9]}]")
        except OSError:
            continue
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) in inodes:
                return int(fd.split("/")[2])
        except OSError:
            continue
    return None


class Px4MavlinkController:
    def __init__(
        self,
        *,
        airframe: str = "",
        ip: str = "0.0.0.0",
        port: int = HIL_PORT,
        sysid: int = 1,
        compid: int = 200,
        ulog_dir: str | None = None,
        target_system: int = 1,
    ):
        """Configure the HIL link; nothing listens until ``connect``.

        Args:
            airframe: The SITL airframe the vehicle declares, `nexus:airframe` of its `NexusPx4API`
                schema, without PX4's ``none_`` prefix: the peer the run starts flies it.
            ip: The address the HIL server binds.
            port: The TCP port the HIL server listens on, ``HIL_PORT`` plus the run's PX4 instance: PX4 dials it.
            sysid: This end's MAVLink system id.
            compid: This end's MAVLink component id.
            ulog_dir: Where PX4's ULog lands, ``PX4_ULOG_DIR`` by default.
            target_system: PX4's MAVLink system id: the run's PX4 instance plus 1.
        """
        self.airframe = airframe
        self.ip = ip
        self.port = port
        self.sysid = sysid
        self.compid = compid
        self.target_system = target_system
        self.gps_fix_type = 3
        self._attitude = (1.0, 0.0, 0.0, 0.0)  # the base body's true attitude on North East Down (NED), [w, x, y, z]
        self._rates = (0.0, 0.0, 0.0)  # its true body rates, Forward Right Down (FRD), rad/s
        self.mav = None
        self.proto = None
        self._ulog_dir = ulog_dir if ulog_dir is not None else PX4_ULOG_DIR
        # The sensors' samples the exchange serializes, each a signal that holds its default when the vehicle
        # declares no such sensor.
        self.imu = Signal("imu", ImuSample, shape=(1,), default=_NO_IMU)
        self.mag = Signal("mag", MagSample, shape=(1,), default=_NO_MAG)
        self.baro = Signal("baro", BaroSample, shape=(1,), default=_NO_BARO)
        self.gps = Signal("gps", GpsSample, shape=(1,), default=_NO_GPS)
        # The time of the last sample of each sensor the exchange sent; a sample with another time is new.
        self._sent = dict.fromkeys(("imu", "mag", "baro", "gps"), math.nan)

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
        try:
            self.mav = mavutil.mavlink_connection(conn_string, source_system=self.sysid, source_component=self.compid)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            holder = _listener(self.port)
            raise ConnectionError(
                f"the HIL port {self.port} is in use by {f'process {holder}' if holder else 'another process'}: "
                "a second run on this machine that flies an autopilot started elsewhere needs the first to end"
            ) from exc
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
        """The ``truth`` and ``exchange`` host stages: the MAVLink lockstep round-trip blocks on the peer, so
        it runs between graph replays. The ``exchange`` stage reads the four sensors' samples.
        """
        (exchange,) = peer_stages(self, reads=(self.imu, self.mag, self.baro, self.gps))
        return [Stage("truth", "host", self._truth), exchange]

    def _truth(self, tick) -> None:
        """Copy the base body's true attitude and rates to the host, in PX4's frames, for the ground truth
        ``exchange`` sends: the vehicle's state, not a sensor reading, so no sensor's mount changes it.
        """
        i = tick.base
        q = transform.quat_xyzw(tick.state.body_q[i : i + 1].numpy()[0])
        omega_world = tick.state.body_qd[i : i + 1].numpy()[0][3:6]
        self._attitude = transform.body_quat_ned(q)
        self._rates = tuple(
            float(x) for x in transform.world_to_body(q, omega_world)
        )  # the body's axes point forward, right and down

    def _new(self, kind: str, sample) -> bool:
        """Whether `sample` of the sensor `kind` is new: it has a time, and not the time of the last one sent.
        A new sample becomes the last one sent.
        """
        t = float(sample["time"])
        if math.isnan(t) or t == self._sent[kind]:
            return False
        self._sent[kind] = t
        return True

    def exchange(self, t, timeout):
        time_usec = t.time_usec
        imu, mag, baro, gps = (signal.read()[0] for signal in (self.imu, self.mag, self.baro, self.gps))
        acc = [float(x) for x in imu["accel"]]
        gyro = [float(x) for x in imu["gyro"]]
        field = [float(x) for x in mag["field"]]
        fields = 0
        for kind, sample, bits in (("imu", imu, _IMU_FIELDS), ("mag", mag, _MAG_FIELDS), ("baro", baro, _BARO_FIELDS)):
            if self._new(kind, sample):
                fields |= bits

        # HIL_SENSOR every tick, which marks a sensor's fields updated only on a new sample of it, as PX4's Gazebo
        # Classic bridge does: PX4 publishes a sensor only on a message that marks its fields, so at its rate.
        self.proto.hil_sensor_send(
            time_usec,
            *acc,
            *gyro,
            *field,
            float(baro["pressure"]),
            0.0,
            float(baro["altitude"]),
            float(baro["temperature"]),
            fields,
            0,
        )

        # HIL_GPS, plus the ground-truth HIL_STATE_QUATERNION, with each new sample of the receiver, so at its rate.
        if self._new("gps", gps):
            lat = int(float(gps["lat"]) * 1e7)
            lon = int(float(gps["lon"]) * 1e7)
            alt = int(float(gps["alt"]) * 1000)
            vn, ve, vd = (int(float(v) * 100) for v in gps["velocity"])
            vel = int(float(gps["ground_speed"]) * 100)
            fix_type, eph, epv, sats = int(gps["fix_type"]), 100, 100, 10
            if self.gps_fix_type < 2:
                fix_type, eph, sats = 0, 9999, 0
            self.proto.hil_gps_send(
                time_usec, fix_type, lat, lon, alt, eph, epv, vel, vn, ve, vd, 65535, sats, id=0, yaw=0
            )
            xacc_mg, yacc_mg, zacc_mg = (int(a * 1000 / 9.81) for a in acc)
            self.proto.hil_state_quaternion_send(
                time_usec,
                list(self._attitude),
                *self._rates,
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
