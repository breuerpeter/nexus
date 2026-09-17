#!/usr/bin/env python3
# Acronyms: Ground Control Station (GCS), Real Time Streaming Protocol (RTSP), Local Area Network (LAN),
# Picture In Picture (PIP).
"""Advertise the sim's MediaMTX camera feeds to the GCS over the MAVLink camera protocol (GH #34).

The GCS's manual "RTSP Video Stream" setting is a single URL wired to its primary receiver only:
a second manual URL is architecturally impossible (`VideoManager::_updateSettings` gates the
manual block on the primary id). Camera-protocol discovery is the multi-feed path: this tool
joins the GCS MAVLink fabric as camera component 100 on the vehicle's system id (topology A,
no emulated companion) and advertises each MediaMTX path as an RTSP stream of the one camera. The GCS then
pulls the RTSP feeds itself and the manual URL setup step becomes unnecessary (discovered
streams take precedence over the manual setting).

With plain streams (the default) the GCS shows ONE feed at a time with a stream-selector
dropdown on the camera panel: pick cam1 or cam2 full-screen. That is the GCS's ceiling for
normal streams: its only simultaneous-two-feed rendering is the main+thermal receiver pair,
where the stream flagged ``VIDEO_STREAM_STATUS_FLAGS_THERMAL`` rides the second receiver as
PIP/Full/Blend.

Which streams get that flag is decided by the STREAM NAME, so the modality carries end to end
from the vehicle USD: an ``ir*`` path is a genuinely thermal feed (the isaacsim runtime publishes
a camera prim authored ``sensor:modality = "ir"`` there) and is flagged on its own. ``--pip``
stays the deliberate mislabel for two-EO vehicles: it flags the SECOND stream thermal to buy the
same simultaneous rendering, at the cost of the UI calling it "Thermal View Mode" and no
tap-to-swap (mode dropdown only). The pixels are untouched either way; the flag is purely
the GCS's receiver-1 routing key.

Run (topology A, any time after mavp2p is up; reconnects until then):

    uv run python tools/camera_advertiser.py                     # astro_max_fpv_lr1: cam1 + cam2
    uv run python tools/camera_advertiser.py --pip               # both at once via the thermal slot
    uv run python tools/camera_advertiser.py --streams cam1      # single-camera vehicles
    uv run python tools/camera_advertiser.py --streams cam1,ir1:640x512@24   # EO + a real IR feed

The advertised WxH@FPS are what the GCS is TOLD, not what it receives, so give a stream that is not
1280x720@30 its own suffix; the thermal one returns at the LWIR core size, 640x512@24.

The advertised RTSP URIs must be reachable from the TABLET, so the default base derives the
host's LAN IP (override with ``--rtsp-base rtsp://<ip>:8554``). Topology B has a real
companion (payload-camera-service) owning camera comp 100 on the link; don't run this there.

Protocol notes (read from the GCS's source, a QGroundControl fork, not the MAVLink spec alone): discovery
triggers on a HEARTBEAT from comp 100..105 on the vehicle's sysid; the GCS then sends the legacy
per-message requests (MAV_CMD_REQUEST_CAMERA_INFORMATION/_SETTINGS/_STORAGE_INFORMATION/
_CAMERA_CAPTURE_STATUS and MAV_CMD_REQUEST_VIDEO_STREAM_INFORMATION/_STATUS, stream id 0 =
all, with per-id retries every 2 s until ``count`` streams arrived). Every command needs a
COMMAND_ACK; VIDEO_START/STOP_STREAMING are ACKed as no-ops (MediaMTX streams are pull-based
and always running).
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from dataclasses import dataclass

# Pin the MAVLink dialect before importing mavutil, the same contract as Px4MavlinkController /
# Px4Offboard; VIDEO_STREAM_INFORMATION ids >255 need MAVLink 2 on the wire.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

mav = mavutil.mavlink


@dataclass
class Stream:
    """One advertised video stream: an RTSP path on MediaMTX."""

    stream_id: int  # 1-based, per the camera protocol
    name: str  # MediaMTX path == advertised stream name: cam1, cam2, …
    uri: str
    width: int
    height: int
    fps: float
    thermal: bool


def lan_ip() -> str:
    """The host's outbound LAN IP, the address the GCS device can reach; it sends no packets."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def parse_stream(spec: str, index: int, base: str, thermal: bool) -> Stream:
    """``cam1`` or ``cam1:1280x720@30`` -> a :class:`Stream` with uri = ``<base>/<name>``."""
    name, _, fmt = spec.partition(":")
    width, height, fps = 1280, 720, 30.0
    if fmt:
        res, _, rate = fmt.partition("@")
        w, _, h = res.partition("x")
        width, height = int(w), int(h)
        if rate:
            fps = float(rate)
    return Stream(index + 1, name, f"{base}/{name}", width, height, fps, thermal)


class CameraAdvertiser:
    """Camera comp 100 on the vehicle's sysid: heartbeat + answer the GCS's camera-protocol requests."""

    def __init__(self, conn, streams: list[Stream], sysid: int):
        self._conn = conn
        self._streams = streams
        self._sysid = sysid
        self._t0 = time.monotonic()
        self._mode = mav.CAMERA_MODE_VIDEO

    def _ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1e3)

    def heartbeat(self) -> None:
        """1 Hz camera-component heartbeat: the discovery trigger, since the GCS keys on comp 100..105."""
        self._conn.mav.heartbeat_send(mav.MAV_TYPE_CAMERA, mav.MAV_AUTOPILOT_INVALID, 0, 0, mav.MAV_STATE_ACTIVE)

    def handle(self, msg) -> None:
        """Dispatch one inbound message: COMMAND_LONGs addressed to this component."""
        if msg.get_type() != "COMMAND_LONG":
            return
        if msg.target_system not in (0, self._sysid) or msg.target_component not in (
            0,
            mav.MAV_COMP_ID_CAMERA,
            mav.MAV_COMP_ID_ALL,
        ):
            return
        handler = {
            mav.MAV_CMD_REQUEST_CAMERA_INFORMATION: self._send_camera_information,
            mav.MAV_CMD_REQUEST_CAMERA_SETTINGS: self._send_camera_settings,
            mav.MAV_CMD_REQUEST_STORAGE_INFORMATION: self._send_storage_information,
            mav.MAV_CMD_REQUEST_CAMERA_CAPTURE_STATUS: self._send_capture_status,
            mav.MAV_CMD_REQUEST_VIDEO_STREAM_INFORMATION: self._send_stream_information,
            mav.MAV_CMD_REQUEST_VIDEO_STREAM_STATUS: self._send_stream_status,
            mav.MAV_CMD_VIDEO_START_STREAMING: None,  # pull-based: always running, the acknowledgement is enough
            mav.MAV_CMD_VIDEO_STOP_STREAMING: None,
            mav.MAV_CMD_SET_CAMERA_MODE: self._set_mode,
        }
        cmd = msg.command
        if cmd not in handler:
            self._ack(msg, cmd, mav.MAV_RESULT_UNSUPPORTED)
            return
        self._ack(msg, cmd, mav.MAV_RESULT_ACCEPTED)
        if handler[cmd] is not None:
            handler[cmd](msg)
        print(
            f"[camera_advertiser] answered {mavutil.mavlink.enums['MAV_CMD'][cmd].name} from {msg.get_srcSystem()}",
            flush=True,
        )

    def _ack(self, msg, cmd: int, result: int) -> None:
        self._conn.mav.command_ack_send(cmd, result, 0, 0, msg.get_srcSystem(), msg.get_srcComponent())

    def _selected(self, msg) -> list[Stream]:
        """The streams a VIDEO_STREAM_* request addresses; param1 is the stream id, 0 = all."""
        sid = int(msg.param1)
        return self._streams if sid == 0 else [s for s in self._streams if s.stream_id == sid]

    def _send_camera_information(self, msg) -> None:
        first = self._streams[0]
        self._conn.mav.camera_information_send(
            self._ms(),
            b"Freefly".ljust(32, b"\0"),  # vendor_name, uint8[32]: pymavlink needs the padding
            b"nexus sim".ljust(32, b"\0"),  # model_name
            0,  # firmware_version
            float("nan"),  # focal_length
            float("nan"),  # sensor_size_h
            float("nan"),  # sensor_size_v
            first.width,
            first.height,
            0,  # lens_id
            mav.CAMERA_CAP_FLAGS_HAS_VIDEO_STREAM,  # -> the GCS's checkForVideoStreams()
            0,  # cam_definition_version
            b"",  # cam_definition_uri: none, so no parameter download phase
        )

    def _send_camera_settings(self, msg) -> None:
        self._conn.mav.camera_settings_send(self._ms(), self._mode, float("nan"), float("nan"))

    def _set_mode(self, msg) -> None:
        self._mode = int(msg.param2)

    def _send_storage_information(self, msg) -> None:
        self._conn.mav.storage_information_send(
            self._ms(), 1, 1, mav.STORAGE_STATUS_NOT_SUPPORTED, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    def _send_capture_status(self, msg) -> None:
        self._conn.mav.camera_capture_status_send(self._ms(), 0, 0, 0.0, 0, 0.0)

    def _send_stream_information(self, msg) -> None:
        for s in self._selected(msg):
            self._conn.mav.video_stream_information_send(
                s.stream_id,
                len(self._streams),
                mav.VIDEO_STREAM_TYPE_RTSP,
                self._flags(s),
                s.fps,
                s.width,
                s.height,
                2_000_000,  # bitrate, cosmetic; the actual rate is the encoder's
                0,  # rotation
                90,  # hfov, cosmetic
                s.name.encode(),
                s.uri.encode(),
            )

    def _send_stream_status(self, msg) -> None:
        for s in self._selected(msg):
            self._conn.mav.video_stream_status_send(
                s.stream_id, self._flags(s), s.fps, s.width, s.height, 2_000_000, 0, 90
            )

    def _flags(self, s: Stream) -> int:
        # THERMAL is the GCS's routing key to its second receiver (PIP); see the module docstring.
        return mav.VIDEO_STREAM_STATUS_FLAGS_RUNNING | (mav.VIDEO_STREAM_STATUS_FLAGS_THERMAL if s.thermal else 0)


def serve(args) -> None:
    """Connect, retrying so bring-up order doesn't matter, then heartbeat + answer until killed."""
    # A stream is thermal because it really is thermal, an ir* MediaMTX path, or because --pip mislabels
    # the second one to borrow the GCS's two-feed receiver on a two-EO vehicle.
    streams = [
        parse_stream(spec, i, args.rtsp_base.rstrip("/"), thermal=(args.pip and i == 1) or spec.startswith("ir"))
        for i, spec in enumerate(args.streams)
    ]
    for s in streams:
        tag = " [thermal-flagged -> GCS PIP]" if s.thermal else ""
        print(f"[camera_advertiser] stream {s.stream_id}: {s.uri} {s.width}x{s.height}@{s.fps:g}{tag}", flush=True)
    while True:
        try:
            conn = mavutil.mavlink_connection(
                args.connect, source_system=args.sysid, source_component=mav.MAV_COMP_ID_CAMERA
            )
        except Exception as exc:
            print(f"[camera_advertiser] {args.connect} not reachable ({exc}); retrying in 2 s", flush=True)
            time.sleep(2.0)
            continue
        print(
            f"[camera_advertiser] connected to {args.connect} as sysid {args.sysid} comp {mav.MAV_COMP_ID_CAMERA}",
            flush=True,
        )
        adv = CameraAdvertiser(conn, streams, args.sysid)
        last_hb = 0.0
        try:
            while True:
                if time.monotonic() - last_hb >= 1.0:
                    last_hb = time.monotonic()
                    adv.heartbeat()
                msg = conn.recv_match(blocking=True, timeout=0.5)
                if msg is not None:
                    adv.handle(msg)
        except (ConnectionError, OSError) as exc:
            print(f"[camera_advertiser] link dropped ({exc}); reconnecting", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(2.0)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--streams",
        default="cam1,cam2",
        help="comma-separated MediaMTX paths, optionally NAME:WxH@FPS (default cam1,cam2; "
        "the GCS shows a stream-selector dropdown, one feed at a time). An ir* path is a real "
        "thermal feed and is advertised THERMAL on its own, so the GCS renders it beside the EO one",
    )
    parser.add_argument(
        "--pip",
        action="store_true",
        help="on a TWO-EO vehicle, render both feeds at once by mislabeling the SECOND stream "
        "thermal (the GCS's PIP/Full/Blend receiver slot; the UI then calls it 'Thermal View Mode', "
        "pixels untouched). Not needed with an ir* stream, which is genuinely thermal",
    )
    parser.add_argument(
        "--rtsp-base",
        default=None,
        help="advertised RTSP base URL; must be tablet-reachable (default rtsp://<LAN-IP>:8554)",
    )
    parser.add_argument(
        "--connect",
        default="tcp:127.0.0.1:5790",
        help="MAVLink fabric to join, pymavlink syntax (default the mavp2p GCS endpoint)",
    )
    parser.add_argument("--sysid", type=int, default=1, help="vehicle system id to attach the camera to (default 1)")
    args = parser.parse_args()
    args.streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    if not args.streams:
        parser.error("--streams is empty")
    if args.rtsp_base is None:
        args.rtsp_base = f"rtsp://{lan_ip()}:8554"
    try:
        serve(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
