#!/usr/bin/env python3
# Acronyms: Ground Control Station (GCS).
"""Smoke-check the GCS MAVLink link (mavp2p TCP:5790 -> PX4 SITL).

Connects as a GCS over TCP, sends a heartbeat so PX4 learns the peer, then
asserts a PX4 HEARTBEAT and an ATTITUDE arrive within the timeout. Exit 0 on
success, 1 on timeout. Usage: uv run python tools/check_gcs_link.py [host:port]
"""

import sys
import time

from pymavlink import mavutil

addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5790"
host, port = addr.split(":")
m = mavutil.mavlink_connection(f"tcp:{host}:{port}", source_system=255, source_component=190)

# Send a GCS heartbeat so PX4's :18570 instance learns the route to this peer and streams back.
m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

deadline = time.time() + 15.0
seen = set()
while time.time() < deadline and not {"HEARTBEAT", "ATTITUDE"} <= seen:
    msg = m.recv_match(type=["HEARTBEAT", "ATTITUDE"], blocking=True, timeout=2.0)
    if msg is None:
        m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        continue
    seen.add(msg.get_type())
    print(f"recv {msg.get_type()} from sys={msg.get_srcSystem()}")

if {"HEARTBEAT", "ATTITUDE"} <= seen:
    print("OK: GCS link carries PX4 telemetry over TCP:5790")
    sys.exit(0)
print(f"FAIL: only saw {seen or 'nothing'} within 15s")
sys.exit(1)
