#!/usr/bin/env bash
# Publish a WebRTC-safe H.264 test pattern to MediaMTX path cam1 (the A2 M1 video
# transport, standing in for the Isaac chase-cam). Requires ffmpeg on the host.
# Usage: tools/video_testpattern.sh [rtsp_base]   (default rtsp://127.0.0.1:8554)
set -euo pipefail
RTSP_BASE="${1:-rtsp://127.0.0.1:8554}"
exec ffmpeg -hide_banner -re \
  -f lavfi -i "testsrc2=size=1280x720:rate=30" \
  -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p \
  -bf 0 -g 30 -tune zerolatency \
  -f rtsp "${RTSP_BASE}/cam1"
