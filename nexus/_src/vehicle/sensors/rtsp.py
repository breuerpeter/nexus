"""The NVENC H.264 to Real Time Streaming Protocol (RTSP) worker behind ``--stream``.

A newest-wins mailbox plus a background encoder thread, fault-isolated so RTSP backpressure never
blocks lockstep. The camera sensor owns one when the launch asks for a stream.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time

import numpy as np

from nexus._src.core import logger


class RtspPublisher:
    """The ``--stream`` consumer: Red Green Blue (RGB) frames → NVENC H.264 → RTSP, then → MediaMTX → WHEP.

    A newest-wins mailbox + a background encoder thread: RTSP/encoder backpressure can never block the
    lockstep thread. Fault-isolated: an encode failure warns once and the feed stops; flight continues.
    """

    def __init__(self, *, width: int, height: int, fps: int, bitrate: str, rtsp_url: str):
        # Per-camera stream at the CAMERA's authored resolution/rate, since the vehicle USD is the
        # authority; the transport knobs, URL base and bitrate, stay scenario config, RtxConfig.
        self.width, self.height, self.fps, self.bitrate = int(width), int(height), int(fps), str(bitrate)
        self.rtsp_url = rtsp_url
        self._failed = False
        self._closed = False
        self._render_n = 0  # frames pushed, for the video-fps readout
        self._t_first = None
        self._proc = self._start_ffmpeg()
        # newest-wins mailbox: drop stale frames if the encoder falls behind; never block lockstep.
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._encode_loop, name="rtx-encode", daemon=True)
        self._worker.start()
        logger.info(f"RtspPublisher: {self.width}x{self.height} @~{self.fps}fps → {self.rtsp_url}")

    def push(self, rgb) -> None:
        """Hand one (H,W,3) uint8 RGB frame to the encoder, drop-if-busy. Fault-isolated."""
        if self._failed or self._closed:
            return
        try:
            now = time.time()
            if self._t_first is None:
                self._t_first = now
            self._render_n += 1
            if self._render_n % 50 == 0:  # surface the live video fps
                logger.info(
                    f"RtspPublisher: video {self._render_n / max(1e-6, now - self._t_first):.1f} fps "
                    f"({self._render_n} frames)"
                )
            frame = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass
        except Exception as exc:
            self._failed = True
            logger.warning(f"RtspPublisher disabled after push error (flight continues): {exc!r}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        if self._proc:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        logger.info("RtspPublisher closed")

    def _start_ffmpeg(self) -> subprocess.Popen:
        c = self
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{c.width}x{c.height}",
            # frames arrive at ~render_hz·RTF, not a fixed rate; timestamp them by wall-clock
            # arrival so the WHEP stream plays at the true production rate; Constant Frame Rate (CFR)
            # -r would mistime it.
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            "-",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "ll",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            c.bitrate,
            "-bf",
            "0",
            "-g",
            str(c.fps),
            "-rtsp_transport",
            "tcp",
            "-f",
            "rtsp",
            self.rtsp_url,
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _encode_loop(self) -> None:
        # Background thread: the only part that can block on RTSP/encoder backpressure. Keeps
        # the ffmpeg stdin write off the lockstep thread.
        while True:
            frame = self._q.get()
            if frame is None:
                return
            if self._proc.poll() is not None:
                if not self._failed:
                    logger.warning(
                        f"RtspPublisher: ffmpeg exited rc={self._proc.returncode}; feed stopped (flight continues)"
                    )
                self._failed = True
                return
            try:
                self._proc.stdin.write(frame)
            except (BrokenPipeError, ValueError) as exc:
                if not self._failed:
                    logger.warning(f"RtspPublisher: encode write failed ({exc!r}); feed stopped (flight continues)")
                self._failed = True
                return
