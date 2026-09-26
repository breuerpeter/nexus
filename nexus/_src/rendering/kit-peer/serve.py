"""The Kit render peer: renders a nexus run's RTX sensors, fed poses over a socket by the host.

The host starts this program in the Kit image as ``python.sh serve.py --port <port>``. It listens
at once, boots Kit while the host builds its side of the run, and serves one connection:

1. on accept it sends ``hello``, so the host knows the peer is up;
2. ``setup`` names the scene, the vehicle, its start pose and the sensors, and the peer composes
   the stage, warms the renderer and answers ``ready``;
3. each ``frame`` carries the sim time and a world matrix per prim path, and the answer carries
   each due sensor's output, stamped with the sim time the render shows;
4. ``close``, or the host's socket closing, ends the program.

No nexus module is importable here: this folder ships in the wheel as package data and runs under
Kit's own Python, so the host and this program share only the wire format in ``link.py``.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time

import link
import numpy as np
from render import Renderer, log

# A host that dies before it connects leaves this peer waiting; after this long it exits instead.
ACCEPT_TIMEOUT_S = 1800.0
REPORT_EVERY = 240  # frames between two timing lines on the console: 10 s of a 24 Hz camera


def _accept(listener: socket.socket, box: dict, done: threading.Event) -> None:
    """Take the host's connection while Kit boots on the main thread, and greet it at once."""
    try:
        conn, _ = listener.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        link.send(conn, {"op": "hello"})
        box["conn"] = conn
    except OSError as exc:
        box["error"] = exc
    finally:
        listener.close()
        done.set()


def main() -> int:
    ap = argparse.ArgumentParser(description="Kit render peer")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    # SimulationApp reads sys.argv as Kit's own arguments.
    sys.argv = sys.argv[:1]

    listener = socket.create_server(("0.0.0.0", args.port))
    listener.settimeout(ACCEPT_TIMEOUT_S)
    box: dict = {}
    accepted = threading.Event()
    threading.Thread(target=_accept, args=(listener, box, accepted), daemon=True).start()

    t0 = time.time()
    from isaacsim import SimulationApp

    # Render-capable boot. multi_gpu off: the Gaussian-splat renderer, omni.rtx.spg, refuses to
    # activate when /renderer/multiGpu/enabled is true, a 6.0.1 validation, and splats render black.
    app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "multi_gpu": False},
        experience="/isaac-sim/apps/isaacsim.exp.full.kit",
    )
    log(f"Kit booted in {time.time() - t0:.1f}s")
    code = 0
    try:
        accepted.wait()
        conn = box.get("conn")
        if conn is None:
            log(f"no host connected ({box.get('error')!r}): exiting")
            return 1
        header, _ = link.recv(conn)
        if header.get("op") != "setup":
            raise RuntimeError(f"expected setup, got {header.get('op')!r}")
        try:
            renderer = Renderer(header)
            renderer.warm()
        except Exception as exc:
            link.send(conn, {"op": "error", "message": f"setup failed: {exc!r}"})
            raise
        link.send(conn, {"op": "ready"})
        log("ready")
        # Where each frame's wall time goes, summed and printed every REPORT_EVERY frames: waiting
        # for the host's request, rendering and reading back, and sending the reply.
        spent = {"wait": 0.0, "render": 0.0, "send": 0.0}
        frames = 0
        while True:
            t0 = time.perf_counter()
            try:
                header, blobs = link.recv(conn)
            except ConnectionError:
                log("the host closed the link")
                break
            if header["op"] == "close":
                shown, outputs = renderer.close()  # the last request's frame rides the closed reply
                arrays = [
                    {"sensor": i, "name": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}
                    for i, name, arr in outputs
                ]
                link.send(conn, {"op": "closed", "t": shown, "arrays": arrays}, [arr for _, _, arr in outputs])
                break
            t1 = time.perf_counter()
            mats = np.frombuffer(blobs[0], dtype=np.float64).reshape(-1, 4, 4)
            try:
                shown, outputs = renderer.frame(float(header["t"]), header["paths"], mats, header["due"])
            except Exception as exc:
                link.send(conn, {"op": "error", "message": f"render failed at t={header['t']}: {exc!r}"})
                raise
            t2 = time.perf_counter()
            arrays = [
                {"sensor": i, "name": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}
                for i, name, arr in outputs
            ]
            link.send(conn, {"op": "frame", "t": shown, "arrays": arrays}, [arr for _, _, arr in outputs])
            spent["wait"] += t1 - t0
            spent["render"] += t2 - t1
            spent["send"] += time.perf_counter() - t2
            frames += 1
            if frames % REPORT_EVERY == 0:
                per = ", ".join(f"{k} {v * 1000.0 / REPORT_EVERY:.1f}ms" for k, v in spent.items())
                update = renderer.update_s * 1000.0 / REPORT_EVERY
                log(f"frames {frames - REPORT_EVERY + 1}-{frames}, per frame: {per} (app.update {update:.1f}ms)")
                spent = dict.fromkeys(spent, 0.0)
                renderer.update_s = 0.0
    except Exception as exc:
        log(f"failed: {exc!r}")
        code = 1
    finally:
        # Kit's teardown can end the process outright, so everything reports before it.
        sys.stdout.flush()
        app.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
