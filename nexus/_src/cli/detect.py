"""Universal Scene Description (USD) driven runtime selection: the user never thinks about runtimes.
Import-light: pxr + stdlib.

The vehicle USD is the single authority for what the vehicle IS, including its sensors. RTX sensors,
camera, lidar and radar prims, need Kit to produce their measurement, so their presence in the USD *is*
the runtime decision: RTX sensors → the isaacsim runtime, auto-launching the container from the host;
none → the lean standalone runtime. No ``--render`` flag: a camera in the USD means "this vehicle has
a camera sensor," and a sensor that exists, measures.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time


def resolve_runtime(args) -> str:
    """Resolve ``--runtime auto`` from the vehicle USD: RTX sensor prims → ``isaacsim``, else
    ``standalone``. Resolves the vehicle through the run's own catalog, cached and sha-verified, to get
    the USD, so the probe and the run that follows it read the same registry.
    """
    if args.runtime != "auto":
        return args.runtime
    from nexus._src.config import LaunchConfig
    from nexus._src.runtimes.launch import resolve_to_vehicle_builder
    from nexus._src.vehicle.sensors.usd import vehicle_rtx_sensor_prims

    launch = LaunchConfig().set_vehicle(args.vehicle)  # registry name / local .usd path / None
    launch.registry = getattr(args, "registry", None)  # the probe reads the catalog the run flies
    builder, _ = resolve_to_vehicle_builder(launch)
    prims = vehicle_rtx_sensor_prims(builder.cfg["usd_path"])
    if prims:
        from nexus._src.core import logger

        logger.info(f"runtime=auto: RTX sensor prims in the vehicle USD {prims} -> isaacsim")
        return "isaacsim"
    return "standalone"


def repo_root() -> str:
    """This checkout's root -- the directory holding ``nexus/`` and ``docker/``.

    From this module's path, not the process cwd: the launch reads the compose file out of
    ``docker/`` and bind-mounts the checkout as ``NEXUS_DIR``, and neither is a property of
    where the user stood. From a subdirectory the cwd version found no compose file at all.
    """
    return str(pathlib.Path(__file__).resolve().parents[3])


def launch_isaacsim_container(argv: list[str], *, pin_runtime: bool = True) -> int:
    """Run the isaacsim runtime for this invocation by launching the Kit container with docker compose.

    The host env can't import Kit, so "this vehicle needs isaacsim" means running the same command-line
    tool inside the container: the repo is bind-mounted, and the compose service's entrypoint runs it
    under Kit's Python. ``pin_runtime`` forwards the caller's args with ``--runtime isaacsim`` pinned,
    which is the ``run`` path; the ``script`` path forwards verbatim, since the target owns its args.
    Raises a clear error with the exact command when docker or the compose file is unavailable.
    """
    compose = os.path.join(repo_root(), "docker", "docker-compose.yml")
    if pin_runtime:
        fwd = [a for a in argv if a not in ("--runtime", "auto", "isaacsim", "standalone")]
        cmd = ["docker", "compose", "-f", compose, "run", "--rm", "isaacsim", *fwd, "--runtime", "isaacsim"]
    else:
        cmd = ["docker", "compose", "-f", compose, "run", "--rm", "isaacsim", *argv]
    if shutil.which("docker") is None or not os.path.exists(compose):
        raise SystemExit(
            "this vehicle's USD has RTX sensors, which need the isaacsim (Kit) runtime, "
            f"run it in the container:\n  {' '.join(cmd)}"
        )
    from nexus._src.core import logger

    logger.info(f"runtime=isaacsim (from the vehicle USD) -> launching the Kit container: {' '.join(cmd)}")
    # The compose file interpolates ${NEXUS_DIR} for the bind mount + workdir; point it at this
    # checkout so auto-launch needs nothing exported.
    env = {**os.environ, "NEXUS_DIR": os.environ.get("NEXUS_DIR") or repo_root()}
    # Tee the container console to a host-side log next to the run's .rrd: `docker run --rm`
    # discards the container, and Kit's log dir isn't mounted, which left the 2026-07-22 GH #32
    # recurrence report with no sim-side evidence. The console is the one durable record of the
    # in-sim warnings: freeze detectors, real-time factor, PX4 link.
    log_dir = os.path.expanduser("~/.cache/nexus/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"console-{time.strftime('%Y%m%d-%H%M%S')}.log")
    logger.info(f"container console -> {log_path}")
    proc = subprocess.Popen(cmd, stdin=sys.stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    with open(log_path, "wb") as sink:
        try:
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                sink.write(chunk)
                sink.flush()
        except KeyboardInterrupt:
            pass  # Ctrl-C goes to the process group; keep the tail below
        finally:
            # Drain whatever the shutting-down container still prints, since shutdown errors are
            # exactly what post-mortems need. Bounded so a wedged container can't hang this process.
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
            if proc.poll() is not None and proc.stdout is not None:
                try:
                    tail = proc.stdout.read()
                    if tail:
                        sys.stdout.buffer.write(tail)
                        sys.stdout.buffer.flush()
                        sink.write(tail)
                except Exception:
                    pass
    return proc.returncode if proc.returncode is not None else 130
