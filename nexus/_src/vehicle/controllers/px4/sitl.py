"""PX4 Software In The Loop (SITL) bring-up: launch the PX4 autopilot container against a sim serving
:4560, and tear it down again. It lives beside the controller because PX4 is the one first-class
controller, so core must be able to start the autopilot it talks to.

No flight logic lives here: the flight is always a script against ``sim.operator``; see
:mod:`nexus.examples.controllers.px4.flight`.

This module is the one definition of the PX4 SITL container. It used to be one of three: a compose
service and a CI shell script held their own copies, which drifted, see GH #86. So a change to the
mounts, the user or the environment belongs here and nowhere else.

The checkout and the image are machine-description facts, so they stay env vars, which keeps them
settable from outside Python. Both resolve per call, so setting ``PX4_DIR`` or ``PX4_IMAGE`` after
the import of this module still takes effect.
"""

from __future__ import annotations

import os
from pathlib import Path

from nexus._src.containers import IMAGE_PREFIX, run_container, stop_container

CONTAINER = "nexus-px4-sitl"  # the container: one PX4 at a time, since 4560/18570/14550 are singletons
PX4_LOG_DIR = os.path.expanduser("~/.cache/nexus/logs")  # beside the run's .rrd and the Kit console tee


def px4_dir() -> Path:
    """The PX4-Autopilot checkout: ``$PX4_DIR``, default ``~/code/px4``."""
    return Path(os.environ.get("PX4_DIR") or Path.home() / "code" / "px4")


def px4_image() -> str:
    """The px4-sitl docker image: ``$PX4_IMAGE``, else the px4-sitl under :data:`IMAGE_PREFIX`."""
    return os.environ.get("PX4_IMAGE") or f"{IMAGE_PREFIX}/px4-sitl:latest"


def _container_kwargs() -> dict:
    """The shared run kwargs for the PX4 SITL container: image, user, mounts, workdir. Callers add
    naming/networking/env on top.

    The user is the checkout's owner, not this process: a process running as root, building as
    uid 0, would leave the checkout's ``build/`` tree root-owned and break the next build as its
    owner. For a process the owner runs, that's the same value ``os.getuid()`` would give.
    """
    px4 = px4_dir()  # resolved per call, not at import: a test, or a caller, can set $PX4_DIR later
    st = px4.stat() if px4.exists() else None
    uid, gid = (st.st_uid, st.st_gid) if st is not None else (os.getuid(), os.getgid())
    return {
        "image": px4_image(),
        "user": f"{uid}:{gid}",
        "environment": {"HOME": "/tmp"},
        "working_dir": str(px4),
        "volumes": {str(px4): {"bind": str(px4), "mode": "rw"}},
        "auto_remove": True,
    }


def build_px4_sitl() -> None:
    """Build-only warm-up: ``make px4_sitl`` with no airframe, so a build and no launch. Run once
    before a batch of :meth:`Px4Sitl.launch` calls: a launch's own make must finish inside the sim's
    30 s preroll window, GH #39, which an incremental no-op fits and a cold build never does.

    Raises:
        RuntimeError: There is no PX4 checkout at ``px4_dir()``, and the message names the fix, or the
            docker daemon is unreachable.
        docker.errors.ContainerError: The build itself failed.
    """
    px4 = px4_dir()
    if not px4.is_dir():
        # Without this, docker creates the empty mount and `make` reports "No rule to make target
        # 'px4_sitl'" as the error, which reads as a broken image rather than a missing checkout.
        raise RuntimeError(
            f"no PX4 checkout at {px4}: clone PX4-Autopilot there (git clone --recursive "
            f"https://github.com/PX4/PX4-Autopilot {px4}) or set $PX4_DIR"
        )
    kwargs = _container_kwargs()
    # `--rm` on a FOREGROUND run is the client removing the container after it exits, which is what
    # `remove` is. The daemon-side `auto_remove` races the client here: the container disappears
    # before the client can read its logs, and every build, passing or failing, dies on a 404 instead.
    kwargs.pop("auto_remove")
    out = run_container(command=["make", "px4_sitl"], detach=False, remove=True, **kwargs)
    if out:  # the build log, as `subprocess.run` used to stream it; a failure raises with its stderr
        print(out.decode(errors="replace"), flush=True)


class Px4Sitl:
    """One PX4 SITL docker container on host networking: it dials the in-process Newton sim's
    Hardware In The Loop (HIL) server on :4560 and streams MAVLink: Ground Control Station (GCS) on
    :18570, offboard on :14540. ``stop()`` is idempotent, and ``launch()`` clears a stale same-name
    container first, since a leftover squats the GCS ports and would fail the next run spuriously.
    """

    def __init__(self, *, container: str, log_path: str, airframe: str = "none_astro_max") -> None:
        self.container = container
        self.log_path = log_path
        self.airframe = airframe
        self._started = False

    def launch(self) -> None:
        kwargs = _container_kwargs()
        # Claim ownership BEFORE the create, so a launch that dies half-way, container created but
        # start failed, still gets cleaned up by this instance's own stop().
        self._started = True
        run_container(
            command=["make", "px4_sitl", self.airframe],
            name=self.container,
            log_path=self.log_path,
            network_mode="host",
            # stdin_open, as in docker run -i, never written and never attached: at the stdin
            # End Of File (EOF) the pxh shell spin-loops printing its prompt, GBs of "pxh>" spam that
            # starves PX4; an open but silent stdin makes it block cleanly. Verified with no client attached: the
            # daemon holds the pipe open, since StdinOnce is false by default and the SDK's run() has
            # no kwarg for it, so a container blocked on a read stays blocked instead of seeing EOF.
            stdin_open=True,
            **kwargs,
        )

    def stop(self) -> None:
        """Stop the PX4 this instance started. A no-op if it never launched one.

        The name is a singleton, so a run that never got as far as launching must not remove it:
        the container of that name then belongs to somebody else's flight. A second sim on one host
        fails at its own ``:4560`` bind, because the port is busy, and its teardown would otherwise stop
        the healthy run's autopilot. Clearing a stale container is ``launch()``'s job, not this one's.
        """
        if not self._started:
            return
        stop_container(self.container)
        self._started = False
