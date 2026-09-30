"""The PX4 Software In The Loop (SITL) peer's container, on the peer contract of
:mod:`nexus._src.peers`.

The run hands the peer its instance, ``px4 -i``, and PX4 numbers every link from it with its own
compiled base ports, which :mod:`nexus._src.peers.px4_sitl` mirrors. This module passes PX4 no
port. The container runs on host networking, so those are the host's ports, and two runs on one
machine take two instances. Each run's container carries its own name, so one run's stop never
touches another's.

The image is the px4-sitl build toolchain, built on this machine from ``image/`` beside this module,
package data, and tagged with a hash of that folder. The PX4 tree comes from :mod:`.checkout`: the
pinned commit fetched into a cache folder, or ``$PX4_DIR``. ``start`` builds the tree incrementally
before it launches, since a launch must reach the sim inside its 30 s preroll window and a cold
build never does.

This module is the one definition of the PX4 SITL container. It used to be one of three: a compose
service and a CI shell script held their own copies, which drifted, see GH #86. So a change to the
mounts, the user or the environment belongs here and nowhere else.

No flight logic lives here: the flight is always a script against ``sim.operator``; see
:mod:`nexus.examples.controllers.px4.flight`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

from nexus._src.core import logger
from nexus._src.peers import LABEL, containers
from nexus._src.peers.containers import ensure_image, run_container, stop_container

from . import checkout
from .instance import claim as _claim

IMAGE = "nexus-px4-sitl"
_INSTANCE = "nexus.px4.instance"  # the label that carries the container's PX4 instance
_OWNER = "nexus.owner"  # the label that carries the process that started the container
IMAGE_DIR = Path(__file__).with_name("image")
PX4_LOG_DIR = os.path.expanduser("~/.cache/nexus/logs")  # beside the run's .rrd and the Kit console tee


def _container_kwargs(tree: Path) -> dict:
    """The shared run kwargs for a PX4 SITL container over ``tree``: image, user, mounts, workdir.
    Callers add naming and the command on top.

    The user is the checkout's owner, not this process: a process running as root, building as
    uid 0, would leave the checkout's ``build/`` tree root-owned and break the next build as its
    owner. For a process the owner runs, that's the same value ``os.getuid()`` would give.
    """
    st = tree.stat() if tree.exists() else None
    uid, gid = (st.st_uid, st.st_gid) if st is not None else (os.getuid(), os.getgid())
    return {
        "image": ensure_image(IMAGE_DIR, IMAGE),
        "user": f"{uid}:{gid}",
        "environment": {"HOME": "/tmp"},
        "working_dir": str(tree),
        "volumes": {str(tree): {"bind": str(tree), "mode": "rw"}},
    }


def build(tree: Path) -> None:
    """``make px4_sitl`` over ``tree``, a build and no launch: incremental, so a second call is a
    no-op that fits the sim's preroll window while a cold build takes minutes.

    Raises:
        RuntimeError: There is no PX4 tree at ``tree``, or the docker daemon is unreachable.
        docker.errors.ContainerError: The build itself failed.
    """
    if not (tree / "Makefile").is_file():
        # Without this, docker creates the empty mount and `make` reports "No rule to make target
        # 'px4_sitl'" as the error, which reads as a broken image rather than a missing checkout.
        raise RuntimeError(f"no PX4 tree at {tree}: unset $PX4_DIR to fetch the pinned one, or point it at a checkout")
    logger.info(f"building PX4 SITL in {tree} (incremental; a cold build takes minutes) …")
    # `--rm` on a FOREGROUND run is the client removing the container after it exits, which is what
    # `remove` is. The daemon-side `auto_remove` races the client here: the container disappears
    # before the client can read its logs, and every build, passing or failing, dies on a 404 instead.
    out = run_container(command=["make", "px4_sitl"], detach=False, remove=True, **_container_kwargs(tree))
    if out:  # the build log, as `subprocess.run` used to stream it; a failure raises with its stderr
        print(out.decode(errors="replace"), flush=True)


def prepare(catalog: Path | None = None) -> Path:
    """Resolve the PX4 tree and build it, the way a run's first use does, and return the tree.

    For a caller that wants the cold work done outside a run's own budget: the CI harness before a
    timed example, or a machine's first setup.
    """
    tree = checkout.tree(catalog)
    build(tree)
    return tree


class Px4Sitl:
    """One PX4 SITL container for one run, on the peer contract.

    Args:
        catalog: The project catalog the run resolved against, whose pin file wins over the shipped
            one; the start resolves the PX4 tree from it through :func:`checkout.tree`.
        airframe: The airframe's make target without PX4's ``none_`` prefix, ``astro_max``.
        instance: PX4's SITL instance, which numbers its ports and its system id.
        name: The container's name, one per run.
        log_path: Where the container's console streams to, the ``px4_log`` artifact.
        claim: The open lock file that holds ``instance`` for this run, which the stop releases; ``None``
            for a peer whose instance no lock holds.
    """

    def __init__(
        self, *, catalog: Path | None, airframe: str, instance: int, name: str, log_path: str, claim: IO | None = None
    ) -> None:
        self.catalog = catalog
        self.tree: Path | None = None  # resolved at start, which can fetch it
        self.airframe = airframe
        self.instance = instance
        self.name = name
        self.log_path = log_path
        self._claim = claim
        self._container = None

    def start(self) -> None:
        """Fetch the PX4 tree, then build the image and the tree if this machine lacks them, then start
        PX4; it boots in the background and dials the sim's Hardware In The Loop (HIL) server.

        Raises:
            RuntimeError: A live process holds this instance, or there is no PX4 tree, or the image
                build failed, or the docker daemon is unreachable.
            docker.errors.ContainerError: The PX4 build failed.
        """
        self._clear_leftovers()
        self.tree = checkout.tree(self.catalog)
        build(self.tree)
        kwargs = _container_kwargs(self.tree)
        # PX4's own `make px4_sitl none_<airframe>` runs the binary with the airframe in its
        # environment; the instance is what that target can't take, so the peer runs the binary
        # itself the way PX4's multi-instance script does. `-d` runs with no pxh shell: at the end of
        # an unattended stdin that shell spin-loops printing its prompt and starves PX4.
        kwargs["environment"]["PX4_SIM_MODEL"] = f"none_{self.airframe}"
        logger.info(
            f"launching PX4 SITL {self.name} (none_{self.airframe}, instance {self.instance}) -> {self.log_path}"
        )
        self._container = run_container(
            command=["build/px4_sitl_default/bin/px4", "-i", str(self.instance), "-d"],
            name=self.name,
            log_path=self.log_path,
            network_mode="host",
            auto_remove=True,
            labels={LABEL: "px4", _INSTANCE: str(self.instance), _OWNER: str(os.getpid())},
            **kwargs,
        )

    @staticmethod
    def claim_instance() -> tuple[int, IO]:
        """The lowest PX4 instance free on this machine and the lock file that holds it, through
        :func:`~nexus._src.peers.px4_sitl.instance.claim`, passing over each instance a live process's container holds.
        """
        held = containers.client().containers.list(all=True, filters={"label": [f"{LABEL}=px4"]})
        live = {int(c.labels[_INSTANCE]) for c in held if (owner := int(c.labels.get(_OWNER) or 0)) and _alive(owner)}
        return _claim(skip=live)

    def _clear_leftovers(self) -> None:
        """Remove a PX4 container of this instance whose process has exited, and fail while a live one
        holds the instance.

        PX4 SITL outlives a run killed without its teardown, a closed terminal or a harness's
        timeout, and keeps dialing the sim's port. Each run's container has a name of its own, so no
        later start would clear it by name. Two live PX4s on one instance would share every port, so
        a live holder fails this start instead of losing its autopilot.
        """
        from docker.errors import NotFound

        # Through the module, not a name imported here, so a stand-in daemon a test patches in reaches it.
        held = containers.client().containers.list(
            all=True, filters={"label": [f"{LABEL}=px4", f"{_INSTANCE}={self.instance}"]}
        )
        for other in held:
            owner = int(other.labels.get(_OWNER) or 0)
            if owner and _alive(owner):
                raise RuntimeError(
                    f"PX4 instance {self.instance} is in use by process {owner}, container {other.name}: "
                    "another run took it as this one started; start this run again"
                )
            logger.info(f"removing the leftover PX4 container {other.name}: process {owner} has exited")
            try:
                other.remove(force=True)
            except NotFound:
                pass

    def alive(self) -> bool:
        """Whether the container still runs; it removes itself when it exits."""
        from docker.errors import NotFound

        if self._container is None:
            return False
        try:
            self._container.reload()
        except NotFound:
            return False
        return self._container.status in ("created", "running")

    def stop(self) -> None:
        """Remove the container this peer started and release its instance. Idempotent."""
        if self._claim is not None:
            self._claim.close()  # closing the file releases its lock, and the instance is free again
            self._claim = None
        if self._container is None:
            return
        stop_container(self.name)
        self._container = None

    def artifacts(self) -> dict:
        """The PX4 console log of this run, the ``px4_log`` artifact the PX4 warnings gate reads."""
        return {"px4_log": self.log_path}


def _alive(pid: int) -> bool:
    """Whether process ``pid`` runs on this machine; one of another user's counts."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = ["IMAGE", "IMAGE_DIR", "PX4_LOG_DIR", "Px4Sitl", "build", "prepare"]
