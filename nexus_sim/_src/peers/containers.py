"""The one place this repo runs a container it owns, through the docker daemon's Python SDK.

A peer the flight starts and stops as part of its own lifecycle runs from here, so the ordering a
runbook used to write as a warning becomes code: PX4 Software In The Loop (SITL) from
:mod:`nexus_sim._src.peers.px4_sitl.runner`, and the Kit render peer from
:mod:`nexus_sim._src.peers.kit.runner`.

The px4-sitl image, PX4's build toolchain, builds on the machine that runs it, from a folder that
ships in the wheel as package data. Its tag hashes that folder, so a change to the image always
rebuilds it and a host code change never does. No registry and no login sit between a pip install
and a first PX4 run. The Kit render peer runs NVIDIA's image as pulled instead, see
:mod:`nexus_sim._src.peers.kit.runner`.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nexus_sim._src.core import logger

if TYPE_CHECKING:
    from docker.models.containers import Container


_client = None


def client():
    """The docker daemon client for this process, created once and then reused.

    Returns:
        docker.DockerClient: A client connected to the daemon, from ``docker.from_env()``.

    Raises:
        RuntimeError: The daemon is unreachable; the message names its address and the fix.
    """
    global _client
    if _client is not None:
        return _client
    import docker
    from docker.errors import DockerException

    try:
        _client = docker.from_env()
        _client.ping()
    except DockerException as exc:
        # The address as a URL, never a bare path: a missing daemon has no socket file to name.
        host = os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock"
        raise RuntimeError(
            f"cannot reach the Docker daemon at {host}: start it, or point DOCKER_HOST at one that runs"
        ) from exc
    return _client


def _pump_logs(container: Container, log_path: str) -> None:
    """Stream the container's console into ``log_path`` on a daemon thread, so the peer's output is a
    file the caller can scan after the run; the PX4 warnings gate reads it.
    """

    def run() -> None:
        try:
            with open(log_path, "wb") as sink:
                for chunk in container.logs(stream=True, follow=True):
                    sink.write(chunk)
                    sink.flush()
        except Exception:
            pass  # output-only: a container that dies mid-stream must not raise on a daemon thread

    threading.Thread(target=run, daemon=True).start()


def run_container(
    *,
    image: str,
    command: list[str] | None = None,
    name: str | None = None,
    log_path: str | None = None,
    detach: bool = True,
    **kwargs: Any,
) -> Container:
    """Run one container this repo owns, clearing any leftover of the same name first.

    A container killed with its creating process survives as a name squatter that also holds its
    ports, so clearing on start is what makes a relaunch reliable rather than something a doc has to
    warn about.

    Args:
        image: The image to run.
        command: The container's command, or ``None`` for the image's own.
        name: The container name; when given, a same-name leftover is force-removed first.
        log_path: When given, and only when detached, stream the container's console into this file.
        detach: ``True`` returns once the container starts; ``False`` runs it to completion
            and raises on a non-zero exit.
        **kwargs: Passed through to ``containers.run``: ``user``, ``volumes``, ``environment``, and so on.

    Returns:
        docker.models.containers.Container: The started, or completed, container.

    Raises:
        RuntimeError: The daemon is unreachable; see :func:`client`.
        docker.errors.ContainerError: ``detach=False`` and the container exited non-zero.
    """
    c = client()
    if name is not None:
        stop_container(name)
    container = c.containers.run(image, command=command, name=name, detach=detach, **kwargs)
    if detach and log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        _pump_logs(container, log_path)
    return container


def stop_container(name: str) -> None:
    """Force-remove the container called ``name``. Idempotent: a container that isn't there counts as success.

    Force-removal is the whole teardown: the container is a child of the daemon, not of this
    process, so there is no client to signal.
    """
    from docker.errors import NotFound

    try:
        client().containers.get(name).remove(force=True)
    except NotFound:
        pass


def _build_files(root: Path) -> list[Path]:
    """The files an image build reads: the folder, minus the bytecode an installer leaves beside it."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.relative_to(root).parts and p.suffix != ".pyc"
    )


def image_tag(root: Path, name: str) -> str:
    """The tag of the image built from ``root``: ``name:`` and a hash of the files the build reads.

    An installed package and a checkout hold the same folder at different paths, so the hash covers
    each file's relative path and content, never its location.
    """
    h = hashlib.sha256()
    for path in _build_files(root):
        data = path.read_bytes()
        h.update(f"{path.relative_to(root).as_posix()}\0{len(data)}\0".encode())
        h.update(data)
    return f"{name}:{h.hexdigest()[:12]}"


def ensure_image(root: Path, name: str) -> str:
    """Build the image from ``root`` when this machine lacks its tag, and return the tag.

    The build prints its own output as it goes.

    Raises:
        RuntimeError: The build failed, with the build's own error, or the daemon is unreachable.
    """
    from docker.errors import ImageNotFound

    tag = image_tag(root, name)
    c = client()
    try:
        c.images.get(tag)
        return tag
    except ImageNotFound:
        pass
    logger.info(f"building the image {tag} from {root}: once per machine and per change to that folder")
    for chunk in c.api.build(path=str(root), tag=tag, rm=True, decode=True):
        if "error" in chunk:
            raise RuntimeError(f"building the image {tag} failed: {chunk['error'].strip()}")
        line = chunk.get("stream")
        if line:
            sys.stderr.write(line)
            sys.stderr.flush()
    logger.info(f"built the image {tag}")
    return tag
