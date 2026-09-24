"""The one place this repo runs a container it owns, through the docker daemon's Python SDK.

The governing rule for the two ways this repo talks to docker: **the SDK where this repo owns the
container definition, compose where compose owns it.** A peer the flight starts and stops as part of
its own lifecycle runs from here, so the ordering a runbook used to write as a warning becomes code:
PX4 Software In The Loop (SITL) from :mod:`nexus._src.vehicle.controllers.px4.sitl`, and the Kit
render peer from :mod:`nexus._src.rendering.peer`. Compose keeps only the ground services a runbook
brings up by hand, in ``docker/docker-compose.yml``.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docker.models.containers import Container

# The registry path the project's images live under: ``docker-images.yml`` pushes them there.
IMAGE_PREFIX = "ghcr.io/breuerpeter/nexus"


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
    from docker.errors import DockerException

    import docker

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
