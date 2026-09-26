"""A stand-in docker daemon for the Kit peer's tests: the daemon is the system boundary here."""

import threading
from pathlib import Path

import pytest
from docker.errors import NotFound

import nexus._src.containers as containers
import nexus._src.rendering.peer as peer


class _Container:
    def __init__(self, daemon, name):
        self._daemon = daemon
        self.name = name
        self.status = "running"

    def reload(self):
        pass

    def logs(self, **kwargs):
        yield from ()

    def wait(self):
        return {"StatusCode": 0}

    def remove(self, force=False):
        self.status = "removed"
        self._daemon.removed.append(self.name)


class Daemon:
    """Every image exists; a started container records its spec, and ``serve`` runs in its place.

    ``serve(port)`` stands in for the program the container runs: the peer's tests pass one that
    speaks the render link on the port the host chose.
    """

    def __init__(self):
        self.runs: list[dict] = []
        self.removed: list[str] = []
        self.serve = None
        daemon = self

        class Images:
            def get(self, tag):
                return object()

        class Containers:
            def run(self, image, **kwargs):
                seen = {src: (Path(src).is_dir(), Path(src).stat().st_uid) for src in kwargs.get("volumes", {})}
                daemon.runs.append({"image": image, **kwargs, "seen": seen})
                container = _Container(daemon, kwargs.get("name"))
                if daemon.serve is not None:
                    port = int(kwargs["command"][kwargs["command"].index("--port") + 1])
                    threading.Thread(target=daemon.serve, args=(port,), daemon=True).start()
                daemon.last = container
                return container

            def get(self, name):
                raise NotFound(f"no container {name}")

        self.images = Images()
        self.containers = Containers()


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    """The stand-in daemon, with the home folder in the test's folder so the peer's caches land there."""
    d = Daemon()
    monkeypatch.setattr(containers, "client", lambda: d)
    monkeypatch.setattr(peer, "client", lambda: d)  # the peer module holds the name it imported
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return d
