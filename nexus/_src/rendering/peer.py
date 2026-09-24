"""The Kit render peer's container: its image, what it mounts, and its start, stop and alive.

The Kit image can't be a public package, since it holds NVIDIA's layers, so each machine builds it
on its first RTX run from ``kit-peer/``, package data beside this module. The tag is a hash of that
folder, the only files the build reads, so a host code change never rebuilds the image and a change
to the peer program always does. The container runs through the docker SDK, as PX4 Software In The
Loop (SITL) does.

The container runs as the host user and mounts only what the run renders and the caches it writes:
the asset cache and the folder of each local Universal Scene Description (USD) file read-only at their host paths, and Kit's own
caches and the logs folder read-write, each created as the user first. Docker creates a missing
bind source owned by root, and a root-owned asset cache failed every later host run.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import time
from pathlib import Path

from nexus._src.containers import client, run_container, stop_container
from nexus._src.core import logger

KIT_DIR = Path(__file__).with_name("kit-peer")
IMAGE = "nexus-kit"
LABEL = "nexus.peer"  # a Kit peer carries nexus.peer=kit, so a runner can find a leftover one
ISAAC_SIM_GID = "1234"  # the base image's isaac-sim group: /isaac-sim is readable by that group only
_KIT_HOME = "/kit-home"


class KitPeerError(RuntimeError):
    """The Kit render peer couldn't start; the message names the cause."""


def _build_files(root: Path) -> list[Path]:
    """The files the image build reads: the folder, minus what ``.dockerignore`` leaves out."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.relative_to(root).parts and p.suffix != ".pyc"
    )


def image_tag(root: Path = KIT_DIR) -> str:
    """The Kit image's tag: ``nexus-kit:`` and a hash of the files the build reads under ``root``."""
    h = hashlib.sha256()
    for path in _build_files(root):
        data = path.read_bytes()
        h.update(f"{path.relative_to(root).as_posix()}\0{len(data)}\0".encode())
        h.update(data)
    return f"{IMAGE}:{h.hexdigest()[:12]}"


def ensure_image() -> str:
    """Build the Kit image when this machine lacks its tag, and return the tag.

    The build pulls NVIDIA's base, about 21 GB, with no login, adds Cesium and the peer program,
    and prints its own output as it goes.

    Raises:
        KitPeerError: The build failed; the message carries the build's own error.
        RuntimeError: The docker daemon is unreachable; see :func:`nexus._src.containers.client`.
    """
    from docker.errors import ImageNotFound

    tag = image_tag()
    c = client()
    try:
        c.images.get(tag)
        return tag
    except ImageNotFound:
        pass
    logger.info(f"building the Kit image {tag} from {KIT_DIR}: once per machine and per peer change")
    for chunk in c.api.build(path=str(KIT_DIR), tag=tag, rm=True, decode=True):
        if "error" in chunk:
            raise KitPeerError(f"building the Kit image {tag} failed: {chunk['error'].strip()}")
        line = chunk.get("stream")
        if line:
            sys.stderr.write(line)
            sys.stderr.flush()
    logger.info(f"built the Kit image {tag}")
    return tag


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class KitPeer:
    """One Kit render peer container for one run.

    Args:
        files: Every USD file the run renders, vehicle and scene: the peer reads each at its host path.
        cache_dir: The asset cache the run fetched into; mounted whole when a file lies inside it.
    """

    def __init__(self, files: list, *, cache_dir):
        self._files = [Path(f).resolve() for f in files if f]
        self._cache_dir = Path(cache_dir).resolve()
        self.name = f"nexus-kit-{os.getpid()}"
        self.port: int | None = None
        self.log_path: Path | None = None
        self._container = None

    def _read_mounts(self) -> dict:
        mounts: list[Path] = []
        for f in self._files:
            folder = self._cache_dir if f.is_relative_to(self._cache_dir) else f.parent
            if folder not in mounts:
                mounts.append(folder)
        return {str(m): {"bind": str(m), "mode": "ro"} for m in mounts}

    def start(self) -> None:
        """Build the image if this machine lacks it, then start the container; Kit boots in the background.

        Raises:
            KitPeerError: The image build or the container start failed, or the docker daemon is
                unreachable; the message names the cause.
        """
        from docker.errors import DockerException
        from docker.types import DeviceRequest

        try:
            tag = ensure_image()
        except KitPeerError:
            raise
        except RuntimeError as exc:
            raise KitPeerError(f"the RTX sensors render in a Kit container, but {exc}") from exc
        logs = Path.home() / ".cache" / "nexus" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        volumes = {
            **self._read_mounts(),
            **_kit_caches(),
            str(logs): {"bind": str(logs), "mode": "rw"},  # the --benchmark JSON beside the run's .rrd
        }
        env = {"HOME": _KIT_HOME, "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"}
        if os.environ.get("CESIUM_ION_TOKEN"):
            env["CESIUM_ION_TOKEN"] = os.environ["CESIUM_ION_TOKEN"]
        self.port = _free_port()
        self.log_path = logs / f"console-{time.strftime('%Y%m%d-%H%M%S')}.log"
        try:
            self._container = run_container(
                image=tag,
                command=["/nexus-kit/serve.py", "--port", str(self.port)],
                name=self.name,
                log_path=str(self.log_path),
                user=f"{os.getuid()}:{os.getgid()}",
                group_add=[ISAAC_SIM_GID],
                environment=env,
                volumes=volumes,
                ports={f"{self.port}/tcp": ("127.0.0.1", self.port)},
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                labels={LABEL: "kit"},
                auto_remove=True,
            )
        except DockerException as exc:
            raise KitPeerError(
                f"starting the Kit container failed: {getattr(exc, 'explanation', None) or exc}"
            ) from exc
        logger.info(f"Kit render peer {self.name} booting on :{self.port}; its console -> {self.log_path}")

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
        """Remove the container. Idempotent."""
        if self._container is not None:
            stop_container(self.name)
            self._container = None

    def console_tail(self, lines: int = 20) -> str:
        """The last ``lines`` of the container's console, for an error message."""
        try:
            text = self.log_path.read_text(errors="replace") if self.log_path else ""
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])


def _kit_caches() -> dict:
    """Kit's own caches, created as the user first, as read-write mounts."""
    nexus_cache = Path.home() / ".cache" / "nexus"
    kit_home, kit_cache = nexus_cache / "kit" / "home", nexus_cache / "kit" / "cache"
    for folder in (kit_home, kit_cache):
        folder.mkdir(parents=True, exist_ok=True)
    return {
        str(kit_home): {"bind": _KIT_HOME, "mode": "rw"},  # Kit's user caches: compute, textures, Warp
        str(kit_cache): {"bind": "/isaac-sim/kit/cache", "mode": "rw"},  # the shader cache
    }


def run_script(argv: list[str]) -> int:
    """``nexus script <path> [args…]``: run one Kit-only script in the Kit image, and return its exit code.

    The scripts in ``scripts/assets/`` that call Kit's extensions run here: the image's launcher
    boots Kit, then runs the script as ``__main__`` with its own arguments. The container mounts the
    working folder read-write at its host path and works in it, and mounts ``$NEXUS_DATA``, default
    ``~/data``, where scans and converted scenes live, the same way: a path typed on the host means the
    same file inside. The console streams to this terminal.

    Raises:
        KitPeerError: The script is no file, or the image build or the container failed to start.
    """
    from docker.errors import DockerException
    from docker.types import DeviceRequest

    target = Path(argv[0]).resolve()
    if not target.is_file():
        raise KitPeerError(f"nexus script runs a Kit-only script by its path, and {argv[0]} is no file")
    try:
        tag = ensure_image()
    except KitPeerError:
        raise
    except RuntimeError as exc:
        raise KitPeerError(f"the script runs in a Kit container, but {exc}") from exc
    cwd = Path.cwd()
    volumes = {str(cwd): {"bind": str(cwd), "mode": "rw"}, **_kit_caches()}
    if not target.is_relative_to(cwd):
        volumes[str(target.parent)] = {"bind": str(target.parent), "mode": "ro"}
    data = Path(os.environ.get("NEXUS_DATA") or Path.home() / "data").resolve()
    if data.is_dir() and not data.is_relative_to(cwd):
        volumes[str(data)] = {"bind": str(data), "mode": "rw"}
    env = {"HOME": _KIT_HOME, "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"}
    for name in ("CESIUM_ION_TOKEN", "NEXUS_DATA"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    try:
        container = client().containers.run(
            tag,
            command=["/nexus-kit/script.py", str(target), *argv[1:]],
            working_dir=str(cwd),
            user=f"{os.getuid()}:{os.getgid()}",
            group_add=[ISAAC_SIM_GID],
            environment=env,
            volumes=volumes,
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            labels={LABEL: "script"},
            detach=True,
        )
    except DockerException as exc:
        raise KitPeerError(f"starting the Kit container failed: {getattr(exc, 'explanation', None) or exc}") from exc
    try:
        for chunk in container.logs(stream=True, follow=True):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return int(container.wait().get("StatusCode", 1))
    finally:
        container.remove(force=True)  # Ctrl-C included: the container dies with this command
