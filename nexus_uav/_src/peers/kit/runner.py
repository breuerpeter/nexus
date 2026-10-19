"""The Kit render peer's container: its image, what it mounts, and its start, stop and alive.

The container runs NVIDIA's Isaac Sim image as pulled, with no image of nexus's own: the peer
program in ``peer-src/``, package data beside this module, mounts read-only, and everything else
the peer needs is a run option. So an RTX run on a fresh machine costs one pull, an edit to the peer
program takes effect on the next run, and an Isaac upgrade is a one-line pin change. The
container runs through the docker SDK, as PX4 Software In The Loop (SITL) does.

The container runs as the host user and mounts only what the run renders and the caches it writes:
the asset cache and the folder of each local Universal Scene Description (USD) file read-only at their host paths, and Kit's own
caches and the logs folder read-write, each created as the user first. Docker creates a missing
bind source owned by root, and a root-owned asset cache failed every later host run.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_uav._src.assets.resolver import fetch
from nexus_uav._src.core import logger
from nexus_uav._src.core.errors import RunError
from nexus_uav._src.core.run_folder import run_folder
from nexus_uav._src.peers import (
    pins,
)
from nexus_uav._src.peers.containers import ContainerPeer, client
from nexus_uav._src.peers.pins import PeerPin

if TYPE_CHECKING:
    from nexus_uav._src.config.catalog import CatalogLayer

KIT_DIR = Path(__file__).with_name("peer-src")
NAME = "kit"  # the peer's folder name, which names its pin file and its variable
ISAAC_SIM_GID = "1234"  # the base image's isaac-sim group: /isaac-sim is readable by that group only
_KIT_HOME = "/kit-home"
_PEER_DIR = "/nexus-kit"
# Cesium for Omniverse, the live-streamed geolocated globe. v0.29.0 is the first release built for
# Kit 110 / Isaac Sim 6.0.1. Its two Kit extensions, cesium.omniverse and cesium.usd.plugins, mount
# at /cesium-exts, and the peer adds that folder to the extension search path for a Cesium scene.
CESIUM = {
    "url": "https://github.com/CesiumGS/cesium-omniverse/releases/download/v0.29.0/"
    "CesiumGS-cesium-omniverse-linux-x86_64-v0.29.0.zip",
    "sha256": "fe38d6195974620e87d7cdad32bb4b38d5933389523d56e75bbae23bcf4da55f",
}
_CESIUM_EXTS = "/cesium-exts"
# The Cesium USD schemas must register at USD start-up: Kit builds the schema registry once at
# boot, so enabling the extension later registers them too late for prim definitions.
_CESIUM_PLUGINS = f"{_CESIUM_EXTS}/cesium.usd.plugins/plugins/CesiumUsdSchemas/resources"


class KitPeerError(RunError):
    """The Kit render peer couldn't start, or died mid-flight; the message names the cause. A run error, so a
    script's ``except nexus_uav.RunError`` catches it.
    """


def resolve(layers: Sequence[CatalogLayer] = ()) -> PeerPin:
    """The image the peer runs, checked, through :func:`nexus_uav._src.peers.pins.resolve`: the shipped
    ``pin.ref`` names NVIDIA's Isaac Sim, and a package's or a project's ``nexus.kit.ref`` or
    ``$NEXUS_KIT_REF`` names another, a local tag among them.

    Args:
        layers: The catalog layers the run loaded, in load order; none for a run handed its catalog.

    Raises:
        ValueError: The pin names no image, or two installed packages pin it; the message names its source.
    """
    return check(pins.resolve(NAME, layers))


def check(pin: PeerPin) -> PeerPin:
    """``pin``, checked: it names one image.

    Raises:
        ValueError: The pin names no image; the message names its source.
    """
    if not pin.ref or any(c.isspace() for c in pin.ref):
        raise ValueError(
            f"{pin.source}: the Kit pin {pin.ref!r} is no image reference, such as nvcr.io/nvidia/isaac-sim:6.0.1"
        )
    return pin


def pull_image(image: str) -> str:
    """Pull ``image`` when this machine lacks it, and return its name.

    NVIDIA's Isaac Sim image is about 21 GB and needs no login. The pull prints a line per finished
    layer as it goes.

    Raises:
        KitPeerError: The pull failed; the message names the image and the registry's error.
        RuntimeError: The docker daemon is unreachable; see :func:`nexus_uav._src.peers.containers.client`.
    """
    from docker.errors import DockerException, ImageNotFound

    c = client()
    try:
        c.images.get(image)
        return image
    except ImageNotFound:
        pass
    logger.info(f"pulling the Kit image {image}: once per machine")
    repository, tag = image, "latest"
    if ":" in image.rsplit("/", 1)[-1]:  # a tag, not the colon of a registry's port
        repository, tag = image.rsplit(":", 1)
    try:
        for chunk in c.api.pull(repository, tag=tag, stream=True, decode=True):
            if "error" in chunk:
                raise KitPeerError(f"pulling the Kit image {image} failed: {chunk['error'].strip()}")
            status = chunk.get("status", "")
            if "id" not in chunk:
                sys.stderr.write(f"{status}\n")
            elif status == "Pull complete":  # one line per layer, not its progress
                sys.stderr.write(f"{chunk['id']}: {status}\n")
            sys.stderr.flush()
    except DockerException as exc:
        raise KitPeerError(f"pulling the Kit image {image} failed: {getattr(exc, 'explanation', None) or exc}") from exc
    logger.info(f"pulled the Kit image {image}")
    return image


def prepare(pin: PeerPin | None = None) -> str:
    """Pull the pinned image the way a run's first start does, and return its name.

    For a caller that wants the cold work done outside a run's budget: CI before a flight or a test, or a
    machine's first setup. ``None`` takes the pin with no project catalog.
    """
    return pull_image((pin or resolve()).ref)


def _declares_cesium_tileset(path: Path) -> bool:
    """Whether the USD at ``path`` holds a Cesium tileset, the prim the peer claims a Cesium scene by."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    return any(p.GetTypeName() == "CesiumTilesetPrim" for p in stage.Traverse())


def _cesium_exts(cache_dir: Path) -> Path:
    """Fetch Cesium for Omniverse into the asset cache, unpack it once beside the zip, and return the
    folder that holds its extensions.

    Raises:
        KitPeerError: The fetch or the unpack failed, a hash mismatch included.
    """
    try:
        zip_path = fetch(CESIUM["url"], CESIUM["sha256"], cache_dir=cache_dir)
        exts = zip_path.parent / "exts"
        if not exts.is_dir():
            # Unpack beside it and rename, so no run mounts a half-unpacked folder.
            tmp = Path(tempfile.mkdtemp(dir=zip_path.parent))
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
            try:
                tmp.rename(exts)
            except OSError:
                if not exts.is_dir():  # another run didn't unpack it first
                    raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise KitPeerError(f"fetching Cesium for Omniverse failed: {exc}") from exc
    return exts


def _cesium_mount(cache_dir: Path) -> tuple[dict, dict]:
    """The mount and the environment that give Kit the Cesium extensions and register their schemas."""
    exts = _cesium_exts(Path(cache_dir).resolve())  # docker reads a relative source as a volume name
    return {str(exts): {"bind": _CESIUM_EXTS, "mode": "ro"}}, {"PXR_PLUGINPATH_NAME": _CESIUM_PLUGINS}


def _run_options(volumes: dict, env: dict) -> dict:
    """The run options every Kit container takes, around its own ``volumes`` and ``env``.

    NVIDIA's image runs as its own user with its own command, so the run sets both: the host user
    with the image's isaac-sim group added, and Kit's Python as the entrypoint. Kit writes its
    settings and logs under /isaac-sim/kit, which only isaac-sim owns, so those two are tmpfs
    mounts that stay in the container. The base's health check greps Kit's log under the image
    user's home, which the peer, with a home folder of its own, never writes, so every healthy peer read as
    unhealthy: the run turns it off, and the render link says whether the peer lives.

    No secret goes in ``env``: Kit prints its whole environment into the console at every boot, and
    the console is a log file.
    """
    from docker.types import DeviceRequest

    return {
        "entrypoint": ["/isaac-sim/python.sh"],
        "user": f"{os.getuid()}:{os.getgid()}",
        "group_add": [ISAAC_SIM_GID],
        "environment": {
            "HOME": _KIT_HOME,
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            **env,
        },
        "volumes": {str(KIT_DIR): {"bind": _PEER_DIR, "mode": "ro"}, **_kit_caches(), **volumes},
        "tmpfs": {"/isaac-sim/kit/data": "", "/isaac-sim/kit/logs": ""},
        "healthcheck": {"test": ["NONE"]},
        "device_requests": [DeviceRequest(count=-1, capabilities=[["gpu"]])],
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class KitPeer(ContainerPeer):
    """One Kit render peer container for one run, a :class:`ContainerPeer`.

    Args:
        files: Every USD file the run renders, vehicle and scene: the peer reads each at its host path.
        cache_dir: The asset cache the run fetched into; mounted whole when a file lies inside it.
        pin: The image the container runs, checked by :meth:`check_pin`.
        log_dir: The run's folder: the peer's files live under ``kit/`` there, Kit's console as ``console.log``
            and its ``--benchmark`` metrics. ``None`` takes a new run folder.
        instance: The run's instance, which names the container; 0 outside a run, as the tests make one.
        **run: The run's other values, which the peer needs none of.
    """

    name = NAME
    check_pin = staticmethod(check)

    def __init__(
        self, files: list = (), *, cache_dir, pin: PeerPin, log_dir: Path | None = None, instance: int = 0, **run
    ):
        super().__init__(instance=instance, log_dir=Path(log_dir) if log_dir is not None else run_folder())
        self._files = [Path(f).resolve() for f in files if f]
        self._cache_dir = Path(cache_dir).resolve()
        self.image = pin.ref
        self.port: int | None = None

    def _read_mounts(self) -> dict:
        mounts: list[Path] = []
        for f in self._files:
            folder = self._cache_dir if f.is_relative_to(self._cache_dir) else f.parent
            if folder not in mounts:
                mounts.append(folder)
        return {str(m): {"bind": str(m), "mode": "ro"} for m in mounts}

    def start(self) -> None:
        """Pull the image if this machine lacks it, then start the container; Kit boots in the background.

        A file that declares a Cesium tileset first fetches Cesium for Omniverse into the asset cache.

        Raises:
            KitPeerError: The Cesium fetch, the pull or the container start failed, or the docker
                daemon is unreachable; the message names the cause.
        """
        from docker.errors import DockerException

        try:
            super().start()
        except KitPeerError:
            raise
        except RuntimeError as exc:
            raise KitPeerError(f"the RTX sensors render in a Kit container, but {exc}") from exc
        except DockerException as exc:
            raise KitPeerError(
                f"starting the Kit container failed: {getattr(exc, 'explanation', None) or exc}"
            ) from exc
        logger.info(f"Kit render peer {self.container} booting on :{self.port}; its console -> {self.log_path}")

    def run_options(self) -> dict:
        """The pulled image serving the render link on a free port, with the files the run renders, the Cesium
        extensions when a file needs them, and the peer's folder mounted. The container removes itself when it
        exits.
        """
        volumes, env = {}, {}
        if any(_declares_cesium_tileset(f) for f in self._files):
            volumes, env = _cesium_mount(self._cache_dir)
        image = pull_image(self.image)
        volumes.update(self._read_mounts())
        volumes[str(self.dir)] = {"bind": str(self.dir), "mode": "rw"}  # the --benchmark JSON
        # The ion token rides the setup message, never the environment.
        self.port = _free_port()
        return {
            "image": image,
            "command": [f"{_PEER_DIR}/serve.py", "--port", str(self.port)],
            "ports": {f"{self.port}/tcp": ("127.0.0.1", self.port)},
            "auto_remove": True,
            **_run_options(volumes, env),
        }

    def console_tail(self, lines: int = 20) -> str:
        """The last ``lines`` of the container's console, for an error message."""
        try:
            text = self.log_path.read_text(errors="replace")
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
