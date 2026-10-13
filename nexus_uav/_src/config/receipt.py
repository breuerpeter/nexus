"""The tested-config receipt.

The fully specified config a run *actually* simulated: the vehicle it flew + asset shas + PX4
airframe/overrides + each peer's pin + runtime. Re-running this reproduces the run's inputs byte-for-byte. It sits
next to the run's ``.rrd``/``.ulg``. The *where* is the runtime's call;
this module owns the *what*.
"""

from __future__ import annotations

import pathlib

from pydantic import Field

from nexus_uav._src.peers.pins import PeerPin

from .catalog import CatalogLayer
from .models import AssetRef, Px4Spec, Runtime, _Base


class TestedConfig(_Base):
    """The fully specified, sha-pinned config the run simulated.

    Must contain **every input that affects the simulation** so re-running it reproduces the run, and
    nothing a run leaves unapplied. ``resolve`` fills it from the launch and the catalog, and the build
    stamps what only a build knows: the airframe the vehicle Universal Scene Description (USD) file declares for PX4, and in ``runtime.device``
    the device the run picked, ``cpu`` or ``cuda``, never ``auto``. A receipt a run flew comes from the
    build; a bare ``resolve`` still carries the launch's device selector.
    """

    __test__ = False  # not a pytest test class despite the name

    vehicle: str  # the variant's key in the catalog, or the local ``.usd`` path a run flew
    # Each catalog the run loaded itself, in load order, with its kind, its owner and its file: the bundled
    # one, each installed package's, the project's. Empty for a run handed its catalog.
    catalogs: list[CatalogLayer] = Field(default_factory=list)
    vehicle_usd: AssetRef
    layer: AssetRef | None = None  # the override layer the run composed over the vehicle and the scene
    px4: Px4Spec | None = None
    peers: dict[str, PeerPin] = Field(default_factory=dict)  # each started peer's pin, by the peer's name
    scene: str
    scene_usd: AssetRef
    scene_start: tuple[float, float, float] | None = None  # scene-frame point placed at the world origin
    runtime: Runtime

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def write(self, path: str | pathlib.Path) -> pathlib.Path:
        p = pathlib.Path(path)
        p.write_text(self.to_json())
        return p


class ResolvedLaunch(_Base):
    """``resolve()`` output: the receipt plus the verified local asset paths, ready to build a sim."""

    tested_config: TestedConfig
    vehicle_usd_path: str | None = None
    scene_usd_path: str | None = None
