"""The tested-config receipt.

The fully specified config a run *actually* simulated: the vehicle it flew + asset shas + PX4
airframe/overrides + runtime. Re-running this reproduces the run's inputs byte-for-byte. It sits
next to the run's ``.rrd``/``.ulg``. The *where* is the runtime's call;
this module owns the *what*.
"""

from __future__ import annotations

import pathlib
from typing import Any

from pydantic import Field

from .models import AssetRef, Control, Environment, GeodeticOrigin, Px4Spec, Runtime, _Base


class TestedConfig(_Base):
    """The fully specified, sha-pinned config the run simulated.

    Must contain **every input that affects the simulation** so re-running it reproduces the run.
    Hence it carries sensors + environment overrides too, not just
    the vehicle/scene/control/runtime.
    """

    __test__ = False  # not a pytest test class despite the name

    vehicle: str  # the variant's `name` handle, or the local ``.usd`` path a run flew
    registry: str | None = None  # the catalog the run resolved against, when it loaded one itself
    vehicle_usd: AssetRef
    px4: Px4Spec | None = None
    scene: str
    scene_usd: AssetRef | None = None
    scene_start: tuple[float, float, float] | None = None  # scene-frame point placed at the world origin
    geodetic_origin: GeodeticOrigin | None = None
    control: Control
    runtime: Runtime
    sensors: dict[str, Any] = Field(default_factory=dict)
    environment: Environment | None = None

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
