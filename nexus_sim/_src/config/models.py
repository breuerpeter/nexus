"""Typed launch models, on pydantic v2.

The single ``LaunchConfig`` schema is what the YAML file and the programmatic setters both
serialize to and deserialize from: one model, two front doors.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    # extra="forbid" turns a typo in a YAML key, say ``vehicel:``, into a load error instead of a
    # silently ignored field: config bugs should fail loudly. The model checks a value set on it later
    # the same way, so ``launch.runtime.device = "cuda:1"`` fails as the launch file would.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AssetRef(_Base):
    """A content-addressed asset resolved to a verified local path: ``{url, sha256, filename}``.

    The catalog declares the compact ``{name, sha256}`` form; a load-time validator expands it to
    this, deriving the ``.usdz`` URL + cache filename from the shared hosting scheme. The sha *is*
    the version pin: the resolver verifies it on fetch.
    """

    url: str
    sha256: str
    filename: str | None = None


class GeodeticOrigin(_Base):
    """The geo coords of the world origin, the scene's ``start`` point where the vehicle spawns: the
    Global Positioning System (GPS) init and the geo-referenced fields, magnetic and gravity, anchor
    to it.

    ``alt`` is the WGS84 *ellipsoidal* height of that surface. Optional: a cesium scene MEASURES
    it from the streamed world at run time with the ground-align probe; a converted scan can author
    it, and ``spawn_site.py --geo`` computes it; ``None`` keeps the scenario default reference
    altitude, which serves GPS realism only.
    """

    lat: float
    lon: float
    alt: float | None = None


class Px4Spec(_Base):
    """The receipt's record of the PX4 airframe: the Software In The Loop (SITL) airframe **make
    target** the vehicle flies as.

    The value is the target without PX4's ``none_`` prefix: ``astro_max`` becomes ``make px4_sitl
    none_astro_max``, and the launcher prepends it. The vehicle Universal Scene Description (USD)
    file declares it, as ``nexus:airframe`` on its ``NexusPx4API`` schema, and is the authority.
    """

    airframe: str


class Runtime(_Base):
    # "auto" resolves to CUDA when present, the captured default. An EXPLICIT "cpu" is the bit-exact
    # determinism authority, and every run honors it, one that renders included. A CUDA run is
    # tolerance-gated. The receipt records the device the run picked, "cpu" or "cuda".
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42
    dt: float = 0.004
    max_steps: int | None = None
    # Real-time factor throttle: 0 = unthrottled, run as fast as the controller keeps up, the
    # headless/CI default. 1.0 paces the loop to wall-clock for human-in-the-loop flying: sticks
    # feel 1:1 and sim-time protocol timeouts line up with wall-clock peers such as the companion's
    # 1 Hz Remote ID heartbeat, since PX4's hrt runs on sim time under lockstep.
    rtf: float = 0.0
    # Physics integrator: mujoco, with contact fidelity, is the SITL default; the
    # gradient-capable semi_implicit / featherstone serve the design-optimization path.
    solver: Literal["mujoco", "semi_implicit", "featherstone"] = "mujoco"


class Output(_Base):
    """The Rerun recording a run produces. Rerun logging is two mutually exclusive flags, serve or
    file but not both: a live server and a complete ``.rrd`` can't both come out of one process. Omitting
    both means the sim builds no ``Logger`` at all: no recording, no per-tick log fan-out, max benchmark/CI speed.
    """

    log: bool = False  # write the full .rrd to disk, no server
    view: bool = False  # serve the live recording on :9876 for a viewer; a viewer + PX4 merge in
    debug: bool = False  # axes-only scene: log each body's coordinate-frame triad, not its mesh, for a small .rrd

    @model_validator(mode="after")
    def _check_log_view(self) -> Output:
        if self.log and self.view:
            raise ValueError("output.log (write .rrd) and output.view (serve viewer) are mutually exclusive")
        return self


class LaunchConfig(_Base):
    """The sim's *fixed* standup properties: what the sim **is**, not what happens to it.

    Dynamics, such as faults and moving actors, are control-API verbs, not config.
    Small in the common case: the vehicle and the scene, and the catalog fills in the rest.
    """

    vehicle: str | None = None
    """The vehicle to fly: a catalog ``name`` handle (``--vehicle astro_max_fpv``), or a path to a
    local vehicle Universal Scene Description (USD) file.

    ``None`` until set; a launch that still names none fails to resolve.
    """
    catalog: str | None = None
    """Path to the catalog that extends the bundled one for this run, the ``--catalog`` value.

    ``None`` takes the nearest ``nexus.catalog.yaml`` in the working directory or a directory over
    it, and only the catalog bundled in the wheel when no directory holds one.
    """
    scene: str | None = None
    """Name of the static scene to stand up, keyed into the catalog's ``scenes``, or a local
    scene Universal Scene Description (USD) path (a converted mesh/splat, flown as the visual world).

    ``None`` until set; a launch that still names none fails to resolve.
    """
    geodetic_origin: GeodeticOrigin | None = None
    """Override the scene's geodetic origin (the lat/lon the local frame anchors to).

    ``None`` uses the catalog scene's ``geodetic_origin`` (its default). A launch value wins: it
    re-anchors the GPS/magnetic/gravity reference and, for the streamed cesium globe, selects the
    place it streams. So `cesium` at any location is `--scene cesium --geo <lat>,<lon>`.
    """
    layer: str | None = None
    """Path to an override layer, a local Universal Scene Description (USD) file the run composes over
    the vehicle, the ``--layer`` value.

    Its opinions win over the vehicle's: it changes a declared value, selects a variant, or drops a
    declaration, such as the PX4 Software In The Loop (SITL) peer's to fly an autopilot started
    elsewhere. The receipt records its sha256 beside the vehicle's. ``None`` flies the vehicle as
    its file declares it.
    """
    runtime: Runtime = Field(default_factory=Runtime)
    """Solver, device, timestep, and seed settings for the run.

    Defaults to the ``auto`` device with the ``mujoco`` solver.
    """
    output: Output = Field(default_factory=Output)
    """The Rerun recording the run produces.

    Defaults to an ``Output`` with no recording and no viewer.
    """

    # --- front-doors: dict / YAML / programmatic all build the one model ---
    @classmethod
    def from_dict(cls, data: dict | None) -> LaunchConfig:
        """Build and validate a config from a plain dict.

        Args:
            data: The mapping of config fields. ``None``, or an empty dict, yields a
                fully defaulted config.

        Returns:
            The validated ``LaunchConfig``.

        Raises:
            pydantic.ValidationError: If ``data`` has unknown keys, which ``extra="forbid"`` rejects,
                or values that fail validation.
        """
        return cls.model_validate(data or {})

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> LaunchConfig:
        """Build and validate a config from a YAML file.

        Args:
            path: Path to the YAML launch-config file.

        Returns:
            The validated ``LaunchConfig``.

        Raises:
            pydantic.ValidationError: If the parsed document fails validation.
        """
        return cls.from_dict(yaml.safe_load(pathlib.Path(path).read_text()))

    def set_vehicle(self, vehicle: str) -> LaunchConfig:
        """Set the vehicle to fly in place: a catalog ``name`` handle or a local ``.usd`` path.

        ``resolve()`` reads a value with a path separator or a ``.usd`` suffix as a file and every other
        value as a catalog name, so this setter takes
        a ``--vehicle`` command-line value unchanged. ``Sim`` and the command-line tool both select
        through it, so the two behave identically.

        Args:
            vehicle: The catalog name or the local ``.usd`` path.

        Returns:
            ``self``, so calls chain.
        """
        self.vehicle = vehicle
        return self

    def set_scene(self, scene: str) -> LaunchConfig:
        """Set the scene name in place.

        Args:
            scene: The scene name to key into the catalog's ``scenes``.

        Returns:
            ``self``, so calls chain.
        """
        self.scene = scene
        return self

    def set_geodetic_origin(self, lat: float, lon: float, alt: float | None = None) -> LaunchConfig:
        """Override the scene's geodetic origin, lat and lon, in place. Returns ``self``, chainable."""
        self.geodetic_origin = GeodeticOrigin(lat=lat, lon=lon, alt=alt)
        return self
