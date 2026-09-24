"""Typed launch models, on pydantic v2.

The single ``LaunchConfig`` schema is what the YAML file and the programmatic setters both
serialize to and deserialize from: one model, two front doors.
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    # extra="forbid" turns a typo in a YAML key, say ``vehicel:``, into a load error instead of a
    # silently ignored field: config bugs should fail loudly.
    model_config = ConfigDict(extra="forbid")


class AssetRef(_Base):
    """A content-addressed asset resolved to a verified local path: ``{url, sha256, filename}``.

    The registry declares the compact ``{name, sha256}`` form; a load-time validator expands it to
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
    """PX4 airframe selection: the Software In The Loop (SITL) airframe **make target** this vehicle
    flies as.

    The value is the target without PX4's ``none_`` prefix: ``astro_max`` becomes ``make px4_sitl
    none_astro_max``, and the launcher prepends it. This is where the sim picks the autopilot it
    starts, so this is the authority, not a receipt.
    """

    airframe: str


class Control(_Base):
    """Who flies the vehicle: the host boundary. PX4 is the one first-class controller;
    every other controller is an example under ``nexus/examples/`` that self-assembles its
    orchestrator and enters via ``Sim.from_orchestrator``, with no control kind.
    """

    kind: Literal["px4-sitl"] = "px4-sitl"


class Runtime(_Base):
    # "auto" resolves to CUDA when present, the captured default. An EXPLICIT "cpu" is the bit-exact
    # determinism authority, and every run honors it, one that renders included.
    device: str = "auto"
    seed: int = 42
    dt: float = 0.004
    substeps: int = 4
    max_steps: int | None = None
    # Real-time factor throttle: 0 = unthrottled, run as fast as the controller keeps up, the
    # headless/CI default. 1.0 paces the loop to wall-clock for human-in-the-loop flying: sticks
    # feel 1:1 and sim-time protocol timeouts line up with wall-clock peers such as the companion's
    # 1 Hz Remote ID heartbeat, since PX4's hrt runs on sim time under lockstep.
    rtf: float = 0.0
    determinism: Literal["bit-exact", "tolerance"] = "bit-exact"
    # Physics integrator: mujoco, with contact fidelity, is the SITL default; the
    # gradient-capable semi_implicit / featherstone serve the design-optimization path.
    solver: Literal["mujoco", "semi_implicit", "featherstone"] = "mujoco"


class Environment(_Base):
    """Ambient fields; derived from the scene's geodetic origin by default, overridable here."""

    wind: dict[str, Any] | None = None


class Output(_Base):
    """Artifacts produced for newton-suite. Rerun logging is two mutually exclusive flags, serve or
    file but not both: a live server and a complete ``.rrd`` can't both come out of one process. Omitting
    both means the sim builds no ``Logger`` at all: no recording, no per-tick log fan-out, max benchmark/CI speed.
    """

    log: bool = False  # write the full .rrd to disk, no server
    view: bool = False  # serve the live recording on :9876 for a viewer; a viewer + PX4 merge in
    debug: bool = False  # axes-only scene: log each body's coordinate-frame triad, not its mesh, for a small .rrd
    ulog: bool = True
    video: bool = False
    run_id: str | None = None

    @model_validator(mode="after")
    def _check_log_view(self) -> Output:
        if self.log and self.view:
            raise ValueError("output.log (write .rrd) and output.view (serve viewer) are mutually exclusive")
        return self


class LaunchConfig(_Base):
    """The sim's *fixed* standup properties: what the sim **is**, not what happens to it.

    Dynamics, such as faults and moving actors, are control-API verbs, not config.
    Small in the common case because everything defaults from the registry.
    """

    vehicle: str | None = None
    """The vehicle to fly: a registry ``name`` handle (``--vehicle astro_max_fpv``), or a path to a
    local vehicle Universal Scene Description (USD) file.

    ``None`` takes the registry's ``defaults.vehicle``.
    """
    registry: str | None = None
    """Path to the catalog that extends the bundled one for this run, the ``--registry`` value.

    ``None`` takes the nearest ``nexus.registry.yaml`` in the working directory or a directory over
    it, and only the catalog bundled in the wheel when no directory holds one.
    """
    scene: str | None = None
    """Name of the static scene to stand up, keyed into the registry's ``scenes``, or a local
    scene Universal Scene Description (USD) path (a converted mesh/splat, flown as the visual world).

    ``None`` falls back to the registry's default scene.
    """
    environment: Environment | None = None
    """Ambient-field overrides (e.g. wind) layered on top of the scene's derived fields.

    ``None`` means take the fields derived from the scene's geodetic origin with no override.
    """
    geodetic_origin: GeodeticOrigin | None = None
    """Override the scene's geodetic origin (the lat/lon the local frame anchors to).

    ``None`` uses the registry scene's ``geodetic_origin`` (its default). A launch value wins: it
    re-anchors the GPS/magnetic/gravity reference and, for the streamed cesium globe, selects the
    place it streams. So `cesium` at any location is `--scene cesium --geo <lat>,<lon>`.
    """
    control: Control = Field(default_factory=Control)
    """Who flies the vehicle: the host boundary.

    Defaults to a :class:`Control` with ``kind="px4-sitl"``.
    """
    runtime: Runtime = Field(default_factory=Runtime)
    """Solver, device, timestep, and determinism settings for the run.

    Defaults to the ``auto`` device with the ``mujoco`` solver and bit-exact determinism.
    """
    sensors: dict[str, Any] = Field(default_factory=dict)
    """Per-sensor configuration overrides, keyed by sensor name.

    Empty by default, meaning sensors take their vehicle/registry defaults.
    """
    output: Output = Field(default_factory=Output)
    """Artifacts produced for newton-suite (Rerun recording, ULog, video).

    Defaults to a :class:`Output` (Rerun viewer + ULog on, video off).
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

    def set_vehicle(self, vehicle: str | None) -> LaunchConfig:
        """Set the vehicle to fly in place: a registry ``name`` handle, a local ``.usd`` path, or ``None``.

        ``resolve()`` reads a value with a path separator or a ``.usd`` suffix as a file, every other
        value as a registry name, and ``None`` as the registry's default vehicle, so this setter takes
        a ``--vehicle`` command-line value unchanged. ``Sim`` and the command-line tool both select
        through it, so the two behave identically.

        Args:
            vehicle: The registry name, the local ``.usd`` path, or ``None`` for the registry default.

        Returns:
            ``self``, so calls chain.
        """
        self.vehicle = vehicle
        return self

    def set_scene(self, scene: str) -> LaunchConfig:
        """Set the scene name in place.

        Args:
            scene: The scene name to key into the registry's ``scenes``.

        Returns:
            ``self``, so calls chain.
        """
        self.scene = scene
        return self

    def set_geodetic_origin(self, lat: float, lon: float, alt: float | None = None) -> LaunchConfig:
        """Override the scene's geodetic origin, lat and lon, in place. Returns ``self``, chainable."""
        self.geodetic_origin = GeodeticOrigin(lat=lat, lon=lon, alt=alt)
        return self

    def set_control(self, kind: str, **kw: Any) -> LaunchConfig:
        """Set the control configuration in place.

        Args:
            kind: A control kind declared by :class:`Control`: PX4 is the one first-class
                controller, and every other controller is an example that self-assembles its
                orchestrator and enters via ``Sim.from_orchestrator`` instead. See
                :class:`Control` for the current set; this docstring deliberately doesn't
                repeat it.
            **kw: Extra :class:`Control` fields. ``Control`` declares none besides ``kind`` and
                sets ``extra="forbid"``, so any keyword passed today raises.

        Returns:
            ``self``, so calls chain.

        Raises:
            pydantic.ValidationError: If ``kind`` isn't a declared control kind, or a keyword
                field that ``Control`` doesn't declare is present.
        """
        self.control = Control(kind=kind, **kw)
        return self
