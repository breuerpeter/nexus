"""The world's values a run's sensors read come from the scene's `PhysicsScene`: its geodetic origin and what follows from it.

Beside the launch tests, since pytest skips a directory named ``build``. Real builds on the Warp CPU
backend: the local fixture vehicle in ``tests/usd/sensor_vehicle.py`` with an analytic sensor of each kind, a stand-in controller
that answers at once, and one tick at rest on the ground. Each build settles a Newton model, so the
flights are module-scoped. Skipped without newton or pxr.
"""

import math
import re

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.config.catalog import load_catalog
from nexus_sim._src.core.schema import BaroSample, GpsSample, ImuSample, MagSample
from nexus_sim._src.core.signals import Signal
from tests.usd import sensor_vehicle as sv
from tests.usd.sensor_vehicle import HostController

ZURICH = (47.3769, 8.5417, 408.0)
SEATTLE_CENTER = (47.6205, -122.3493, 20.0)
# The attribute a scene's PhysicsScene prim authors its origin in: latitude and longitude in degrees, altitude in metres.
ORIGIN_ATTR = "nexus:geodeticOrigin"
# The World Magnetic Model (WMM) field at each origin as a North East Down (NED) vector in gauss, from
# PX4's coarse table: Zurich declination 2.6°, inclination 63.2°, strength 0.481. The vehicle rests level
# with its nose north, so its Forward Right Down (FRD) body axes are the NED axes.
ZURICH_NED = (0.216, 0.010, 0.429)
FIELD_TOL = 0.015  # five sigma of the vehicle's authored magnetometer noise, 0.003 per axis
# An analytic sensor of each kind on the base body, with the noise the shipped vehicle authors.
SENSORS = (
    sv.prim("Imu", "NexusImuAPI", "float nexus:accNoise = 0.02\nfloat nexus:gyroNoise = 0.02")
    + sv.prim("Mag", "NexusMagAPI", "float3 nexus:noise = (0.003, 0.003, 0.003)")
    + sv.prim("Baro", "NexusBaroAPI", "float nexus:noise = 0.02")
    + sv.prim("Gps", "NexusGpsAPI", "int nexus:fixType = 3")
)


class _Controller(HostController):
    """Answers the preroll at once and keeps the last sample of each sensor its exchange read, by signal."""

    def __init__(self):
        super().__init__()
        self.signals = [
            Signal(name, sample, shape=(1,))
            for name, sample in (("imu", ImuSample), ("mag", MagSample), ("baro", BaroSample), ("gps", GpsSample))
        ]
        self.reads = tuple(self.signals)
        self.samples = {}

    def exchange(self, t, timeout=None):
        self.samples = {signal.name: signal.read()[0] for signal in self.signals}
        return np.zeros(4)


def _catalog(tmp_path):
    """A project catalog that adds nothing to the bundled one: every run here flies a local scene."""
    path = tmp_path / "nexus.catalog.yaml"
    path.write_text("scenes: {}\n")
    return load_catalog(path)


def _settled(launch, catalog):
    """Build the run *launch* names, settle it on the ground and run its start; the run and each sensor's
    sample of the settled state, by signal. The start samples the settled state, and the run flies no tick.

    The build's ``force_cpu`` sets the Warp device, and the scope puts it back; a module fixture runs
    before the function-scoped ``warp_cpu`` fixture would.
    """
    from nexus_sim._src.build.assembly import build_loop
    from nexus_sim._src.build.launch import resolve_scenario

    with wp.ScopedDevice("cpu"):
        builder, resolved, cfg = resolve_scenario(launch, catalog=catalog)
        cfg["physics"]["force_cpu"] = True
        controller = _Controller()
        orch = build_loop(resolved.tested_config.vehicle, cfg, builder, controller=controller, max_steps=0)
        orch.run()
    return orch, controller.samples


def _field(samples):
    return samples["mag"]["field"]


def _gps(samples):
    gps = samples["gps"]
    return float(gps["lat"]), float(gps["lon"]), float(gps["alt"])


def _near(origin):
    """What a Global Positioning System (GPS) receiver at rest at `origin` reports: its latitude and longitude, and its altitude within a metre,
    since the base body rests a quarter of a metre over the ground.
    """
    lat, lon, alt = origin
    return pytest.approx(lat, abs=1e-4), pytest.approx(lon, abs=1e-4), pytest.approx(alt, abs=1.0)


def _scene(tmp_path, physics_scene: str | None) -> str:
    """Write a local scene, an empty world the catalog doesn't list, and return its path.

    `physics_scene`, when given, is the text the scene's ``/World/PhysicsScene`` prim authors under
    ``NexusSceneAPI``; ``None`` writes no ``PhysicsScene`` prim.
    """
    prim = (
        ""
        if physics_scene is None
        else '    def PhysicsScene "PhysicsScene" (\n        prepend apiSchemas = ["NexusSceneAPI"]\n    )\n'
        f"    {{\n        {physics_scene}\n    }}\n"
    )
    path = tmp_path / "local_scene.usda"
    path.write_text(
        f'#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        f'def Xform "World"\n{{\n{prim}}}\n'
    )
    return str(path)


def _origin(origin) -> str:
    return f"double3 {ORIGIN_ATTR} = {tuple(origin)}"


@pytest.fixture(scope="module")
def local_zurich_scene(tmp_path_factory):
    """A run on a local scene whose ``PhysicsScene`` prim authors Zurich as its origin."""
    tmp = tmp_path_factory.mktemp("local_zurich")
    launch = LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene(_scene(tmp, _origin(ZURICH)))
    return _settled(launch, _catalog(tmp))


@pytest.fixture(scope="module")
def layer_moved_origin(tmp_path_factory):
    """A run on that scene under an override layer that moves its origin to Seattle Center."""
    tmp = tmp_path_factory.mktemp("layer_origin")
    layer = tmp / "origin_layer.usda"
    layer.write_text(
        f'#usda 1.0\n\nover "World"\n{{\n    over "PhysicsScene"\n    {{\n        {_origin(SEATTLE_CENTER)}\n    }}\n}}\n'
    )
    launch = LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene(_scene(tmp, _origin(ZURICH)))
    launch.layer = str(layer)
    return _settled(launch, _catalog(tmp))


@pytest.fixture(scope="module")
def gravity_five_vehicle(tmp_path_factory):
    """The fixture vehicle under a local layer whose ``PhysicsScene`` sets gravity to 5."""
    from pxr import Usd, UsdPhysics

    tmp = tmp_path_factory.mktemp("grav5")
    fixture = sv.vehicle(tmp, SENSORS)
    path = tmp / "grav5.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().subLayerPaths.append(fixture)
    UsdPhysics.Scene(stage.OverridePrim("/PhysicsScene")).CreateGravityMagnitudeAttr(5.0)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(path))
    prim = reopened.GetPrimAtPath("/PhysicsScene")
    assert prim, f"the fixture vehicle did not compose under the layer: {fixture!r} {reopened.GetUsedLayers()}"
    composed = UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Get()
    assert composed == 5.0, f"the layer's gravity did not compose over the fixture vehicle: {composed}"
    return _settled(LaunchConfig().set_vehicle(str(path)).set_scene(sv.SCENE), _catalog(tmp))


# A scene whose PhysicsScene authors the world's values beside its origin: the magnetic field in the world
# frame, X north, Y west, Z up, in gauss; the pressure at mean sea level, hPa; the temperature, degrees Celsius.
AUTHORED_FIELD_WORLD = (0.2, -0.05, -0.45)
AUTHORED_FIELD_BODY = (0.2, 0.05, 0.45)  # the same field in the FRD body axes of the level vehicle, nose north
AUTHORED_PRESSURE = 950.0
AUTHORED_TEMPERATURE = 31.0
WORLD_VALUES = (
    f"float3 nexus:magneticField = {AUTHORED_FIELD_WORLD}\n"
    f"        float nexus:seaLevelPressure = {AUTHORED_PRESSURE}\n"
    f"        float nexus:temperature = {AUTHORED_TEMPERATURE}"
)


@pytest.fixture(scope="module")
def authored_world_scene(tmp_path_factory):
    """A run on a local scene whose ``PhysicsScene`` authors Zurich as its origin and the world's values beside it."""
    tmp = tmp_path_factory.mktemp("authored_world")
    scene = _scene(tmp, f"{_origin(ZURICH)}\n        {WORLD_VALUES}")
    launch = LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene(scene)
    return _settled(launch, _catalog(tmp))


def test_a_scene_that_authors_its_field_flies_with_it_and_one_that_authors_none_with_its_origins(
    authored_world_scene, local_zurich_scene
):
    """A scene whose `PhysicsScene` authors its magnetic field flies with that field, and one that authors
    none flies with the field of its origin.

    Given two copies of a scene, one authoring the field, when a vehicle flies each, then the magnetometer
    reports the authored field in the body's axes on the first and the origin's World Magnetic Model field
    on the second.
    """
    _, authored = authored_world_scene
    _, plain = local_zurich_scene

    np.testing.assert_allclose((_field(authored), _field(plain)), (AUTHORED_FIELD_BODY, ZURICH_NED), atol=FIELD_TOL)


def test_a_scene_that_authors_its_pressure_and_temperature_flies_with_them_and_one_that_authors_none_with_the_standard_atmosphere(
    authored_world_scene, local_zurich_scene
):
    """A scene that authors its pressure at sea level and its temperature flies with them, and one that
    authors none flies with the standard atmosphere.

    Given two copies of a scene, one authoring both values, when a vehicle flies each, then the barometer
    reports the authored pressure and temperature on the first and 1013.25 hPa and 25 °C on the second.
    """
    _, authored = authored_world_scene
    _, plain = local_zurich_scene
    read = lambda samples: (float(samples["baro"]["pressure"]), float(samples["baro"]["temperature"]))  # noqa: E731

    # The vehicle rests a quarter of a metre over the ground, where the pressure is within 0.1 hPa of its
    # sea-level value.
    assert (*read(authored), *read(plain)) == pytest.approx((950.0, 31.0, 1013.25, 25.0), abs=0.1)


def test_the_magnetometer_reports_the_field_at_the_scenes_origin(local_zurich_scene):
    """The magnetometer reports the WMM field at the origin the scene's `PhysicsScene` authors."""
    _, samples = local_zurich_scene

    np.testing.assert_allclose(_field(samples), ZURICH_NED, atol=FIELD_TOL)


def test_a_run_flies_at_the_origin_its_scenes_physics_scene_authors(local_zurich_scene):
    """A run flies at the geodetic origin its scene's `PhysicsScene` prim authors, for a local scene the catalog doesn't list too.

    Given a local scene whose `PhysicsScene` authors an origin in Zurich, when a vehicle with a GPS flies
    on it, then the GPS reports Zurich's latitude, longitude and altitude at the start.
    """
    _, samples = local_zurich_scene

    assert _gps(samples) == _near(ZURICH)


def test_an_override_layer_moves_the_scenes_origin(layer_moved_origin):
    """An override layer moves the scene's origin.

    Given a local scene whose `PhysicsScene` authors Zurich and a layer that authors Seattle Center on
    that prim, when the run flies with the layer, then the GPS reports Seattle Center at the start.
    """
    _, samples = layer_moved_origin

    assert _gps(samples) == _near(SEATTLE_CENTER)


def test_a_scene_whose_physics_scene_authors_no_origin_fails_the_build(tmp_path):
    """A scene whose `PhysicsScene` authors no origin fails the build and names the prim.

    Given a local scene whose `PhysicsScene` authors no origin, when the run builds, then it fails, and
    the message names the scene's `PhysicsScene` prim.
    """
    launch = LaunchConfig().set_vehicle(sv.vehicle(tmp_path, SENSORS)).set_scene(_scene(tmp_path, ""))

    with pytest.raises(ValueError, match="/World/PhysicsScene"):
        _settled(launch, _catalog(tmp_path))


def test_a_scene_with_no_physics_scene_fails_the_build(tmp_path):
    """A scene with no `PhysicsScene` prim fails the build and names the scene.

    Given a local scene with no `PhysicsScene` prim, when the run builds, then it fails, and the message
    names the scene's file.
    """
    launch = LaunchConfig().set_vehicle(sv.vehicle(tmp_path, SENSORS)).set_scene(_scene(tmp_path, None))

    with pytest.raises(ValueError, match=re.escape("local_scene.usda")):
        _settled(launch, _catalog(tmp_path))


def test_the_imu_reports_the_gravity_the_physics_applies(gravity_five_vehicle):
    """The IMU reports the gravity the physics applies, one value."""
    _, samples = gravity_five_vehicle
    reported = math.hypot(*(float(a) for a in samples["imu"]["accel"]))

    # The world's one gravity, 9.81: the model-gravity row pins the physics at it, and the inertial measurement unit reads it.
    assert reported == pytest.approx(9.81, abs=0.1), f"the IMU reports {reported}"


def test_the_barometer_reports_the_sites_pressure_and_temperature(gravity_five_vehicle):
    """The barometer reports the scene's pressure and temperature as before."""
    orch, samples = gravity_five_vehicle
    altitude = float(orch.physics.current_state.body_q.numpy()[0, 2])  # the settled base body, above the ground
    isa_pressure = 1013.25 * (1 - 2.25577e-5 * altitude) ** 5.25588
    baro = samples["baro"]

    assert (float(baro["pressure"]), float(baro["altitude"]), float(baro["temperature"])) == pytest.approx(
        (isa_pressure, altitude, 25.0), abs=0.1
    )
