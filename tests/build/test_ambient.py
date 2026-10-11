"""The ambient values a run's sensors read resolve at build from the scene's geodetic origin.

Beside the launch tests, since pytest skips a directory named ``build``. Real builds on the Warp CPU
backend: the local fixture vehicle in ``tests/usd/sensor_vehicle.py`` with an analytic sensor of each kind, a stand-in controller
that answers at once, and one tick at rest on the ground. Each build settles a Newton model, so the
flights are module-scoped. Skipped without newton or pxr.
"""

import math

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
WOODINVILLE = (47.747944, -122.163917)
# The World Magnetic Model (WMM) field at each origin as a North East Down (NED) vector in gauss, from
# PX4's coarse table: Zurich D 2.6°, I 63.2°, F 0.481; Woodinville D 15.5°, I 69.4°, F 0.539. The
# vehicle rests level with its nose north, so its Forward Right Down (FRD) body axes are the NED axes.
ZURICH_NED = (0.216, 0.010, 0.429)
WOODINVILLE_NED = (0.182, 0.051, 0.504)
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
    """A project catalog over the bundled one: two scenes with an origin each, and `bare` with none. None names a scene file, so no build fetches one."""
    path = tmp_path / "nexus.catalog.yaml"
    path.write_text(
        "scenes:\n"
        f"  zurich:\n    geodetic_origin: {{ lat: {ZURICH[0]}, lon: {ZURICH[1]}, alt: {ZURICH[2]} }}\n"
        f"  woodinville:\n    geodetic_origin: {{ lat: {WOODINVILLE[0]}, lon: {WOODINVILLE[1]}, alt: 5.02 }}\n"
        "  bare: {}\n"
    )
    return load_catalog(path)


def _settled(launch, catalog):
    """Build the run *launch* names, settle it on the ground and run its seed pass; the run and each sensor's
    sample of the settled state, by signal. A tick's exchange follows its step, so the settled state's samples
    are the seed pass's, and the run flies no tick.

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


@pytest.fixture(scope="module")
def zurich_scene(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zurich")
    return _settled(LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene("zurich"), _catalog(tmp))


@pytest.fixture(scope="module")
def geo_override(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("geo")
    launch = LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene("woodinville").set_geodetic_origin(*ZURICH)
    return _settled(launch, _catalog(tmp))


@pytest.fixture(scope="module")
def empty_scene(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("empty")
    return _settled(LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene("bare"), _catalog(tmp))


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


# A scene of the test's own whose PhysicsScene prim authors the world's values: the magnetic field in the world
# frame, X north, Y west, Z up, in gauss; the pressure at mean sea level, hPa; the temperature, degrees Celsius.
AUTHORED_FIELD_WORLD = (0.2, -0.05, -0.45)
AUTHORED_FIELD_BODY = (0.2, 0.05, 0.45)  # the same field in the FRD body axes of the level vehicle, nose north
AUTHORED_PRESSURE = 950.0
AUTHORED_TEMPERATURE = 31.0


def _scene(tmp, authored: bool) -> str:
    """Write an empty scene under `tmp` and return its path; its PhysicsScene authors the world's values when
    `authored`, and nothing otherwise.
    """
    values = (
        (
            f"    custom float3 newton:nexus:magnetic = {AUTHORED_FIELD_WORLD}\n"
            f"    custom float newton:nexus:pressure = {AUTHORED_PRESSURE}\n"
            f"    custom float newton:nexus:temperature = {AUTHORED_TEMPERATURE}\n"
        )
        if authored
        else ""
    )
    path = tmp / ("authored_scene.usda" if authored else "plain_scene.usda")
    path.write_text(
        '#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "World"\n{\n}\n\n'
        f'def PhysicsScene "PhysicsScene"\n{{\n{values}}}\n'
    )
    return str(path)


@pytest.fixture(scope="module")
def authored_scene(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("authored")
    return _settled(LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene(_scene(tmp, True)), _catalog(tmp))


@pytest.fixture(scope="module")
def plain_scene(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("plain")
    return _settled(LaunchConfig().set_vehicle(sv.vehicle(tmp, SENSORS)).set_scene(_scene(tmp, False)), _catalog(tmp))


def test_a_scene_that_authors_its_field_flies_with_it_and_one_that_authors_none_with_its_origins(
    authored_scene, plain_scene
):
    """A scene whose `PhysicsScene` authors its magnetic field flies with that field, and one that authors
    none flies with the field of its origin.

    Given two copies of a scene, one authoring the field, when a vehicle flies each, then the magnetometer
    reports the authored field in the body's axes on the first and the origin's World Magnetic Model field
    on the second.
    """
    _, authored = authored_scene
    _, plain = plain_scene

    np.testing.assert_allclose(
        (_field(authored), _field(plain)), (AUTHORED_FIELD_BODY, WOODINVILLE_NED), atol=FIELD_TOL
    )


def test_a_scene_that_authors_its_pressure_and_temperature_flies_with_them_and_one_that_authors_none_with_the_standard_atmosphere(
    authored_scene, plain_scene
):
    """A scene that authors its pressure at sea level and its temperature flies with them, and one that
    authors none flies with the standard atmosphere.

    Given two copies of a scene, one authoring both values, when a vehicle flies each, then the barometer
    reports the authored pressure and temperature on the first and 1013.25 hPa and 25 °C on the second.
    """
    _, authored = authored_scene
    _, plain = plain_scene
    read = lambda samples: (float(samples["baro"]["pressure"]), float(samples["baro"]["temperature"]))  # noqa: E731

    # The vehicle rests a few centimetres over the ground, where the pressure is within 0.1 hPa of its
    # sea-level value.
    assert (*read(authored), *read(plain)) == pytest.approx((950.0, 31.0, 1013.25, 25.0), abs=0.1)


def test_the_magnetometer_reports_the_field_at_the_catalog_scenes_origin(zurich_scene):
    """The magnetometer of a run built on a catalog scene reports the WMM field at that scene's origin."""
    _, samples = zurich_scene

    np.testing.assert_allclose(_field(samples), ZURICH_NED, atol=FIELD_TOL)


def test_a_geo_override_wins_over_the_catalog_origin(geo_override):
    """A `--geo` override wins over the catalog origin for every ambient value."""
    _, samples = geo_override

    np.testing.assert_allclose(_field(samples), ZURICH_NED, atol=FIELD_TOL)


def test_a_scene_with_no_origin_reads_the_default_origins_field(empty_scene):
    """A scene whose catalog entry carries no origin resolves to the bundled default origin, Woodinville."""
    _, samples = empty_scene

    np.testing.assert_allclose(_field(samples), WOODINVILLE_NED, atol=FIELD_TOL)


def test_a_scene_with_no_origin_reports_the_default_origin_on_gps(empty_scene):
    """A scene whose catalog entry carries no origin resolves to the bundled default origin, Woodinville."""
    _, samples = empty_scene
    gps = samples["gps"]

    assert (float(gps["lat"]), float(gps["lon"])) == pytest.approx(WOODINVILLE, abs=1e-4)


def test_the_imu_reports_the_gravity_the_physics_applies(gravity_five_vehicle):
    """The IMU reports the gravity the physics applies, one value."""
    _, samples = gravity_five_vehicle
    reported = math.hypot(*(float(a) for a in samples["imu"]["accel"]))

    # The site's one gravity, 9.81: the model-gravity row pins the physics at it, and the IMU reads it.
    assert reported == pytest.approx(9.81, abs=0.1), f"the IMU reports {reported}"


def test_the_barometer_reports_the_sites_pressure_and_temperature(gravity_five_vehicle):
    """The barometer reports the site's pressure and temperature as before."""
    orch, samples = gravity_five_vehicle
    altitude = float(orch.physics.current_state.body_q.numpy()[0, 2])  # the settled base body, above the ground
    isa_pressure = 1013.25 * (1 - 2.25577e-5 * altitude) ** 5.25588
    baro = samples["baro"]

    assert (float(baro["pressure"]), float(baro["altitude"]), float(baro["temperature"])) == pytest.approx(
        (isa_pressure, altitude, 25.0), abs=0.1
    )
