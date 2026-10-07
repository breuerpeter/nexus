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
from nexus_sim._src.core.schema import Controls
from nexus_sim._src.core.stages import peer_stages
from tests.usd import sensor_vehicle as sv

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


class _Controller:
    """Answers the preroll at once and keeps the last measurement it received: the read and exchange
    host stages over ``exchange``.
    """

    def __init__(self):
        self.meas = None

    def connect(self):
        pass

    def stages(self):
        return peer_stages(self)

    def exchange(self, meas, t, timeout=None):
        self.meas = meas
        return Controls(command=np.zeros(4))

    def close(self):
        pass


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
    """Build the run *launch* names, settle it on the ground and run its seed pass; the run and the
    measurement of the settled state. A tick's exchange follows its step, so the settled state's
    measurement is the seed pass's, and the run flies no tick.

    The build's ``force_cpu`` sets the Warp device, and the scope puts it back; a module fixture runs
    before the function-scoped ``warp_cpu`` fixture would.
    """
    from nexus_sim._src.build.assembly import build_orchestrator
    from nexus_sim._src.build.launch import resolve_scenario

    with wp.ScopedDevice("cpu"):
        builder, resolved, cfg = resolve_scenario(launch, catalog=catalog)
        cfg["physics"]["force_cpu"] = True
        controller = _Controller()
        orch = build_orchestrator(resolved.tested_config.vehicle, cfg, builder, controller=controller, max_steps=0)
        orch.run()
    return orch, controller.meas


def _field(meas):
    return (meas.xmag, meas.ymag, meas.zmag)


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


def test_the_magnetometer_reports_the_field_at_the_catalog_scenes_origin(zurich_scene):
    """The magnetometer of a run built on a catalog scene reports the WMM field at that scene's origin."""
    _, meas = zurich_scene

    np.testing.assert_allclose(_field(meas), ZURICH_NED, atol=FIELD_TOL)


def test_a_geo_override_wins_over_the_catalog_origin(geo_override):
    """A `--geo` override wins over the catalog origin for every ambient value."""
    _, meas = geo_override

    np.testing.assert_allclose(_field(meas), ZURICH_NED, atol=FIELD_TOL)


def test_a_scene_with_no_origin_reads_the_default_origins_field(empty_scene):
    """A scene whose catalog entry carries no origin resolves to the bundled default origin, Woodinville."""
    _, meas = empty_scene

    np.testing.assert_allclose(_field(meas), WOODINVILLE_NED, atol=FIELD_TOL)


def test_a_scene_with_no_origin_reports_the_default_origin_on_gps(empty_scene):
    """A scene whose catalog entry carries no origin resolves to the bundled default origin, Woodinville."""
    _, meas = empty_scene

    assert (meas.lat_deg, meas.lon_deg) == pytest.approx(WOODINVILLE, abs=1e-4)


def test_the_imu_reports_the_gravity_the_physics_applies(gravity_five_vehicle):
    """The IMU reports the gravity the physics applies, one value."""
    _, meas = gravity_five_vehicle
    reported = math.hypot(meas.xacc, meas.yacc, meas.zacc)

    # The site's one gravity, 9.81: the model-gravity row pins the physics at it, and the IMU reads it.
    assert reported == pytest.approx(9.81, abs=0.1), f"the IMU reports {reported}"


def test_the_barometer_reports_the_sites_pressure_and_temperature(gravity_five_vehicle):
    """The barometer reports the site's pressure and temperature as before."""
    orch, meas = gravity_five_vehicle
    altitude = float(orch.physics.current_state.body_q.numpy()[0, 2])  # the settled base body, above the ground
    isa_pressure = 1013.25 * (1 - 2.25577e-5 * altitude) ** 5.25588

    assert (meas.abs_pressure, meas.pressure_alt, meas.temperature) == pytest.approx(
        (isa_pressure, altitude, 25.0), abs=0.1
    )
