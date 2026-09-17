"""Universal Scene Description (USD) authored sensor suite: ``sensor:*`` prims round-trip through the
parser into the sensor constructors, with the vehicle USD as the single authority for *all* sensors.
Skipped without pxr.
"""

import pytest

pytest.importorskip("pxr")
pytest.importorskip("warp")
import warp as wp

wp.set_device("cpu")

from nexus._src.core.seedtree import SeedTree  # noqa: E402
from nexus._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor  # noqa: E402
from nexus._src.vehicle.sensors.usd import SensorSpec, build_sensors, parse_sensor_prims  # noqa: E402

_GPS_INIT = {"lat": 47.7, "lon": -122.1, "alt": 5.0}


def _author(path: str) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    body = UsdGeom.Xform.Define(stage, "/Vehicle/body")
    imu = UsdGeom.Xform.Define(stage, "/Vehicle/body/Imu")
    imu.AddTranslateOp().Set(Gf.Vec3d(0.01, -0.02, 0.03))  # a non-origin mount, the lever arm
    imu.GetPrim().CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("imu")
    imu.GetPrim().CreateAttribute("sensor:acc_noise", Sdf.ValueTypeNames.Float, custom=True).Set(0.05)
    gps = UsdGeom.Xform.Define(stage, "/Vehicle/body/Gps").GetPrim()
    gps.CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("gps")
    gps.CreateAttribute("sensor:fix_type", Sdf.ValueTypeNames.Int, custom=True).Set(2)
    mag = UsdGeom.Xform.Define(stage, "/Vehicle/body/Mag").GetPrim()
    mag.CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("mag")
    mag.CreateAttribute("sensor:noise", Sdf.ValueTypeNames.Float3, custom=True).Set(Gf.Vec3f(0.003, 0.003, 0.003))
    baro = UsdGeom.Xform.Define(stage, "/Vehicle/body/Baro").GetPrim()
    baro.CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("baro")
    body.GetPrim()  # body itself carries no sensor:type, so it must not parse as a sensor
    stage.GetRootLayer().Save()


def test_parse_and_build_round_trip(tmp_path):
    usd = str(tmp_path / "sensors.usda")
    _author(usd)
    specs = parse_sensor_prims(usd)
    assert [s.kind for s in specs] == ["imu", "gps", "mag", "baro"]  # authoring order
    imu_spec = specs[0]
    assert imu_spec.mount == pytest.approx((0.01, -0.02, 0.03))
    assert imu_spec.params == {"acc_noise": pytest.approx(0.05)}

    sensors = build_sensors(specs, seedtree=SeedTree(42), dt=0.004, gps_init=_GPS_INIT, ref_alt=5.0)
    imu, gps, mag, baro = sensors
    assert isinstance(imu, ImuSensor) and isinstance(gps, GpsSensor)
    assert isinstance(mag, MagSensor) and isinstance(baro, BaroSensor)
    assert tuple(imu.r_com_to_mount) == pytest.approx((0.01, -0.02, 0.03), abs=1e-6)
    assert imu.acc_noise == pytest.approx(0.05)
    assert gps.fix_type == 2 and gps.ref_lat == pytest.approx(47.7)
    assert tuple(mag.sigma) == pytest.approx((0.003, 0.003, 0.003), abs=1e-9)


def test_no_sensor_prims_parses_empty(tmp_path):
    from pxr import Usd, UsdGeom

    usd = str(tmp_path / "bare.usda")
    stage = Usd.Stage.CreateNew(usd)
    UsdGeom.Xform.Define(stage, "/Vehicle/body")
    stage.GetRootLayer().Save()
    assert parse_sensor_prims(usd) == []


def test_deactivated_sensor_prim_is_skipped(tmp_path):
    """The parser must honor active=false, the standard USD way to remove a sensor in an override layer."""
    from pxr import Usd

    usd = str(tmp_path / "sensors.usda")
    _author(usd)
    stage = Usd.Stage.Open(usd)
    stage.GetPrimAtPath("/Vehicle/body/Mag").SetActive(False)
    stage.GetRootLayer().Save()
    assert [s.kind for s in parse_sensor_prims(usd)] == ["imu", "gps", "baro"]


def _build(specs):
    return build_sensors(specs, seedtree=SeedTree(42), dt=0.004, gps_init=_GPS_INIT, ref_alt=5.0)


def test_config_bugs_fail_loudly():
    with pytest.raises(ValueError, match="unknown sensor:type"):
        _build([SensorSpec("sonar")])
    with pytest.raises(TypeError):  # a typo'd sensor:* attr isn't a constructor kwarg
        _build([SensorSpec("baro", params={"nois": 0.02})])
    with pytest.raises(ValueError, match="duplicate"):  # duplicates would share seed + Measurement fields
        _build([SensorSpec("imu"), SensorSpec("imu")])
    with pytest.raises(ValueError, match="reserved"):  # the mount is the prim translation, not an attr
        _build([SensorSpec("imu", params={"mount_offset": (0.1, 0.0, 0.0)})])
    # only the Inertial Measurement Unit (IMU) has a lever arm
    with pytest.raises(ValueError, match="cannot model a mount"):
        _build([SensorSpec("gps", mount=(0.1, 0.0, 0.0))])
