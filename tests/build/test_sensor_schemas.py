"""A vehicle declares each analytic sensor with one applied schema on its mount prim, and the build makes it.

Real builds on the Warp CPU
backend of the fixture vehicle in ``tests/usd/sensor_vehicle.py``: a layer over the hosted
``astro_max_base`` with the sensor prims a test adds, flown by a stand-in controller that keeps every
``Measurement`` it receives. Skipped without newton or pxr.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus._src.config.registry import load_registry
from nexus._src.core.registry import default_registry
from nexus._src.core.schema import Measurement
from nexus._src.peers.px4_sitl.fake import Px4Fake
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = ROOT / "tests" / "usd" / "stand_in"

ZURICH = (47.3769, 8.5417, 408.0)
# The World Magnetic Model (WMM) field as a North East Down (NED) vector in gauss, from PX4's coarse
# table: Zurich declination 2.6°, inclination 63.2°, strength 0.481; Woodinville, the default origin, declination
# 15.5°, inclination 69.4°, strength 0.539. The
# vehicle rests level with its nose north, so its Forward Right Down (FRD) body axes are the NED axes.
ZURICH_NED = (0.216, 0.010, 0.429)
WOODINVILLE_NED = (0.182, 0.051, 0.504)
WOODINVILLE_ALT = 5.02
FIELD_TOL = 0.015

IMU = sv.prim("Imu0", "NexusImuAPI", "float nexus:accNoise = 0.07")
MAG = sv.prim("Mag0", "NexusMagAPI", "float3 nexus:noise = (0.004, 0.005, 0.006)")
BARO = sv.prim("Baro0", "NexusBaroAPI", "float nexus:noise = 0.09")
GPS = sv.prim("Gps0", "NexusGpsAPI", "int nexus:fixType = 5")
# The same suite with no noise, so a reading is the value the site and the body give it.
QUIET = (
    sv.prim("Imu0", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0")
    + sv.prim("Mag0", "NexusMagAPI", "float3 nexus:noise = (0, 0, 0)")
    + sv.prim("Baro0", "NexusBaroAPI", "float nexus:noise = 0")
    + sv.prim("Gps0", "NexusGpsAPI")
)
QUIET_OFF_BASE = (
    sv.prim("Mag0", "NexusMagAPI", "float3 nexus:noise = (0, 0, 0)")
    + sv.prim("Baro0", "NexusBaroAPI", "float nexus:noise = 0")
    + sv.prim("Gps0", "NexusGpsAPI")
)


def _field(meas) -> tuple:
    return (meas.xmag, meas.ymag, meas.zmag)


def test_each_analytic_sensor_schema_on_an_xform_under_a_body_builds_that_sensor_with_its_authored_values(tmp_path):
    """An Inertial Measurement Unit (IMU), magnetometer, barometer or Global Positioning System (GPS) schema on an Xform under a body builds that sensor with its authored values.

    Given a fixture vehicle with one Xform per kind under the base body, each applying its schema with a
    distinctive value, when the run builds it, then the loop holds one sensor per prim, in the prims'
    order, and each carries its authored value under its keyword's name.
    """
    loop = sv.build(sv.vehicle(tmp_path, IMU + MAG + BARO + GPS))
    built = [type(s).__name__ for s in loop.sensors]
    imu, mag, baro, gps = loop.sensors if len(loop.sensors) == 4 else (None,) * 4
    values = [imu.acc_noise, *mag.noise, baro.noise, gps.fix_type]
    loop.close()

    assert (built, values) == (
        ["ImuSensor", "MagSensor", "BaroSensor", "GpsSensor"],
        pytest.approx([0.07, 0.004, 0.005, 0.006, 0.09, 5]),
    )


def test_a_sensor_built_from_a_schema_gets_the_runs_site_values(tmp_path):
    """A sensor built from a schema gets the run's site values.

    Given the fixture with every noise at zero and a catalog scene whose geodetic origin is Zurich, when
    the run steps once, then the Global Positioning System (GPS) reports the origin, the magnetometer the field at Zurich, and the
    barometer the standard pressure a quarter of a metre over mean sea level, where the base body rests.
    """
    catalog = tmp_path / "nexus.registry.yaml"
    catalog.write_text(
        f"scenes:\n  zurich:\n    geodetic_origin: {{ lat: {ZURICH[0]}, lon: {ZURICH[1]}, alt: {ZURICH[2]} }}\n"
    )
    loop = sv.build(sv.vehicle(tmp_path, QUIET), scene="zurich", registry=load_registry(catalog))

    (meas,) = sv.fly(loop, 1)

    assert (
        (meas.lat_deg, meas.lon_deg) == pytest.approx(ZURICH[:2], abs=1e-4),
        _field(meas) == pytest.approx(ZURICH_NED, abs=FIELD_TOL),
        meas.abs_pressure == pytest.approx(1013.22, abs=0.03),
    ) == (True, True, True), meas


def _samples(tmp_path, name: str, seed: int) -> list[tuple]:
    """Every sensor sample the controller receives in 100 ticks of the fixture with noise on, built with `seed`."""
    folder = tmp_path / name
    folder.mkdir()
    loop = sv.build(sv.vehicle(folder, IMU + MAG + BARO + GPS), seed=seed)
    return [dataclasses.astuple(meas) for meas in sv.fly(loop, 100)]


def test_two_runs_with_one_seed_give_equal_sensor_samples_and_another_seed_gives_different_ones(tmp_path):
    """Two runs with one seed give equal sensor samples, and another seed gives different ones.

    Given the fixture with noise on, when two runs with seed 7 and a third with seed 8 each step 100
    ticks, then the first two agree sample for sample and the third differs.
    """
    first, second, other = _samples(tmp_path, "a", 7), _samples(tmp_path, "b", 7), _samples(tmp_path, "c", 8)

    assert (len(first), first == second, first == other) == (100, True, False)


def test_a_vehicle_with_two_magnetometers_builds_both_with_their_own_noise_and_the_first_fills_the_measurement(
    tmp_path,
):
    """A vehicle can declare two sensors of one kind: both build, each draws its own noise, and the first declared fills `Measurement`.

    Given the fixture with two magnetometer prims that author offsets of 0.1 and 0.2 gauss on the
    forward axis and the same noise, when the run steps, then the loop holds two magnetometers, their
    readings less their offsets differ, and the controller's `Measurement` carries the first prim's.
    """
    noise = "float3 nexus:noise = (0.003, 0.003, 0.003)"
    mags = sv.prim("MagA", "NexusMagAPI", f"float3 nexus:offset = (0.1, 0, 0)\n{noise}") + sv.prim(
        "MagB", "NexusMagAPI", f"float3 nexus:offset = (0.2, 0, 0)\n{noise}"
    )
    loop = sv.build(sv.vehicle(tmp_path, mags))
    sensors = list(loop.sensors)

    (meas,) = sv.fly(loop, 1)
    readings = []
    for sensor in sensors:
        own = Measurement()
        sensor.read(own)
        readings.append(_field(own))
    first, second = readings if len(readings) == 2 else ((0.0,) * 3,) * 2
    first_noise = (first[0] - 0.1, first[1], first[2])
    second_noise = (second[0] - 0.2, second[1], second[2])

    assert (
        [type(s).__name__ for s in sensors],
        first_noise != pytest.approx(second_noise, abs=1e-5),
        _field(meas) == pytest.approx(first, abs=1e-6),
        meas.xmag == pytest.approx(WOODINVILLE_NED[0] + 0.1, abs=FIELD_TOL),
    ) == (["MagSensor", "MagSensor"], True, True, True), (readings, _field(meas))


def test_a_magnetometer_barometer_or_gps_reads_its_parent_body_not_body_0(tmp_path):
    """A magnetometer, barometer or Global Positioning System (GPS) reads its parent body, not body 0.

    Given a fixture with a second body 1 m over the base, rolled 180 degrees, with the three sensors
    under it and their noise at zero, when the run steps, then the barometer and the GPS report an
    altitude 1 m over the base body's, and the magnetometer the field in the second body's axes: north,
    west and up, where the base body's are north, east and down.
    """
    loop = sv.build(sv.vehicle(tmp_path, mast=QUIET_OFF_BASE))

    (meas,) = sv.fly(loop, 1)
    base_z = float(loop.physics.current_state.body_q.numpy()[0, 2])  # the base body, at rest on the ground
    north, east, down = WOODINVILLE_NED

    assert (
        meas.pressure_alt - base_z == pytest.approx(1.0, abs=0.05),
        meas.alt_m - WOODINVILLE_ALT - base_z == pytest.approx(1.0, abs=0.05),
        _field(meas) == pytest.approx((north, -east, -down), abs=FIELD_TOL),
    ) == (True, True, True), (meas.pressure_alt, meas.alt_m, base_z, _field(meas))


def test_a_barometer_prim_with_a_non_zero_translation_fails_the_build_and_names_the_prim(tmp_path):
    """A magnetometer, barometer or Global Positioning System (GPS) prim with a non-zero translation fails the build and names the prim.

    Given the fixture with the barometer's Xform moved 0.1 m from its body's origin, when built, then the
    build fails naming the prim path.
    """
    moved = sv.prim("Baro0", "NexusBaroAPI", translate=(0.1, 0, 0))

    with pytest.raises(ValueError) as err:
        sv.build(sv.vehicle(tmp_path, moved)).close()

    assert f"{sv.BODY}/Baro0" in str(err.value)


def test_a_sensor_schema_on_a_prim_whose_parent_is_not_a_rigid_body_fails_the_build_and_names_the_prim(tmp_path):
    """A sensor schema on a prim whose parent isn't a rigid body fails the build and names the prim.

    Given the fixture with the Global Positioning System (GPS) Xform under a plain Xform, itself under the base body, when built,
    then the build fails naming the GPS prim's path.
    """
    bracket = 'def Xform "Bracket"\n{\n' + sv.prim("Gps", "NexusGpsAPI") + "}\n"

    with pytest.raises(ValueError) as err:
        sv.build(sv.vehicle(tmp_path, bracket)).close()

    assert f"{sv.BODY}/Bracket/Gps" in str(err.value)


def test_an_imu_prim_whose_transform_scales_fails_the_build_and_names_the_prim(tmp_path):
    """An Inertial Measurement Unit (IMU) prim whose transform scales or shears fails the build and names the prim.

    Given the fixture with a scale of 2 on the IMU's prim, when built, then the build fails naming the prim
    path.
    """
    scaled = sv.prim(
        "Imu0", "NexusImuAPI", 'float3 xformOp:scale = (2, 2, 2)\nuniform token[] xformOpOrder = ["xformOp:scale"]'
    )

    with pytest.raises(ValueError) as err:
        sv.build(sv.vehicle(tmp_path, scaled)).close()

    assert f"{sv.BODY}/Imu0" in str(err.value)


def test_the_ground_truth_px4_receives_is_the_base_bodys_whatever_the_imus_mount_and_body(tmp_path):
    """The ground truth PX4 receives is the base body's attitude and rates in PX4's frames, whatever the Inertial Measurement Unit's (IMU) mount and body.

    Given the fixture vehicle level and nose north, the IMU's prim turned 90 degrees about z under the
    second body with a gyro noise of 0.5 rad/s, a Global Positioning System (GPS) receiver on the base body
    and the PX4 Software In The Loop (SITL) peer mapped to its fake, when the run steps 100 ticks, then the last
    `HIL_STATE_QUATERNION` the fake receives carries the identity attitude, the base body's Forward Right
    Down (FRD) axes on North East Down (NED), and body rates under 1e-3 rad/s.
    """
    turned = sv.prim(
        "Imu0",
        "NexusImuAPI",
        'float nexus:gyroNoise = 0.5\nfloat xformOp:rotateZ = 90\nuniform token[] xformOpOrder = ["xformOp:rotateZ"]',
    )
    path = sv.vehicle(tmp_path, sv.prim("Gps0", "NexusGpsAPI"), mast=turned, px4=True)
    loop = sv.build(path, components=default_registry(), peers={"px4_sitl": Px4Fake})
    fake = loop.peers[0]

    n = 0
    while n < 100 and loop.step():
        n += 1
    state = fake.last.get("HIL_STATE_QUATERNION")
    loop.close()
    q = list(state.attitude_quaternion) if state else [0.0] * 4
    q = [-x for x in q] if q[0] < 0 else q  # q and -q are one attitude
    rate = max(abs(state.rollspeed), abs(state.pitchspeed), abs(state.yawspeed)) if state else 1.0

    assert (n, q == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-3), rate < 1e-3) == (100, True, True), (q, rate)


def test_a_vehicle_that_declares_no_sensor_builds(tmp_path):
    """A vehicle that declares no sensor builds.

    Given the fixture with every sensor prim deactivated and none added, when built, then the build
    succeeds and the loop holds no sensor.
    """
    loop = sv.build(sv.vehicle(tmp_path))
    sensors = list(loop.sensors)
    loop.close()

    assert sensors == []


def test_a_registry_handed_to_the_builder_decides_an_analytic_and_an_rtx_sensors_class(tmp_path):
    """A registry handed to the builder decides every sensor's class, analytic and RTX.

    Given a registry that maps the Global Positioning System (GPS) schema and the camera schema to a stand-in, and a fixture with an
    Inertial Measurement Unit (IMU), a GPS and a camera, when built, then the loop holds the IMU and one stand-in for each of the
    other two prims.
    """
    prims = IMU + sv.prim("Gps0", "NexusGpsAPI") + sv.prim("Cam", "NexusCameraAPI", kind="Camera")
    registry = sv.components(NexusGpsAPI=sv.StandInSensor, NexusCameraAPI=sv.StandInSensor)

    loop = sv.build(sv.vehicle(tmp_path, prims), components=registry)
    built = sorted(type(s).__name__ for s in loop.sensors)
    loop.close()

    assert built == ["ImuSensor", "StandInSensor", "StandInSensor"]


def test_an_analytic_sensor_still_samples_every_tick_whatever_rate_it_declares(tmp_path):
    """An analytic sensor still samples every tick, whatever rate it declares.

    Given the fixture with a rate of 5 Hz authored on the barometer, whose noise is on, when the run
    steps 10 ticks of 0.004 s, then the controller receives 10 different pressures, one fresh sample a tick.
    """
    baro = sv.prim("Baro0", "NexusBaroAPI", "float nexus:noise = 0.02\nfloat nexus:rate = 5")
    loop = sv.build(sv.vehicle(tmp_path, baro))

    pressures = [meas.abs_pressure for meas in sv.fly(loop, 10)]

    assert len(set(pressures)) == 10, pressures


def test_a_projects_own_sensor_schema_builds_its_class_with_the_runs_values_and_the_loop_samples_it(tmp_path):
    """A project's own sensor schema builds its class with the run's values, and the loop samples it.

    Given the stand-in package under `tests/usd/stand_in`, installed with a sensor schema and an entry
    in the entry-point group, when runs with seeds 7, 7 and 8 build a fixture applying the schema with a
    gain of 2.5 and step two ticks, then the class gets a tick of 0.004 s, the same seed in the first two
    runs and another in the third, and its stage writes the gain into the `Measurement` the controller
    receives. A child process, so the installed package changes no module state here.
    """
    site = tmp_path / "site"
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--target", str(site), str(STAND_IN)],
        check=True,
        capture_output=True,
    )
    code = f"""
import pathlib
import warp as wp
import nexus
from tests.usd import sensor_vehicle as sv

prims = sv.prim("Own", "StandInSensorAPI", "float nexus:gain = 2.5")
seeds, ticks, gains = [], [], []
with wp.ScopedDevice("cpu"):
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        folder = pathlib.Path({str(tmp_path)!r}) / name
        folder.mkdir()
        loop = sv.build(sv.vehicle(folder, prims), seed=seed)
        (own,) = loop.sensors
        seeds.append(own.seed)
        ticks.append(own.dt)
        gains.append(sv.fly(loop, 2)[-1].eph)
print(">", type(own).__module__, seeds[0] == seeds[1], seeds[0] != seeds[2], ticks, gains)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(ROOT)])}
    r = subprocess.run([sys.executable, "-c", code], check=False, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    (line,) = [line[2:] for line in r.stdout.splitlines() if line.startswith("> ")]

    assert line == "nexus_stand_in.sensor True True [0.004, 0.004, 0.004] [2.5, 2.5, 2.5]"
