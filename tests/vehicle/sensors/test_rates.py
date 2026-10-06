"""Each in-graph sensor samples at the rate its schema declares: the Inertial Measurement Unit (IMU), the
magnetometer, the barometer and the Global Positioning System (GPS).

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on a 250 Hz tick: a layer over the
hosted ``astro_max_base`` with the sensor prims a test adds, flown by a stand-in controller that keeps
every ``Measurement`` it receives. A new sample shows as a tick whose fields differ from the tick before:
the noise of the IMU, the magnetometer and the barometer makes each sample its own, and the GPS has no
noise, so its vehicle climbs at full throttle. Skipped without newton or pxr.
"""

import itertools

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

SCHEMA = {"imu": "NexusImuAPI", "mag": "NexusMagAPI", "baro": "NexusBaroAPI", "gps": "NexusGpsAPI"}
# The `Measurement` fields each sensor writes.
FIELDS = {
    "imu": ("xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"),
    "mag": ("xmag", "ymag", "zmag"),
    "baro": ("abs_pressure", "pressure_alt"),
    "gps": ("lat_deg", "lon_deg", "alt_m", "vn", "ve", "vd"),
}
# The ticks a run steps before the ones a test counts, so the vehicle has left the ground.
LEAD = 20


class _Hop(sv.Climb):
    """Full throttle for 100 ticks, then none: the vehicle is in free fall from tick 110 until it lands after tick 200."""

    burn = 100


def _prims(**rates) -> str:
    """One prim per sensor kind in `rates`, each declaring its rate in hertz, or no rate for ``None``."""
    return "".join(
        sv.prim(f"{kind.capitalize()}0", SCHEMA[kind], "" if rate is None else f"float nexus:rate = {rate}")
        for kind, rate in rates.items()
    )


def _fly(tmp_path, prims: str, ticks: int, *, controller=sv.Climb, **kw) -> list:
    """Fly the fixture with `prims` and return the `Measurement` of each of its last `ticks` ticks and of the tick before them."""
    loop = sv.build(sv.vehicle(tmp_path, prims), components=sv.components(NexusPx4API=controller), **kw)
    return sv.fly(loop, LEAD + ticks)[-(ticks + 1) :]


def _values(received, kinds=FIELDS) -> list[tuple]:
    """The fields of the sensors `kinds` in each `Measurement`, one flat tuple a tick."""
    return [tuple(getattr(meas, field) for kind in kinds for field in FIELDS[kind]) for meas in received]


def _new_samples(received, kind: str) -> list[int]:
    """The ticks, counted from 1, on which the fields of the sensor `kind` differ from the tick before."""
    values = _values(received, (kind,))
    return [i for i in range(1, len(values)) if values[i] != values[i - 1]]


def test_a_sensor_gives_a_new_sample_only_when_its_declared_rate_makes_one_due(tmp_path):
    """Each of the IMU, the magnetometer, the barometer and the GPS gives a new sample only when its declared rate makes one due, and `Measurement` holds the last value between samples.

    Given a climbing fixture vehicle whose four sensors each declare `nexus:rate = 50` on a 250 Hz tick,
    when the run steps 1 s, then each sensor's `Measurement` fields change on 50 ticks and repeat on the
    other 200.
    """
    received = _fly(tmp_path, _prims(imu=50, mag=50, baro=50, gps=50), 250)

    assert {kind: len(_new_samples(received, kind)) for kind in FIELDS} == dict.fromkeys(FIELDS, 50)


def test_a_rate_that_does_not_divide_the_tick_rate_neither_quantises_nor_drifts(tmp_path):
    """A rate that doesn't divide the tick rate neither quantises nor drifts.

    Given a barometer at `nexus:rate = 24` on a 250 Hz tick, on a vehicle at rest, when the run steps
    10 s, then it gives 240 samples and each gap between two is 40 ms or 44 ms, 10 or 11 ticks.
    """
    received = _fly(tmp_path, _prims(baro=24), 2500, controller=sv.Controller)

    ticks = _new_samples(received, "baro")
    gaps = {b - a for a, b in itertools.pairwise(ticks)}
    assert (len(ticks), gaps) == (240, {10, 11})


def test_a_sensor_that_declares_no_rate_gives_a_new_sample_every_tick(tmp_path):
    """A sensor that declares no rate gives a new sample every tick.

    Given a climbing fixture vehicle that authors no `nexus:rate`, when the run steps 100 ticks, then
    each sensor's `Measurement` fields change on all 100.
    """
    received = _fly(tmp_path, _prims(imu=None, mag=None, baro=None, gps=None), 100)

    assert {kind: len(_new_samples(received, kind)) for kind in FIELDS} == dict.fromkeys(FIELDS, 100)


def test_a_rate_above_the_tick_rate_gives_a_new_sample_every_tick(tmp_path):
    """A rate faster than the tick rate gives a new sample every tick.

    Given a barometer at `nexus:rate = 1000` on a 250 Hz tick, when the run steps 100 ticks, then its
    fields change on all 100.
    """
    received = _fly(tmp_path, _prims(baro=1000), 100)

    assert len(_new_samples(received, "baro")) == 100


def test_a_negative_rate_fails_the_build_and_the_message_names_the_prim(tmp_path):
    """A negative rate fails the build, and the message names the prim.

    Given a barometer prim with `nexus:rate = -1`, when the vehicle builds, then the build raises and
    the message holds the prim's path.
    """
    path = sv.vehicle(tmp_path, _prims(baro=-1))

    with pytest.raises(ValueError, match=f"{sv.BODY}/Baro0"):
        sv.build(path).close()


def test_a_captured_run_gives_the_same_samples_as_an_eager_run_at_declared_rates(tmp_path):
    """A captured run gives the same samples as an eager run at declared rates.

    Given a climbing fixture vehicle with the rates 100, 50, 24 and 10 on its four sensors, when a
    captured run, on CUDA, and an eager run, on the CPU, step 250 ticks from one seed, then every
    `Measurement` field matches on every tick, within the tolerance of GPU physics. A sample that one run
    holds and the other renews differs by its noise, or by a tick of the climb, which is far more.
    Auto-skips without CUDA.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    prims = _prims(imu=100, mag=50, baro=24, gps=10)

    eager = _values(_fly(tmp_path, prims, 250))
    with wp.ScopedDevice("cuda:0"):
        captured = _values(_fly(tmp_path, prims, 250, device="cuda"))

    assert captured == [pytest.approx(row, rel=1e-4, abs=1e-3) for row in eager]


def test_an_imu_slower_than_the_tick_still_reads_the_true_specific_force(tmp_path):
    """An IMU slower than the tick still reads the true specific force.

    Given a fixture vehicle whose IMU declares `nexus:rate = 125` on a 250 Hz tick with zero noise, flown
    at full throttle for 100 ticks and then at none, when the run steps the 80 ticks of its free fall
    from tick 120, then every accelerometer sample reads zero within 0.05 m/s^2.
    """
    imu = sv.prim("Imu0", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0\nfloat nexus:rate = 125")
    loop = sv.build(sv.vehicle(tmp_path, imu), components=sv.components(NexusPx4API=_Hop))

    fall = sv.fly(loop, 200)[120:]

    assert max(abs(a) for meas in fall for a in (meas.xacc, meas.yacc, meas.zacc)) < 0.05


def test_two_runs_from_one_seed_give_the_same_samples_bit_for_bit_at_declared_rates(tmp_path):
    """Two runs from one seed give the same samples bit for bit at declared rates.

    Given a climbing fixture vehicle with the rates 100, 50, 24 and 10 on its four sensors and their
    noise on, when two runs step 250 ticks from one seed, then every `Measurement` field is equal bit
    for bit on every tick.
    """
    prims = _prims(imu=100, mag=50, baro=24, gps=10)

    first, second = (_values(_fly(tmp_path, prims, 250, seed=7)) for _ in range(2))

    assert first == second
