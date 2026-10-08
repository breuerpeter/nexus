"""Each in-graph sensor samples at the rate its schema declares: the Inertial Measurement Unit (IMU), the
magnetometer, the barometer and the Global Positioning System (GPS) receiver.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on a 250 Hz tick, with the sensor prims a
test adds and a stand-in estimator whose host stage keeps what each sensor's signal holds on each tick. A new
sample shows as a tick whose sample differs from the tick before: each sample carries the sim time of the tick
that took it, so a sample the signal still holds repeats bit for bit. Each test scopes the device it uses, so
the default device is the same after it. Skipped without newton or pxr.
"""

import itertools

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.core.schema import BaroSample, GpsSample, ImuSample, MagSample
from nexus_sim._src.core.signals import Signal
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

SCHEMA = {"imu": "NexusImuAPI", "mag": "NexusMagAPI", "baro": "NexusBaroAPI", "gps": "NexusGpsAPI"}
SAMPLE = {"imu": ImuSample, "mag": MagSample, "baro": BaroSample, "gps": GpsSample}
# Where a falling run starts: 10 m over the ground, level, so the vehicle falls 4.9 m in 1 s and stays in the air.
FALL_FROM = (0.0, 0.0, 10.0)
# A rate on each of the four sensors, two of which don't divide the tick rate.
DECLARED = {"imu": 100, "mag": 50, "baro": 24, "gps": 10}


def _prim(kind: str, rate: float | None, attrs: str = "") -> str:
    """The prim of one sensor of `kind` that declares `rate` in hertz, or no rate for ``None``, and authors `attrs`."""
    declared = "" if rate is None else f"float nexus:rate = {rate}"
    return sv.prim(f"{kind.capitalize()}0", SCHEMA[kind], f"{declared}\n{attrs}")


def _rated(rates: dict) -> dict[str, str]:
    """The prim of one sensor of each kind in `rates`, which maps the kind to its rate, by kind."""
    return {kind: _prim(kind, rate) for kind, rate in rates.items()}


def _held(tmp_path, ticks: int, prims: dict[str, str], **kw) -> list[dict]:
    """Fly the fixture with the sensor prims `prims`, by kind, and return the sample each sensor's signal held, by
    kind, on the tick before the last `ticks` ticks and on each of them. `kw` goes to the build, such as
    `fall_from`, `seed` or `device`.
    """
    reader = sv.estimator(*(Signal(kind, SAMPLE[kind], shape=(1,)) for kind in prims))
    loop = sv.build(
        sv.vehicle(tmp_path, "".join(prims.values()), estimator=""),
        components=sv.components(StandInEstimatorAPI=reader),
        **kw,
    )
    sv.steps(loop, ticks + 1)
    return [dict(zip(prims, row[1:], strict=True)) for row in reader.kept]


def _raw(sample) -> tuple[bytes, ...]:
    """A sample's fields, each as its bytes, which leaves out the struct's padding: it holds no field."""
    return tuple(sample[name].tobytes() for name in sample.dtype.names)


def _new_samples(held: list[dict], kind: str) -> list[int]:
    """The ticks, counted from 1, on which the sample of the sensor `kind` differs from the tick before's."""
    raw = [_raw(tick[kind]) for tick in held]
    return [i for i in range(1, len(raw)) if raw[i] != raw[i - 1]]


def _fields(tick: dict) -> list[float]:
    """Every field of every sample a tick held, as one flat list of floats, each sample's time first."""
    return [float(x) for sample in tick.values() for name in sample.dtype.names for x in np.ravel(sample[name])]


def test_a_sensor_gives_a_new_sample_only_when_its_declared_rate_makes_one_due(tmp_path):
    """Each of the IMU, the magnetometer, the barometer and the GPS gives a new sample only when its declared rate makes one due, and its signal holds the last sample until the next.

    Given a falling fixture vehicle whose four sensors each declare `nexus:rate = 50` on a 250 Hz tick, when the
    run steps 1 s, then each sensor's sample changes on 50 ticks and repeats on the other 200.
    """
    held = _held(tmp_path, 250, _rated(dict.fromkeys(SCHEMA, 50)), fall_from=FALL_FROM)

    assert {kind: len(_new_samples(held, kind)) for kind in SCHEMA} == dict.fromkeys(SCHEMA, 50)


def test_a_rate_that_does_not_divide_the_tick_rate_neither_quantises_nor_drifts(tmp_path):
    """A rate that doesn't divide the tick rate neither quantises nor drifts.

    Given a barometer at `nexus:rate = 24` on a 250 Hz tick, when the run steps 10 s, then it gives 240 samples
    and each gap between two is 40 ms or 44 ms, 10 or 11 ticks.
    """
    ticks = _new_samples(_held(tmp_path, 2500, _rated({"baro": 24})), "baro")

    gaps = {b - a for a, b in itertools.pairwise(ticks)}
    assert (len(ticks), gaps) == (240, {10, 11})


def test_a_sensor_that_declares_no_rate_gives_a_new_sample_every_tick(tmp_path):
    """A sensor that declares no rate gives a new sample every tick.

    Given a falling fixture vehicle that authors no `nexus:rate`, when the run steps 100 ticks, then each
    sensor's sample changes on all 100.
    """
    held = _held(tmp_path, 100, _rated(dict.fromkeys(SCHEMA)), fall_from=FALL_FROM)

    assert {kind: len(_new_samples(held, kind)) for kind in SCHEMA} == dict.fromkeys(SCHEMA, 100)


def test_a_rate_above_the_tick_rate_gives_a_new_sample_every_tick(tmp_path):
    """A rate faster than the tick rate gives a new sample every tick.

    Given a barometer at `nexus:rate = 1000` on a 250 Hz tick, when the run steps 100 ticks, then its sample
    changes on all 100.
    """
    held = _held(tmp_path, 100, _rated({"baro": 1000}))

    assert len(_new_samples(held, "baro")) == 100


def test_a_negative_rate_fails_the_build_and_the_message_names_the_prim(tmp_path):
    """A negative rate fails the build, and the message names the prim.

    Given a barometer prim with `nexus:rate = -1`, when the vehicle builds, then the build raises and the
    message holds the prim's path.
    """
    path = sv.vehicle(tmp_path, _prim("baro", -1))

    with pytest.raises(ValueError, match=f"{sv.BODY}/Baro0"):
        sv.build(path).close()


@pytest.mark.gpu
def test_a_captured_run_gives_the_same_samples_as_an_eager_run_at_declared_rates(tmp_path):
    """A captured run gives the same samples as an eager run at declared rates.

    Given a falling fixture vehicle with the rates 100, 50, 24 and 10 on its four sensors, when a captured run,
    on CUDA, and an eager run, on the CPU, step 250 ticks from one seed, then every field of every sample matches
    on every tick, within the tolerance of GPU physics. A sample that one run holds and the other renews differs
    by its time, a tick or more, which the tolerance doesn't cover.
    """
    eager = _held(tmp_path, 250, _rated(DECLARED), fall_from=FALL_FROM)
    with wp.ScopedDevice("cuda:0"):
        captured = _held(tmp_path, 250, _rated(DECLARED), fall_from=FALL_FROM, device="cuda")

    assert [_fields(tick) for tick in captured] == [pytest.approx(_fields(tick), rel=1e-4, abs=1e-3) for tick in eager]


def test_an_imu_slower_than_the_tick_still_reads_the_true_specific_force(tmp_path):
    """An IMU slower than the tick still reads the true specific force.

    Given a falling fixture vehicle whose IMU declares `nexus:rate = 125` on a 250 Hz tick with zero noise, when
    the run steps 1 s, then every accelerometer sample reads zero within 1e-3 m/s^2. The test leaves out the first
    four ticks: a sample whose interval starts before the first physics step reads the vehicle at rest.
    """
    imu = _prim("imu", 125, "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0")
    held = _held(tmp_path, 250, {"imu": imu}, fall_from=FALL_FROM)

    accel = np.array([tick["imu"]["accel"] for tick in held[5:]])
    assert np.abs(accel).max() < 1e-3


def test_two_runs_from_one_seed_give_the_same_samples_bit_for_bit_at_declared_rates(tmp_path):
    """Two runs from one seed give the same samples bit for bit at declared rates.

    Given a falling fixture vehicle with the rates 100, 50, 24 and 10 on its four sensors and their noise on, when
    two runs step 250 ticks from one seed, then every sample is equal bit for bit on every tick.
    """
    first, second = (
        [{kind: _raw(sample) for kind, sample in tick.items()} for tick in held]
        for held in (_held(tmp_path, 250, _rated(DECLARED), seed=7, fall_from=FALL_FROM) for _ in range(2))
    )

    assert first == second
