"""Each sensor samples at the rate its schema declares, by one due-time rule: the in-graph Inertial Measurement
Unit (IMU), magnetometer, barometer and Global Positioning System (GPS) receiver, and the RTX camera and lidar.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on a 250 Hz tick, with the sensor prims a
test adds and a stand-in estimator whose host stage keeps what each sensor's signal holds on each tick. A new
sample shows as a tick whose sample differs from the tick before: each sample carries the sim time of the tick
that took it, so a sample the signal still holds repeats bit for bit. An RTX sensor's frames come from the Kit
peer's fake, one due tick after the tick they show. Each test scopes the device it uses, so the default device
is the same after it. Skipped without newton or pxr.
"""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_uav._src.core.schema import BaroSample, GpsSample, Image, ImuSample, MagSample, PointCloud
from nexus_uav._src.core.signals import Signal
from nexus_uav._src.peers.kit.fake import KitFake
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

SCHEMA = {"imu": "NexusImuAPI", "mag": "NexusMagAPI", "baro": "NexusBaroAPI", "gps": "NexusGpsAPI"}
SAMPLE = {"imu": ImuSample, "mag": MagSample, "baro": BaroSample, "gps": GpsSample}
# The RTX sensors: the schema, the frame's type, a host signal, and the prim type the schema applies to.
RTX = {"camera": ("NexusCameraAPI", Image, "Camera"), "lidar": ("NexusLidarAPI", PointCloud, "OmniLidar")}
# The peer mapping of a run with an RTX sensor: the Kit peer goes to its fake, which starts no container, and
# whose every lidar scan holds one point.
KIT = {"kit": KitFake.configured(points=[(1.0, 2.0, 3.0)])}
TICK_HZ = 250
# Where a falling run starts: 10 m over the ground, level, so the vehicle falls 4.9 m in 1 s and stays in the air.
FALL_FROM = (0.0, 0.0, 10.0)
# A rate on each of the four sensors, two of which don't divide the tick rate.
DECLARED = {"imu": 100, "mag": 50, "baro": 24, "gps": 10}


def _prim(kind: str, rate: float | None, attrs: str = "") -> str:
    """The prim of one sensor of `kind` that declares `rate` in hertz, or no rate for ``None``, and authors `attrs`.

    An RTX camera's prim is a `Camera` of 8x6 pixels, and an RTX lidar's an `OmniLidar`.
    """
    declared = "" if rate is None else f"float nexus:rate = {rate}"
    if kind in RTX:
        schema, _, prim_type = RTX[kind]
        size = "int nexus:width = 8\nint nexus:height = 6" if kind == "camera" else ""
        return sv.prim(f"{kind.capitalize()}0", schema, f"{declared}\n{size}\n{attrs}", kind=prim_type)
    return sv.prim(f"{kind.capitalize()}0", SCHEMA[kind], f"{declared}\n{attrs}")


def _signal(kind: str) -> Signal:
    """The signal the sensor of `kind` writes, for the stand-in estimator to read."""
    if kind in RTX:
        return Signal(kind, RTX[kind][1])
    return Signal(kind, SAMPLE[kind], shape=(1,))


def _rated(rates: dict) -> dict[str, str]:
    """The prim of one sensor of each kind in `rates`, which maps the kind to its rate, by kind."""
    return {kind: _prim(kind, rate) for kind, rate in rates.items()}


def _held(tmp_path, ticks: int, prims: dict[str, str], **kw) -> list[dict]:
    """Fly the fixture with the sensor prims `prims`, by kind, and return the sample each sensor's signal held, by
    kind, on the tick before the last `ticks` ticks and on each of them. The start's reading stays out. `kw` goes
    to the build, such as `fall_from`, `seed` or `device`.
    """
    reader = sv.estimator(*(_signal(kind) for kind in prims))
    loop = sv.build(
        sv.vehicle(tmp_path, "".join(prims.values()), estimator=""),
        components=sv.components(StandInEstimatorAPI=reader),
        **kw,
    )
    sv.steps(loop, ticks + 1)
    return [dict(zip(prims, row[1:], strict=True)) for row in reader.kept[1:]]


def _record(sample) -> np.void:
    """A sample, an instance of its struct, as one numpy record of its fields."""
    return np.array(sample.numpy_value(), dtype=sample.numpy_dtype())[()]


def _raw(sample) -> tuple[bytes, ...]:
    """A sample's fields, each as its bytes, which leaves out the struct's padding: it holds no field."""
    record = _record(sample)
    return tuple(record[name].tobytes() for name in record.dtype.names)


def _new_samples(held: list[dict], kind: str) -> list[int]:
    """The ticks, counted from 1, on which the sample of the sensor `kind` differs from the tick before's."""
    raw = [_raw(tick[kind]) for tick in held]
    return [i for i in range(1, len(raw)) if raw[i] != raw[i - 1]]


def _sampled(held: list[dict], kind: str, window: int) -> list[int]:
    """The ticks that took the samples the sensor `kind`'s signal held, in order, each its sim time over the tick,
    from the first sample's tick through the `window` ticks after it, the last left out. An RTX sensor's signal
    holds nothing before its first frame.
    """
    times = []
    for tick in held:
        sample = tick[kind]
        if sample is None:
            continue
        t = float(sample.time)
        if t not in times:
            times.append(t)
    ticks = [round(t * TICK_HZ) for t in times]
    return [tick for tick in ticks if tick < ticks[0] + window] if ticks else []


def _fields(tick: dict) -> list[float]:
    """Every field of every sample a tick held, as one flat list of floats, each sample's time first."""
    records = [_record(sample) for sample in tick.values()]
    return [float(x) for record in records for name in record.dtype.names for x in np.ravel(record[name])]


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


def test_a_sensor_whose_rate_divides_the_tick_rate_samples_on_every_nth_tick(tmp_path):
    """A sensor whose rate divides the tick rate samples on every n-th tick.

    Given the falling fixture with a GPS at `nexus:rate = 10` on a 250 Hz tick, when the run steps 10 s, then it
    gives 100 samples, each 25 ticks after the one before.
    """
    ticks = _new_samples(_held(tmp_path, 2500, _rated({"gps": 10}), fall_from=FALL_FROM), "gps")

    gaps = {b - a for a, b in itertools.pairwise(ticks)}
    assert (len(ticks), gaps) == (100, {25})


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


def test_an_imu_slower_than_the_tick_reads_the_true_specific_force_from_the_first_sample_a_tick_takes(tmp_path):
    """An IMU slower than the tick reads the true specific force from the first sample a tick takes.

    Given the falling fixture whose IMU declares `nexus:rate = 125` on a 250 Hz tick with zero noise, when the run
    steps 1 s, then every accelerometer sample stamped after `0.0` reads zero within 1e-3 m/s^2, the first ticks
    included. The sample the start takes, at `0.0`, has no earlier velocity to difference.
    """
    imu = _prim("imu", 125, "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0")
    held = _held(tmp_path, 250, {"imu": imu}, fall_from=FALL_FROM)

    accel = np.array([[*tick["imu"].accel] for tick in held if float(tick["imu"].time) > 0.0])
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


# Ten simulated seconds, and the run's ticks: 50 more, so the frames in flight at 10 s reach their signals.
TEN_S = 10 * TICK_HZ
RUN = TEN_S + 50


def test_an_rtx_sensor_whose_rate_divides_the_tick_rate_takes_a_frame_on_every_nth_tick(tmp_path):
    """An RTX sensor whose rate divides the tick rate takes a frame on every n-th tick.

    Given an RTX lidar at `nexus:rate = 10` on a 250 Hz tick, on the Kit peer's fake, when the run steps 10 s,
    then its output signal gets 100 scans, each 25 ticks after the one before.
    """
    ticks = _sampled(_held(tmp_path, RUN, _rated({"lidar": 10}), peers=KIT), "lidar", TEN_S)

    gaps = {b - a for a, b in itertools.pairwise(ticks)}
    assert (len(ticks), gaps) == (100, {25})


def test_a_sensor_whose_rate_does_not_divide_the_tick_rate_takes_each_sample_on_the_first_tick_at_or_after_its_exact_due_time(
    tmp_path,
):
    """A sensor whose rate doesn't divide the tick rate takes each sample on the first tick at or after its exact due time, the RTX sensors too.

    Given a barometer and an RTX camera, each at `nexus:rate = 24` on a 250 Hz tick, when the run steps 10 s, then
    each gives 240 samples, and sample k falls on the first tick at or after the first sample's time plus k/24 s,
    computed in exact fractions.
    """
    held = _held(tmp_path, RUN, _rated({"baro": 24, "camera": 24}), peers=KIT)
    sampled = {kind: _sampled(held, kind, TEN_S) for kind in ("baro", "camera")}

    exact = {
        kind: [ticks[0] + math.ceil(Fraction(k * TICK_HZ, 24)) for k in range(240)] for kind, ticks in sampled.items()
    }
    assert sampled == exact


def test_an_rtx_sensor_at_rate_0_takes_a_frame_every_tick(tmp_path):
    """An RTX sensor at `nexus:rate = 0` takes a frame every tick.

    Given an RTX camera at `nexus:rate = 0` on a 250 Hz tick, on the Kit peer's fake, when the run steps 100
    ticks, then it raises nothing and its output signal gets a frame on each of the 100 ticks.
    """
    raised, held = None, []
    try:
        held = _held(tmp_path, 101, _rated({"camera": 0}), peers=KIT)
    except Exception as exc:  # the row asks that the run raise nothing, so a raise is a result to compare
        raised = repr(exc)

    assert (raised, _sampled(held, "camera", 100)) == (None, list(range(1, 101)))


def test_an_rtx_sensor_faster_than_the_tick_takes_a_frame_every_tick(tmp_path):
    """An RTX sensor faster than the tick takes a frame every tick.

    Given an RTX camera at `nexus:rate = 1000` on a 250 Hz tick, on the Kit peer's fake, when the run steps 100
    ticks, then its output signal gets a frame on each of the 100 ticks.
    """
    held = _held(tmp_path, 101, _rated({"camera": 1000}), peers=KIT)

    assert _sampled(held, "camera", 100) == list(range(1, 101))


def test_an_rtx_sensor_at_a_negative_rate_fails_the_build_and_the_message_names_its_prim(tmp_path):
    """An RTX sensor at a negative rate fails the build, and the message names its prim.

    Given an RTX camera prim with `nexus:rate = -1`, when the vehicle builds, then the build raises and the
    message names the camera's prim path.
    """
    path = sv.vehicle(tmp_path, _prim("camera", -1))

    with pytest.raises(ValueError, match=f"{sv.BODY}/Camera0"):
        sv.build(path, peers=KIT).close()
