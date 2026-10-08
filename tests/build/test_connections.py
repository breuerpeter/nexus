"""Where two sensors write one signal, a reader picks one through a connection on its prim, or takes them all
as a list, and a build whose reader can't tell which sensor it reads fails before any stage runs.

A connection is a relationship the reader's prim authors, named for the signal under ``nexus:inputs:``,
whose target is the prim of the sensor it reads. Real builds on the Warp CPU backend of the fixture vehicle in
``tests/usd/sensor_vehicle.py``, with the sensor prims a test adds and a stand-in estimator that the vehicle
declares on a scope of its own, and whose host stage keeps what each signal it reads holds.
Skipped without newton or pxr.
"""

import re

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_sim._src.core.schema import ImuSample, MagSample
from nexus_sim._src.core.signals import Signal
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

# Where a falling run starts: 2 m over the ground, where an accelerometer reads zero.
FALL_FROM = (0.0, 0.0, 2.0)
# Two Inertial Measurement Units (IMU) on the base body: a quiet one, declared first, and one whose
# accelerometer adds a noise of 0.5 m/s^2.
IMUS = sv.prim("ImuQuiet", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0") + sv.prim(
    "ImuNoisy", "NexusImuAPI", "float nexus:accNoise = 0.5"
)
QUIET, NOISY = f"{sv.BODY}/ImuQuiet", f"{sv.BODY}/ImuNoisy"
IMU = sv.prim("Imu0", "NexusImuAPI")
BARO = sv.prim("Baro0", "NexusBaroAPI")
MAGS = sv.prim("MagA", "NexusMagAPI") + sv.prim("MagB", "NexusMagAPI")


def _connection(signal: str, target: str) -> str:
    """The text of a connection on a reader's prim: its input `signal` reads the sensor at the prim path `target`."""
    return f"rel nexus:inputs:{signal} = <{target}>"


def _accel(sample) -> np.ndarray:
    """An IMU sample's accelerometer reading, as a flat array of floats."""
    return np.asarray(sample["accel"], dtype=float).reshape(-1)


def _zero(accel) -> bool:
    return bool(np.abs(accel).max() < 1e-3)


def _stopped(tmp_path, body: str, reader: type, *, estimator: str = "") -> str:
    """The message of the `ValueError` a run of the fixture with `body` under its base body stops with, when the
    vehicle declares the stand-in estimator class `reader` and authors `estimator` on its scope.
    """
    with pytest.raises(ValueError) as e:
        path = sv.vehicle(tmp_path, body, estimator=estimator)
        sv.steps(sv.build(path, components=sv.components(StandInEstimatorAPI=reader)), 1)
    return str(e.value)


def _names(message: str, signal: str) -> bool:
    """Whether `message` names the stand-in estimator and the signal `signal`."""
    return "StandInEstimator" in message and re.search(rf"\b{signal}\b", message) is not None


def test_of_two_imus_a_reader_reads_the_one_that_a_connection_on_its_prim_names(tmp_path):
    """Of two sensors of one kind, a reader reads the one that a connection on its prim names.

    Given the falling fixture vehicle with two Inertial Measurement Units (IMU), one quiet and one with an
    accelerometer noise of 0.5 m/s^2, and a stand-in estimator whose prim connects its IMU input to the quiet
    one, when the run steps 10 ticks, then from the second tick every accelerometer reading it gets is zero
    within 1e-3 m/s^2; and once connected to the noisy one, then its readings vary.
    """

    def readings(name: str, target: str) -> list[np.ndarray]:
        folder = tmp_path / name
        folder.mkdir()
        reader = sv.estimator(Signal("imu", ImuSample, shape=(1,)))
        path = sv.vehicle(folder, IMUS, estimator=_connection("imu", target))
        sv.steps(sv.build(path, fall_from=FALL_FROM, components=sv.components(StandInEstimatorAPI=reader)), 10)
        return [_accel(imu) for _, imu in reader.kept[1:]]

    quiet, noisy = readings("quiet", QUIET), readings("noisy", NOISY)

    assert (
        len(quiet),
        all(_zero(accel) for accel in quiet),
        len(noisy),
        len({accel.tobytes() for accel in noisy}),
    ) == (9, True, 9, 9), (quiet, noisy)


def test_a_reader_that_takes_every_imu_as_a_list_reads_each_ones_sample_on_every_tick_in_declared_order(tmp_path):
    """A reader that takes every sensor of one kind as a list reads each one's sample on every tick, in the order the vehicle declares them.

    Given the falling fixture vehicle with two Inertial Measurement Units (IMU), the quiet one declared first
    and one with an accelerometer noise of 0.5 m/s^2 second, and a stand-in estimator that reads every IMU as
    a list, when the run steps 10 ticks, then on each tick from the second it reads two samples, the first
    zero within 1e-3 m/s^2 and the second not.
    """
    reader = sv.estimator(Signal("imu", list[ImuSample], shape=(1,)))
    path = sv.vehicle(tmp_path, IMUS, estimator="")
    sv.steps(sv.build(path, fall_from=FALL_FROM, components=sv.components(StandInEstimatorAPI=reader)), 10)
    read = [
        (len(imus), _zero(_accel(imus[0])), _zero(_accel(imus[1])) if len(imus) > 1 else None)
        for _, imus in reader.kept[1:]
    ]

    assert read == [(2, True, False)] * 9, reader.kept


def test_a_list_input_with_no_sensor_of_its_kind_reads_its_default_and_without_one_fails_the_build(tmp_path):
    """A list input on a vehicle with no sensor of its kind reads its reader's default, and without one fails the build and names the reader and the signal.

    Given the fixture vehicle with no Inertial Measurement Unit (IMU) and a stand-in estimator that reads every
    IMU as a list with no default, when the vehicle builds, then it fails before any stage runs, and the error
    names the estimator and the IMU's signal; and once with a default of no samples, then the run steps 3 ticks
    and the estimator reads no sample on each.
    """
    for name in ("strict", "lenient"):
        (tmp_path / name).mkdir()
    strict = sv.estimator(Signal("imu", list[ImuSample], shape=(1,)))
    message = _stopped(tmp_path / "strict", "", strict)
    lenient = sv.estimator(Signal("imu", list[ImuSample], shape=(1,), default=[]))
    path = sv.vehicle(tmp_path / "lenient", estimator="")
    sv.steps(sv.build(path, components=sv.components(StandInEstimatorAPI=lenient)), 3)

    assert (_names(message, "imu"), strict.runs, [len(imus) for _, imus in lenient.kept]) == (True, 0, [0, 0, 0]), (
        message
    )


def test_two_magnetometers_that_a_reader_reads_through_neither_a_connection_nor_a_list_fail_the_build(tmp_path):
    """Two sensors of one kind that a reader reads through neither a connection nor a list fail the build, and the error names both prims and the reader and says that a connection on the reader's prim picks one.

    Given the fixture vehicle with two magnetometers and a stand-in estimator that reads the magnetometer with
    no connection, when the vehicle builds, then it fails before any stage runs, and the error names both
    magnetometer prims and the estimator, and says that a connection on the estimator's prim picks one.
    """
    reader = sv.estimator(Signal("mag", MagSample, shape=(1,)))
    message = _stopped(tmp_path, MAGS, reader)
    named = [f"{sv.BODY}/MagA" in message, f"{sv.BODY}/MagB" in message, "StandInEstimator" in message]

    assert (named, "connection" in message, reader.runs) == ([True, True, True], True, 0), message


def test_a_connection_that_names_a_prim_which_writes_none_of_the_signals_its_reader_reads_fails_the_build(tmp_path):
    """A connection that names a prim which writes none of the signals its reader reads fails the build, and the error names the reader, the prim and the signal.

    Given the fixture vehicle with an Inertial Measurement Unit (IMU) and a barometer, and a stand-in estimator
    whose prim connects its IMU input to the barometer, when the vehicle builds, then it fails before any stage
    runs, and the error names the estimator, the barometer's prim and the IMU's signal; and once with a
    connection to a path that holds no prim.
    """

    def stopped(name: str, target: str) -> tuple[bool, bool, int]:
        folder = tmp_path / name
        folder.mkdir()
        reader = sv.estimator(Signal("imu", ImuSample, shape=(1,)))
        message = _stopped(folder, IMU + BARO, reader, estimator=_connection("imu", target))
        return target in message, _names(message, "imu"), reader.runs

    assert (stopped("baro", f"{sv.BODY}/Baro0"), stopped("nowhere", f"{sv.BODY}/Nowhere")) == (
        (True, True, 0),
        (True, True, 0),
    )
