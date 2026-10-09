"""The Recorder keeps each signal's history, and no component records itself, #250.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py``, on the Warp CPU backend and, where a
row says so, captured on a CUDA device, with the sensor prims a test adds, an Inertial Measurement Unit (IMU)
or a Global Positioning System (GPS) receiver among them, the passthrough estimator and a stand-in
controller. A test reads the histories through ``Sim`` and the loop's Recorder, and a recorded run
through the ``.rrd`` it wrote. Each test scopes the device it uses, so the default device is the same after
it. Skipped without newton or pxr.
"""

import contextlib
import functools

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

import nexus_sim as nx
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.vehicle.estimators import GroundTruthEstimator
from tests.conftest import fly_recorded
from tests.usd import sensor_vehicle as sv

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
DT = 0.004  # the fixture's 250 Hz tick
TICKS = 5
# The fixture scene's site, Woodinville: its latitude, which a GPS at the origin reads.
WOODINVILLE_LAT = 47.747944
IMU = sv.prim("Imu", "NexusImuAPI")
GPS = sv.prim("Gps", "NexusGpsAPI")
# An IMU at a quarter of the tick rate: a sample every fourth tick.
RATED_IMU = sv.prim("Imu", "NexusImuAPI", "float nexus:rate = 62.5")
# Two IMUs on the base body: a quiet one and one whose accelerometer adds a noise of 0.5 m/s^2.
TWO_IMUS = sv.prim("imu_a", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0") + sv.prim(
    "imu_b", "NexusImuAPI", "float nexus:accNoise = 0.5"
)
# A camera and a lidar on the base body, which the run names `fpvcam` and `lidar`.
FPVCAM = sv.prim("FpvCam", "NexusCameraAPI", kind="Camera")
LIDAR = sv.prim("Lidar", "NexusLidarAPI", "float nexus:rate = 25", kind="OmniLidar")
POINT = [1.0, 2.0, 3.0]


class _TickWriter:
    """A controller named `standin` whose host stage writes the tick's step index into each of its 16 controls."""

    name = "standin"

    def __init__(self, **kwargs):
        self.controls = Signal("controls", wp.float32, shape=(1, 16))

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("act", "host", self._act, writes=(self.controls,))]

    def _act(self, tick):
        self.controls.write(np.full((1, 16), tick.t.step_index, dtype=np.float32))
        return True


class _Windy(GroundTruthEstimator):
    """The passthrough estimator, named `truth`, whose stage writes a second device signal, `wind`, beside its
    estimate.
    """

    name = "truth"

    def __init__(self):
        super().__init__()
        self.wind = Signal("wind", wp.vec3, shape=(1,))

    def stages(self):
        return [Stage("estimate", "device", self._run, writes=(self.estimate, self.wind))]


def _build(tmp_path, body: str, *, device: str = "cpu", controller=None, estimator=None):
    """A run of the fixture with `body` under its base body on `device`, flown by `controller`, by default the
    stand-in that commands nothing, over `estimator`, by default the passthrough.
    """
    registry = sv.components() if controller is None else sv.components(NexusPx4API=controller)
    loop = sv.build(sv.vehicle(tmp_path, body), device=device, components=registry)
    loop.estimator = GroundTruthEstimator() if estimator is None else estimator
    return loop


@contextlib.contextmanager
def _flown(loop, ticks: int):
    """`loop` hosted by `Sim` and stepped `ticks` ticks, open for a test to read, then closed."""
    with nx.Sim.from_orchestrator(loop) as sim:
        for _ in range(ticks):
            assert sim.step()
        yield sim


def _structured(rows) -> np.ndarray:
    """`rows`, a history's rows, as one structured array, or an empty one when they read as anything else."""
    arr = np.asarray(rows)
    return arr if arr.dtype.names else np.empty(0, dtype=[("t", "f8")])


def _history(histories: dict, signal: str) -> np.ndarray:
    """The rows of the one history whose key ends in the signal's name, structured, or none."""
    key = next((k for k in histories if k.endswith(f"/{signal}")), None)
    return _structured(histories[key].history() if key is not None else [])


def _column(rows: np.ndarray, name: str) -> np.ndarray:
    """The column `name` of structured rows, or an empty array when they have none."""
    return rows[name] if name in (rows.dtype.names or ()) else np.empty(0)


def _flat(rows: np.ndarray, *names: str) -> list[list[float]]:
    """Each row's fields `names`, flattened into one list, or no rows when a field is missing."""
    if any(name not in (rows.dtype.names or ()) for name in names):
        return []
    return [[float(x) for name in names for x in np.ravel(row[name])] for row in rows]


def _record(row):
    """`row`, a history's newest row, when it's a structured record, else None."""
    return row if getattr(getattr(row, "dtype", None), "names", None) else None


def _typed(record, name: str) -> tuple[str, int] | None:
    """The dtype and the number of values of the field `name` of a record, or None without the field."""
    if record is None or name not in record.dtype.names:
        return None
    field = np.asarray(record[name])
    return str(field.dtype), int(field.size)


def _bits(values) -> np.ndarray:
    """The values as 32-bit floats, viewed as their bits, so two arrays compare bit for bit."""
    return np.ascontiguousarray(np.array(values, dtype=np.float32)).view(np.uint32)


@pytest.mark.parametrize("device", DEVICES)
def test_every_device_signal_a_component_writes_keeps_a_history_and_no_component_records_itself(device, tmp_path):
    """Every device signal a component writes keeps a history in the Recorder, the seed row and one row per
    tick, and no component records itself.

    Given the fixture vehicle with an IMU, the passthrough estimator and a stand-in controller whose host
    stage writes its tick count into its 16 controls, when the run steps 5 ticks, eagerly on the CPU device
    and captured on a CUDA device, then the Recorder holds a history for the IMU's sample, the estimate and
    the controls, each of 6 rows whose `t` runs from 0 to `5 * dt`, the controls' rows hold 0 then 1 to 5,
    and each estimate row equals, to the bit, the base body's position and velocity of that tick: the state
    the tick's flight stack read, which is the settled state for the seed row and the body's row before for
    each tick, since the estimator runs before the physics steps and the record stage after.
    """
    with wp.ScopedDevice(device):
        loop = _build(tmp_path, IMU, device="cpu" if device == "cpu" else "cuda", controller=_TickWriter)
        with _flown(loop, TICKS) as sim:
            imu, estimate, controls = (
                _history(loop.recorder.histories, signal) for signal in ("imu", "estimate", "controls")
            )
            rows = sim.physics[sim.base_body].history()
            truth = [[*r.position, *r.velocity] for r in (rows[0], *rows[:-1])]
    read = (
        [_column(rows, "t").tolist() for rows in (imu, estimate, controls)],
        [row[0] for row in _flat(controls, "controls")],
        np.array_equal(_bits(_flat(estimate, "position", "linear_velocity")), _bits(truth)),
    )

    assert read == (
        [pytest.approx([k * DT for k in range(TICKS + 1)])] * 3,
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        True,
    )


@pytest.mark.usefixtures("warp_cpu")
def test_a_historys_key_is_its_writers_path_then_the_signals_name_and_the_clock_keeps_none(tmp_path):
    """A history's key is its writer's path then the signal's name, one rule for every component, and the loop's
    clock keeps none.

    Given the run of the preceding row, its estimator a stand-in that writes a second device signal `wind` beside
    its estimate, when it steps, then the keys are `vehicle/sensors/imu/imu`, `vehicle/estimators/<name>/estimate`,
    `vehicle/estimators/<name>/wind` and `vehicle/controllers/standin/controls`, no history holds the clock's
    `time`, and `sim.sensors["imu"]` resolves the IMU's history.
    """
    loop = _build(tmp_path, IMU, controller=_TickWriter, estimator=_Windy())
    with _flown(loop, 1) as sim:
        histories = loop.recorder.histories
        keys = sorted(k for k in histories if not k.startswith(("vehicle/body/", "vehicle/joints/")))
        resolved = sim.sensors["imu"] is histories.get("vehicle/sensors/imu/imu")

    assert (keys, resolved) == (
        [
            "vehicle/controllers/standin/controls",
            "vehicle/estimators/truth/estimate",
            "vehicle/estimators/truth/wind",
            "vehicle/sensors/imu/imu",
        ],
        True,
    )


@pytest.mark.usefixtures("warp_cpu")
def test_a_historys_row_is_one_record_t_then_the_signals_quantities_as_its_type_declares_them(tmp_path):
    """A history's row is one record, `t` then the signal's quantities, each with the type and width its
    signal's type declares, and a value-typed signal is one quantity named after the signal.

    Given the fixture vehicle with an IMU and a GPS at a site at latitude 47.3977, and the stand-in controller,
    when the run steps one tick, then the newest row of `sim.sensors["imu"]` is a record whose dtype names `t`,
    `time`, `accel` and `gyro`, with `accel` and `gyro` float32 of width 3, the `lat` of the newest row of
    `sim.sensors["gps"]` is float64 within 1e-9 degrees of the site's, the controls' newest row holds
    `controls`, 16 float32, and `history()` gives the rows as one array and `arrays()` one array per quantity.
    The fixture scene's site is Woodinville, so the latitude read is its.
    """
    loop = _build(tmp_path, IMU + GPS)
    with _flown(loop, 1) as sim:
        imu, gps = (_record(sim.sensors[name].latest()) for name in ("imu", "gps"))
        key = next((k for k in loop.recorder.histories if k.endswith("/controls")), None)
        controls = _record(loop.recorder.histories[key].latest()) if key is not None else None
        rows = sim.sensors["imu"].history()
        arrays = sim.sensors["imu"].arrays()
    lat = float(gps["lat"]) if gps is not None and "lat" in gps.dtype.names else None
    read = (
        tuple(imu.dtype.names) if imu is not None else (),
        [_typed(imu, "accel"), _typed(imu, "gyro")],
        (_typed(gps, "lat"), lat is not None and abs(lat - WOODINVILLE_LAT) < 1e-9),
        _typed(controls, "controls"),
        isinstance(rows, np.ndarray) and rows.dtype.names is not None and len(rows) == 2,
        {name: (isinstance(a, np.ndarray), len(a)) for name, a in arrays.items()},
    )

    assert read == (
        ("t", "time", "accel", "gyro"),
        [("float32", 3), ("float32", 3)],
        (("float64", 1), True),
        ("float32", 16),
        True,
        {"t": (True, 2), "time": (True, 2), "accel": (True, 2), "gyro": (True, 2)},
    )


@pytest.mark.usefixtures("warp_cpu")
def test_a_sensors_history_holds_on_each_tick_the_sample_its_signal_holds_with_its_own_time(tmp_path):
    """A sensor's history holds, on each tick, the sample its signal holds, so the rows between two due ticks
    repeat the last sample with its own time.

    Given the fixture vehicle at 250 Hz with an IMU at 62.5 Hz, when the run steps 8 ticks, then each row's
    `time` is the sim time of the last tick at which a sample was due, 0, `4 * dt` or `8 * dt`, and the rows
    between two due ticks hold equal `accel` and `gyro`.
    """
    loop = _build(tmp_path, RATED_IMU)
    with _flown(loop, 8) as sim:
        rows = _structured(sim.sensors["imu"].history())
    samples = [tuple(row) for row in _flat(rows, "accel", "gyro")]
    held = [len(set(samples[a:b])) for a, b in ((0, 4), (4, 8))] if len(samples) == 9 else []

    assert (_column(rows, "time").tolist(), held) == (
        pytest.approx([0.0] * 4 + [4 * DT] * 4 + [8 * DT], abs=1e-9),
        [1, 1],
    )


@pytest.mark.usefixtures("warp_cpu")
def test_a_signal_with_no_fixed_width_keeps_no_history_and_its_component_logs_it_live_at_the_signals_row(
    tmp_path, rrd_rows
):
    """A signal with no fixed width keeps no history, and its component logs it live at the row named after the
    signal.

    Given the fixture vehicle with a camera and a lidar on the Kit fake and a stand-in controller that reads a
    planned reference, when the run steps 5 ticks and ends, then the Recorder holds no history for the camera's
    frame, the lidar's scan or the reference, and the `.rrd` holds the camera's frames at
    `sim/vehicle/sensors/fpvcam/camera`, its frustum at `sim/vehicle/sensors/fpvcam`, and the lidar's points at
    `sim/vehicle/sensors/lidar/lidar`. A history shows in the `.rrd` as a series under its writer's path, so none
    sits under the camera's, the lidar's or the guidance's.
    """
    from nexus_sim._src.peers.kit.fake import KitFake

    kit = functools.partial(KitFake, points=[POINT])
    rrd = fly_recorded(tmp_path, sv.vehicle(tmp_path, FPVCAM + LIDAR), scene=sv.SCENE, ticks=30, kit=kit)
    rows = rrd_rows(rrd)
    no_history = ("/sim/vehicle/sensors/fpvcam", "/sim/vehicle/sensors/lidar", "/sim/guidance/setpoint")
    framed = sorted({e for e, _, c in rows if "EncodedImage:blob" in c})
    frustums = sorted({e for e, _, c in rows if "Pinhole:image_from_camera" in c})
    pointed = sorted({e for e, _, c in rows if "Points3D:positions" in c and e.startswith("/sim/vehicle/sensors/")})
    series = sorted({e for e, _, c in rows if "Scalars:scalars" in c and e.startswith(no_history)})

    assert (framed, frustums, pointed, series) == (
        ["/sim/vehicle/sensors/fpvcam/camera"],
        ["/sim/vehicle/sensors/fpvcam"],
        ["/sim/vehicle/sensors/lidar/lidar"],
        [],
    )


@pytest.mark.usefixtures("warp_cpu")
@pytest.mark.xdist_group("px4_ports")  # binds or dials PX4's ports, with the other tests that do
def test_an_input_no_component_writes_keeps_no_history(tmp_path):
    """An input no component writes keeps no history.

    Given the fixture vehicle with an IMU and no GPS on the PX4 fake, when the run steps 3 ticks, then the
    Recorder holds no history whose key ends in `gps`.
    """
    from nexus_sim._src.core.registry import default_registry
    from nexus_sim._src.peers.px4_sitl.fake import Px4Fake

    loop = sv.build(sv.vehicle(tmp_path, IMU, px4=True), components=default_registry(), peers={"px4_sitl": Px4Fake})
    with _flown(loop, 3):
        keys = sorted(k for k in loop.recorder.histories if k.endswith("gps"))

    assert keys == []


@pytest.mark.usefixtures("warp_cpu")
def test_two_sensors_of_one_kind_record_apart_each_under_its_own_instance(tmp_path):
    """Two sensors of one kind record apart, each under its own instance.

    Given the fixture vehicle with two IMUs on the prims `imu_a`, quiet, and `imu_b`, with an accelerometer
    noise of 0.5 m/s^2, when the run steps 5 ticks, then `sim.sensors["imu_a"]` and `sim.sensors["imu_b"]` are
    two histories whose accelerometer rows differ from the second row on.
    """
    loop = _build(tmp_path, TWO_IMUS)
    with _flown(loop, TICKS) as sim:
        a, b = sim.sensors["imu_a"], sim.sensors["imu_b"]
        rows_a, rows_b = a.arrays(), b.arrays()
    differ = [k for k in rows_a if k != "t" and not np.array_equal(rows_a[k][1:], rows_b.get(k, rows_a[k])[1:])]

    assert (a is not b, sorted(rows_a) == sorted(rows_b), differ != []) == (True, True, True)
