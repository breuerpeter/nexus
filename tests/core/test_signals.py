"""Components declare typed signals, and the builder wires them once: each stage declares the signals it
reads and writes, the loop hands the writer and every reader one buffer before the capture, and a
declaration that doesn't fit stops the run before any stage runs. Stand-in components at the loop's
roles, and the fixture quad's real physics where the run's recording shows that no stage ran; each
CUDA test scopes the device it uses, so the default device is the same after it.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Component, Stage
from nexus_sim._src.core.loop import Loop
from nexus_sim._src.core.schema import ReferenceTrajectory, SimTime
from nexus_sim._src.core.signals import Signal, wire
from nexus_sim._src.core.stages import Bound
from nexus_sim._src.guidance import TrackingGuidance

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@wp.struct
class Count:
    """A signal type this test defines and core doesn't name: one 32-bit count."""

    n: wp.int32


@wp.kernel
def _count(n: wp.array(dtype=wp.int32), out: wp.array(dtype=Count)):
    n[0] = n[0] + 1
    c = Count()
    c.n = n[0]
    out[0] = c


@wp.kernel
def _copy(src: wp.array(dtype=Count), dst: wp.array(dtype=wp.int32)):
    dst[0] = src[0].n


class _Clock:
    dt = 0.004

    def __init__(self):
        self._t = SimTime()

    def now(self):
        return self._t

    def advance(self):
        self._t = SimTime(self._t.time_ns + round(self.dt * 1e9), self._t.step_index + 1)
        return self._t

    def throttle(self):
        pass


class _State:
    """The physics state: one body at rest at the origin, which a guidance reads."""

    def __init__(self):
        self.body_q = wp.array(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


class _Physics:
    """Physics that moves nothing."""

    def reset(self):
        return _State()

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: None),
            Stage("step", "device", lambda tick: None),
            Stage("state", "device", lambda tick: None),
        ]


class _Counter(Component):
    """A sensor whose device stage counts its runs and writes the count to the signal `count`. The start runs
    it once, so on the k-th tick the count is k + 1.
    """

    def __init__(self, name: str = "counter"):
        self.name = name
        self.count = Signal("count", Count, shape=(1,))
        self._n = wp.zeros(1, dtype=wp.int32)

    def stages(self):
        return [Stage("count", "device", self._run, writes=(self.count,))]

    def _run(self, tick):
        wp.launch(_count, dim=1, inputs=[self._n], outputs=[self.count.buffer])


class _DeviceReader(Component):
    """A controller whose device stage copies the signal `count` into a buffer of its own, `seen`."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.seen = wp.zeros(1, dtype=wp.int32)

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "device", self._read, reads=(self.count,))]

    def _read(self, tick):
        wp.launch(_copy, dim=1, inputs=[self.count.buffer], outputs=[self.seen])


class _HostReader(Component):
    """A controller whose host stage reads the signal `count` with a copy. It keeps each value it read."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.seen = []

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "host", self._read, reads=(self.count,))]

    def _read(self, tick):
        self.seen.append(int(self.count.buffer.numpy()[0]["n"]))
        return True


class _KeepingReader(Component):
    """A controller whose host stage keeps each value of the signal `count` it reads, as read gave it."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.kept = []

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        def keep(tick):
            self.kept.append(self.count.read())
            return True

        return [Stage("keep", "host", keep, reads=(self.count,))]


class _Reference:
    """A planned reference, the shape a tracking guidance writes."""

    duration = 4.0

    def set_start(self, p0):
        pass

    def reference_path(self):
        return []


class _SteerOnDevice(Component):
    """A controller whose device stage, `steer`, reads the planned reference, which is a host signal."""

    def __init__(self):
        self.setpoint = Signal("setpoint", ReferenceTrajectory)

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("steer", "device", lambda tick: None, reads=(self.setpoint,))]


class _CountReader(Component):
    """A controller whose device stage reads the signal `count` and writes the fixture quad's four controls."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.controls = Signal("controls", wp.float32, shape=(1, 4))

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "device", lambda tick: None, reads=(self.count,), writes=(self.controls,))]


def _orch(sensors, controller, **kw):
    return Loop(clock=_Clock(), physics=_Physics(), sensors=sensors, controller=controller, **kw)


def _stopped(tmp_path, controller, *, sensors=(), guidance=None) -> tuple[str, int]:
    """Fly the fixture quad of `tests/vehicle/quad.py` around `controller`, with `sensors` beside its own, the
    passthrough estimator and `guidance`, on real physics, and return the message of the `ValueError` the run stops with and the
    histories its recording holds: the loop taps the signals once it wires the ring, so a run that stopped
    before any stage ran holds none.
    """
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    from nexus_sim._src.api.sim import Sim
    from nexus_sim._src.build.assembly import assemble, resolve_device
    from nexus_sim._src.physics import NewtonPhysics
    from nexus_sim._src.physics.vehicle import VehicleUsd
    from nexus_sim._src.vehicle.estimators import GroundTruthEstimator
    from tests.usd.sensor_vehicle import scenario
    from tests.vehicle import quad

    cfg = scenario()
    cfg["physics"]["force_cpu"] = True
    resolve_device(cfg)
    vehicle_usd = VehicleUsd({"usd_path": str(quad.author(tmp_path / "quad.usda"))})
    physics = NewtonPhysics(vehicle_usd=vehicle_usd, cfg=cfg)
    a = assemble(physics, vehicle_usd, cfg, controller=controller)
    orch = Loop(
        clock=a.clock,
        physics=a.physics,
        commands=a.commands,
        forces=a.forces,
        sensors=[*a.sensors, *sensors],
        estimator=GroundTruthEstimator(),
        controller=a.controller,
        max_steps=quad.FLIGHT_STEPS,
    )
    with Sim._from_loop(orch, guidance=guidance) as sim:
        with pytest.raises(ValueError) as e:
            sim.run()
        return str(e.value), len(sim)


@pytest.mark.parametrize("device", DEVICES)
def test_a_component_reads_a_value_another_writes_of_a_type_core_does_not_name(device):
    """A component reads a value another component writes, of a type core doesn't name, with no change to
    the loop.

    Given a stand-in sensor whose device stage writes its tick count to a signal of a type the test defines,
    and a stand-in controller whose device stage reads it, when the run steps 3 ticks, eagerly on the CPU
    device and captured on a CUDA device, then on each tick the controller reads the count the sensor wrote
    on that tick: 2, 3 and 4, after the start's 1.
    """
    with wp.ScopedDevice(device):
        controller = _DeviceReader()
        orch = _orch([_Counter()], controller)
        seen = []
        for _ in range(3):
            orch.step()
            seen.append(int(controller.seen.numpy()[0]))
        orch.close()
    assert seen == [2, 3, 4]


@pytest.mark.parametrize("device", DEVICES)
def test_a_host_stage_reads_the_value_a_device_stage_wrote_earlier_in_the_same_tick(device):
    """A host stage reads the value a device stage wrote earlier in the same tick.

    Given a stand-in sensor whose device stage writes its tick count to a signal, and a stand-in controller
    whose host stage reads it, when the run steps 3 ticks, eagerly on the CPU device and captured on a CUDA
    device, then on each tick the controller reads the count the sensor wrote on that tick: 2, 3 and 4, after
    the start's 1.
    """
    with wp.ScopedDevice(device):
        controller = _HostReader()
        orch = _orch([_Counter()], controller)
        seen = []
        for _ in range(3):
            orch.step()
            seen.append(controller.seen[-1])
        orch.close()
    assert seen == [2, 3, 4]


@pytest.mark.parametrize("device", DEVICES)
def test_a_host_stage_keeps_the_copy_it_read_after_a_later_write(device):
    """A host stage reads a device signal with a copy, so what it keeps doesn't change at a later write.

    Given a stand-in sensor whose device stage writes its tick count to a signal, and a stand-in controller
    whose host stage keeps each value it reads, when the run steps 3 ticks, eagerly on the CPU device and
    captured on a CUDA device, then the controller keeps 2, 3 and 4, after the start's 1.
    """
    with wp.ScopedDevice(device):
        controller = _KeepingReader()
        orch = _orch([_Counter()], controller)
        for _ in range(3):
            orch.step()
        orch.close()
    assert [int(value[0]["n"]) for value in controller.kept[-3:]] == [2, 3, 4]


@pytest.mark.usefixtures("warp_cpu")
def test_a_device_stage_that_reads_a_host_signal_fails_the_run_naming_the_stage_and_the_signal(tmp_path):
    """A device stage that reads or writes a host signal fails the run before any stage runs, and the error
    names the stage and the signal.

    Given a `TrackingGuidance` and a stand-in controller whose device stage reads the `ReferenceTrajectory` it
    writes, when the run starts, then it fails before any stage runs, and the error names that stage and the
    setpoint. It flies the fixture quad, and no stage ran, so the recording holds no history.
    """
    guidance = TrackingGuidance(planner=lambda waypoints: _Reference())
    guidance.set_mission([(1.0, 0.0, 2.0)])
    message, histories = _stopped(tmp_path, _SteerOnDevice(), guidance=guidance)
    assert (all(word in message for word in ("steer", "setpoint")), histories) == (True, 0)


@pytest.mark.usefixtures("warp_cpu")
def test_two_components_that_write_one_signal_a_third_reads_fail_the_run_naming_both_writers(tmp_path):
    """Two components that write one signal that a third reads fail the run before any stage runs, and the
    error names both writers.

    Given two stand-in sensors whose device stages write one signal and a stand-in controller that reads it,
    when the run starts, then it fails before any stage runs, and the error names both sensors. It flies the
    fixture quad, and no stage ran, so the recording holds no history.
    """
    writers = [_Counter("first_counter"), _Counter("second_counter")]
    message, histories = _stopped(tmp_path, _CountReader(), sensors=writers)
    assert (all(name in message for name in ("first_counter", "second_counter")), histories) == (True, 0)


class _Writer(Component):
    """A sensor on the prim `path`, named after its leaf, whose host stage writes `value` to the signal `count`, or
    to the signal named `signal`.
    """

    def __init__(self, path: str, value: int, signal: str = "count"):
        self.prim_path = path
        self.name = path.rsplit("/", 1)[-1]
        self.value = value
        self.count = Signal(signal, Count, shape=(1,))

    def stages(self):
        return [Stage("write", "host", lambda tick: self.count.write([(self.value,)]), writes=(self.count,))]


class _Reader(Component):
    """An estimator on the prim `/vehicle/Estimator` whose host stage reads the signal `count`, as `type`."""

    prim_path = "/vehicle/Estimator"

    def __init__(self, type=Count):
        self.count = Signal("count", type, shape=(1,))

    def stages(self):
        return [Stage("read", "host", lambda tick: None, reads=(self.count,))]


def test_a_connection_wires_an_input_to_a_writers_output_of_another_name():
    """A connection wires an input to a writer's output of another name, and without one the input has no writer.

    Given a sensor whose host stage writes 7 to its signal `tally` and an estimator whose host stage reads
    `count`, when the loop wires them with a link from the estimator's `count` to the sensor's `tally`, then
    the estimator reads 7; and with no link, the wiring fails and names the estimator and `count`.
    """
    writer, reader = _Writer("/vehicle/body/A", 7, signal="tally"), _Reader()
    pairs = [(writer, "sensor"), (reader, "estimator")]
    ring = [Bound(stage, c, role) for c, role in pairs for stage in c.stages()]
    wire(ring, [(reader, "count", writer, "tally")])
    writer.count.write([(7,)])
    read = int(reader.count.read()[0]["n"])
    with pytest.raises(ValueError) as e:
        wire(ring)

    assert (read, "_Reader" in str(e.value), "'count'" in str(e.value)) == (7, True, True)


def _read(writers, reader, connections=None):
    """What `reader` reads once `writers` have each written their value, wired as the loop wires them, with
    `connections`, by the reader's prim, the signal and the prim it names, which the builder resolves to the
    component on each prim before the loop wires.
    """
    pairs = [*((w, "sensor") for w in writers), (reader, "estimator")]
    by_prim = {c.prim_path: c for c, _ in pairs}
    links = [
        (by_prim[reader_prim], signal, by_prim[writer_prim], signal)
        for reader_prim, inputs in (connections or {}).items()
        for signal, writer_prim in inputs.items()
    ]
    wire([Bound(stage, c, role) for c, role in pairs for stage in c.stages()], links)
    for w in writers:
        w.count.write([(w.value,)])
    return reader.count.read()


@pytest.mark.usefixtures("warp_cpu")
def test_of_two_writers_of_a_signal_a_reader_reads_the_one_a_connection_on_its_prim_names():
    """Of two writers of a signal, a reader reads the one a connection on its prim names.

    Given two stand-in sensors on the prims `A` and `B` that write 1 and 2 to the signal `count`, and a reader
    whose prim connects `count` to `B`, when the loop wires them, then the reader reads 2.
    """
    writers = [_Writer("/vehicle/body/A", 1), _Writer("/vehicle/body/B", 2)]
    read = _read(writers, _Reader(), {"/vehicle/Estimator": {"count": "/vehicle/body/B"}})

    assert int(read[0]["n"]) == 2


@pytest.mark.usefixtures("warp_cpu")
def test_a_list_input_reads_every_writers_value_in_the_order_of_the_ring():
    """A list input reads every writer's value, in the order of the ring.

    Given two stand-in sensors on the prims `A` and `B` that write 1 and 2 to the signal `count`, in that
    order, and a reader that reads `count` as a list, when the loop wires them, then the reader reads 1 and 2.
    """
    writers = [_Writer("/vehicle/body/A", 1), _Writer("/vehicle/body/B", 2)]
    read = _read(writers, _Reader(list[Count]))

    assert [int(value[0]["n"]) for value in read] == [1, 2]


@pytest.mark.usefixtures("warp_cpu")
def test_two_writers_a_reader_takes_through_neither_a_connection_nor_a_list_fail_the_wiring_naming_both():
    """Two writers of a signal that a reader takes through neither a connection nor a list fail the wiring, and
    the error names both writers' prims and the reader, and says a connection on the reader's prim picks one.

    Given two stand-in sensors on the prims `A` and `B` that write the signal `count`, and a reader of it with
    no connection, when the loop wires them, then it fails, naming both prims, the reader and a connection.
    """
    with pytest.raises(ValueError) as e:
        _read([_Writer("/vehicle/body/A", 1), _Writer("/vehicle/body/B", 2)], _Reader())

    named = [word in str(e.value) for word in ("A", "B", "_Reader", "connection")]
    assert named == [True] * 4


@pytest.mark.usefixtures("warp_cpu")
class _Silent(Component):
    """A sensor on the prim `path`, named after its leaf, whose host stage writes nothing."""

    def __init__(self, path: str):
        self.prim_path = path
        self.name = path.rsplit("/", 1)[-1]

    def stages(self):
        return [Stage("idle", "host", lambda tick: None)]


def test_a_connection_to_a_component_that_writes_no_such_signal_fails_the_wiring_naming_the_reader_writer_and_signal():
    """A connection to a component that writes none of the signal fails the wiring, and the error names the
    reader, the component and the signal.

    Given a stand-in sensor on the prim `A` that writes the signal `count`, a sensor `C` that writes nothing,
    and a reader whose prim connects `count` to `C`, when the loop wires them, then it fails, naming the reader,
    `C` and `count`.
    """
    with pytest.raises(ValueError) as e:
        _read(
            [_Writer("/vehicle/body/A", 1), _Silent("/vehicle/body/C")],
            _Reader(),
            {"/vehicle/Estimator": {"count": "/vehicle/body/C"}},
        )

    assert [word in str(e.value) for word in ("_Reader", "C", "'count'")] == [True] * 3
