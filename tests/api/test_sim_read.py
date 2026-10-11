"""A script reads a signal's live value by its path, typed by the signal, and the run's sim time, #280.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, flown by a
stand-in controller named ``stand_in`` and hosted by ``Sim``. Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.signals import Signal
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

COMMAND = (0.25, 0.5, 0.75, 1.0)  # the stand-in's controls, exact in float32
NOTE = {"mode": "hover"}  # the object the stand-in writes to its host signal


class _Named(sv.Controller):
    """The stand-in controller, named `stand_in`: it writes `COMMAND` to its controls and `NOTE` to the host
    signal `note`, so its signals sit at `vehicle/controllers/stand_in/<signal>`.
    """

    name = "stand_in"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.note = Signal("note", dict)

    def stages(self):
        return [Stage("exchange", "host", self._exchange, writes=(self.controls, self.note))]

    def exchange(self, t, timeout=None):
        self.note.write(NOTE)
        return np.array(COMMAND, dtype=np.float32)


def _sim(tmp_path, *, fall_from=(0.0, 0.0, 5.0)):
    """A `Sim` of the fixture vehicle in free fall from `fall_from`, flown by the named stand-in."""
    loop = sv.build(sv.vehicle(tmp_path), components=sv.components(NexusPx4API=_Named), fall_from=fall_from)
    return nx.Sim._from_loop(loop)


def test_a_struct_signal_reads_as_one_instance_of_its_type(tmp_path):
    """A struct signal reads as one instance of its type.

    Given the fixture vehicle stepped 10 ticks, when a script calls `sim.read("vehicle/body/<base>")`, then it
    gets a `PoseTwist` whose `.position` equals `sim[path].history()[-1]["position"]`.
    """
    with _sim(tmp_path) as sim:
        for _ in range(10):
            sim.step()
        path = f"vehicle/body/{sim.base_body}"
        value, newest = sim.read(path), sim[path].history()[-1]["position"]

    assert isinstance(value, type(nx.PoseTwist())) and list(value.position) == pytest.approx(list(newest))


def test_a_value_signal_reads_without_its_leading_axis_of_one(tmp_path):
    """A value signal reads without its leading axis of one.

    Given a stand-in controller that writes known `controls` of width n, when a script calls `sim.read` on its
    `controls` path after a tick, then it gets an array of shape `(n,)` equal to them.
    """
    with _sim(tmp_path) as sim:
        sim.step()
        value = sim.read("vehicle/controllers/stand_in/controls")

    assert np.asarray(value).tolist() == [0.25, 0.5, 0.75, 1.0]


def test_a_host_signal_reads_by_its_path_though_it_keeps_no_history(tmp_path):
    """A host signal reads by its path, though it keeps no history.

    Given a stand-in component that writes an object to a host signal, when a script calls `sim.read` on that
    signal's path after a tick, then it gets that object.
    """
    with _sim(tmp_path) as sim:
        sim.step()
        value = sim.read("vehicle/controllers/stand_in/note")

    assert value == {"mode": "hover"}


def test_a_value_keeps_what_it_held_when_the_script_read_it(tmp_path):
    """A value keeps what it held when the script read it.

    Given a value from `sim.read` on the base body, when the sim steps 10 more ticks while the body falls, then
    the value read first keeps its position.
    """
    with _sim(tmp_path) as sim:
        sim.step()
        value = sim.read(f"vehicle/body/{sim.base_body}")
        height = None if value is None else float(value.position[2])
        for _ in range(10):
            sim.step()

    assert height is not None and float(value.position[2]) == height


def test_a_path_that_names_no_signal_fails_with_the_path(tmp_path):
    """A path that names no signal fails with the path.

    Given a started run, when a script calls `sim.read("vehicle/nope")`, then a `KeyError` names `vehicle/nope`.
    """
    with _sim(tmp_path) as sim, pytest.raises(KeyError, match="vehicle/nope"):
        sim.read("vehicle/nope")


def test_sim_read_before_the_run_fails_as_indexing_does(tmp_path):
    """`sim.read` before the run fails as `sim[path]` does.

    Given a `Sim` outside its context, when a script calls `sim.read` on any path, then a `RuntimeError` says to
    enter the context.
    """
    sim = _sim(tmp_path)
    with pytest.raises(RuntimeError, match="context"):
        sim.read("vehicle/body/body")


def test_sim_time_is_the_sim_time_of_the_last_tick(tmp_path):
    """`sim.time` is the sim time of the last tick.

    Given a run at a 4 ms tick, when a script reads `sim.time` before the first tick and after 25 ticks, then it
    gets `0.0`, then `0.1`, equal to the base body's `history()[-1]["t"]`.
    """
    with _sim(tmp_path) as sim:
        before = sim.time
        for _ in range(25):
            sim.step()
        after, newest = sim.time, float(sim[f"vehicle/body/{sim.base_body}"].history()[-1]["t"])

    assert (before, after, newest) == (0.0, pytest.approx(0.1), pytest.approx(0.1))
