"""A script reads any recorded history, a component's or the plant's, by indexing the run with its path below the
run's root, and ``Sim`` gives no role and no fixed part an accessor of its own, #44.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, with an Inertial
Measurement Unit (IMU) and a stand-in controller, hosted by ``Sim``. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

DT = 0.004  # the fixture's 250 Hz tick
TICKS = 5


class _Named(sv.Controller):
    """The stand-in controller, named `stand_in`, so its rows land under `vehicle/controllers/stand_in`."""

    name = "stand_in"


def _build(tmp_path):
    """A run of the fixture with an IMU on its base body, flown by the named stand-in controller."""
    registry = sv.components(NexusPx4API=_Named)
    return sv.build(sv.vehicle(tmp_path, sv.prim("Imu", "NexusImuAPI")), components=registry)


def test_a_script_reads_any_history_by_indexing_the_run_with_its_path(tmp_path):
    """A script reads any recorded history, a component's or the plant's, by indexing the run with the history's
    path below the run's root.

    Given a recorded run on the fixture vehicle, when it ends, then `sim["vehicle/sensors/imu/imu"]` is the IMU's
    history, `sim["vehicle/controllers/stand_in/controls"]` the controller's and `sim["vehicle/body/body"]` the base
    body's, each read by its newest row and by all its rows: the IMU's newest row is the last tick's, and the controls and
    the body each hold the seed row and one row per tick.
    """
    with nx.Sim._from_loop(_build(tmp_path)) as sim:
        for _ in range(TICKS):
            sim.step()
        imu, controls, body = (
            sim[path]
            for path in ("vehicle/sensors/imu/imu", "vehicle/controllers/stand_in/controls", "vehicle/body/body")
        )
        read = None not in (imu, controls, body) and (
            float(imu.history()[-1]["t"]),
            len(controls.history()),
            len(body.history()),
        )

    assert read == (pytest.approx(TICKS * DT), TICKS + 1, TICKS + 1)


def test_sim_gives_no_role_and_no_fixed_part_an_accessor_of_its_own(tmp_path):
    """`sim.sensors`, `sim.physics` and `sim.guidance` went: a script indexes the run by path, so `Sim` gives
    no role and no fixed part an accessor of its own.

    Given a run on the fixture vehicle, when a script looks for `sensors`, `physics` and `guidance` on `Sim`,
    then none is an attribute of it, so a read of any raises `AttributeError`.
    """
    with nx.Sim._from_loop(_build(tmp_path)) as sim:
        present = [name for name in ("sensors", "physics", "guidance") if hasattr(type(sim), name)]

    assert present == []
