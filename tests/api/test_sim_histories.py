"""A script reads any component's recorded history by its path below the run's root, through one view, and
``Sim`` gives no role an accessor of its own, #44.

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


def test_a_script_reads_any_components_history_by_its_path_through_one_view(tmp_path):
    """A script reads any component's recorded history by its path below the run's root through one view.

    Given a recorded run on the fixture vehicle, when it ends, then the IMU's history reads at
    `vehicle/sensors/imu/imu` and the controller's controls at `vehicle/controllers/stand_in/controls` with
    `.latest()` and `.history()`: the IMU's newest row is the last tick's, and the controls hold the seed row
    and one row per tick.
    """
    with nx.Sim.from_orchestrator(_build(tmp_path)) as sim:
        for _ in range(TICKS):
            sim.step()
        view = sim.histories
        read = view is not None and (
            float(view["vehicle/sensors/imu/imu"].latest()["t"]),
            len(view["vehicle/controllers/stand_in/controls"].history()),
        )

    assert read == (pytest.approx(TICKS * DT), TICKS + 1)


def test_sim_gives_no_role_an_accessor_of_its_own(tmp_path):
    """`sim.sensors` and `sim.guidance` are gone: `Sim` gives no role an accessor of its own.

    Given a run on the fixture vehicle, when a script looks for `sensors` and `guidance` on `Sim`, then neither is
    an attribute of it, so a read of either raises `AttributeError`.
    """
    with nx.Sim.from_orchestrator(_build(tmp_path)) as sim:
        present = [name for name in ("sensors", "guidance") if hasattr(type(sim), name)]

    assert present == []
