"""``Sim`` takes a registry value beside the catalog and the layer, #44.

A real build of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, through
``Sim``. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.vehicle.forces.propellers import Propeller
from nexus_sim._src.vehicle.sensors import ImuSensor
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")


class _StandInImu:
    """A stand-in for the Inertial Measurement Unit (IMU): its one host stage notes that it ran."""

    ran = False

    def __init__(self, run, **kwargs):
        pass  # the IMU schema's keyword arguments, which it doesn't read

    def stages(self):
        def note(tick):
            type(self).ran = True

        return [Stage("note", "host", note)]


def test_a_schema_mapped_in_the_registry_sim_takes_builds_its_class_and_the_default_registry_is_untouched(tmp_path):
    """`Sim` takes a registry value beside the catalog and the layer, and a schema mapped there builds its class in
    place of the installed one, with the default registry untouched.

    Given a registry built from the default with the fixture vehicle's IMU schema mapped to a stand-in class, its
    controller's to the stand-in that answers at once and its propellers' to the shipped class, when
    `Sim(..., registry=...)` runs on the CPU device, then the stand-in's stage ran, and the default registry,
    `Registry()`, still resolves the IMU schema to the shipped class.
    """
    registry = nx.Registry({"NexusImuAPI": _StandInImu, "NexusPx4API": sv.Controller, "NexusPropellerAPI": Propeller})
    vehicle = sv.vehicle(tmp_path, sv.prim("Imu", "NexusImuAPI"))
    with nx.Sim(vehicle, scene=sv.SCENE, device="cpu", registry=registry) as sim:
        sim.start(timeout=5.0)

    assert (_StandInImu.ran, nx.Registry().resolve("NexusImuAPI")) == (True, ImuSensor)
