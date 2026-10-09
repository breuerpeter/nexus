"""PX4 reads the sensor a connection on its prim names, where a vehicle declares two of one kind.

Real builds on the Warp CPU backend of the fixture vehicle in ``tests/usd/sensor_vehicle.py``, flown by the PX4
controller against the fake PX4 peer, which keeps the last message of each kind it receives. Skipped without
newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_sim._src.core.registry import default_registry
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

# Where a falling run starts: 2 m over the ground, where an accelerometer reads zero.
FALL_FROM = (0.0, 0.0, 2.0)
# Two Inertial Measurement Units (IMU) on the base body: a quiet one, declared first, and one whose
# accelerometer adds a noise of 0.5 m/s^2.
IMUS = sv.prim("ImuQuiet", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0") + sv.prim(
    "ImuNoisy", "NexusImuAPI", "float nexus:accNoise = 0.5"
)


@pytest.mark.xdist_group("px4_ports")  # binds or dials PX4's ports, with the other tests that do
def test_px4_reads_the_imu_a_connection_on_its_prim_names_of_two(tmp_path):
    """PX4 reads the IMU a connection on its prim names, of two.

    Given the falling fixture vehicle with two IMUs, the quiet one declared first and one with an accelerometer
    noise of 0.5 m/s^2 second, the fake PX4 peer, and PX4's prim connecting its IMU input to the noisy one, when
    the run steps 10 ticks, then the forward acceleration of each `HIL_SENSOR` PX4 receives differs from the
    others, as the noisy IMU's does.
    """
    connection = f"rel nexus:inputs:imu = <{sv.BODY}/ImuNoisy>"
    path = sv.vehicle(tmp_path, IMUS, px4=True, controller=connection)
    loop = sv.build(path, fall_from=FALL_FROM, components=default_registry(), peers={"px4_sitl": Px4Fake})
    fake = loop.peers[0]
    accels = []
    for _ in range(10):
        loop.step()
        accels.append(fake.last["HIL_SENSOR"].xacc)
    loop.close()

    assert len(set(accels)) == 10, accels
