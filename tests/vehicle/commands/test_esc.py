"""The rotors of ``astro_max_base`` turn as the real Astro Max's do: at each motor output, the speed its ESC telemetry
records, and not at all before PX4 arms, #377.

Real builds of the hosted ``astro_max_base`` through ``Sim`` on the Warp CPU backend, its PX4 SITL peer sent to the
fake. Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_uav as nx
from nexus_uav._src.peers.px4_sitl import fake
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from nexus_uav._src.vehicle.rotors import RPM_PER_RADS
from tests.usd import sensor_vehicle as sv

pytestmark = [pytest.mark.usefixtures("warp_cpu"), pytest.mark.xdist_group("px4_ports")]

HELD_S = 2.0  # how long each test holds its motor outputs, sim time [s]


def _rotor_speeds(sim) -> np.ndarray:
    """Each rotor joint's recorded speed [rpm], one row per tick, one column per rotor."""
    rotors = [path for path in sim if path.startswith("vehicle/joints/rotor")]
    return np.abs(np.stack([sim[path].history()["qd"][:, 0] for path in rotors], axis=1)) * RPM_PER_RADS


@pytest.mark.parametrize(("output", "rpm"), [(0.40, 1930.0), (0.52, 2285.0), (0.68, 2735.0)])
def test_each_rotor_turns_at_the_real_astro_max_speed_for_its_motor_output(tmp_path, output, rpm):
    """Each rotor of `astro_max_base` turns at the real Astro Max's speed for each motor output: 1930, 2285 and
    2735 rpm at outputs 0.40, 0.52 and 0.68, within 5%, as two bare Astro Max flights' ESC telemetry records.

    Given `astro_max_base` with a stand-in controller that holds every motor output at one value, when the run
    steps 2 s, then each rotor's speed reads the real one's within 5%.
    """

    class Hold(sv.Controller):
        """A stand-in controller: it answers at once with every motor output at `output`."""

        def exchange(self, t, timeout=None):
            return np.full(self.width, output, dtype=np.float32)

    registry = nx.Registry({"NexusPx4API_1": Hold, "NexusPx4SitlAPI": Px4Fake})
    with nx.Sim("astro_max_base", scene="empty", device="cpu", registry=registry, run_dir=tmp_path / "run") as sim:
        sim.start(timeout=10.0)
        sim.sleep(HELD_S)
        speeds = _rotor_speeds(sim)[-1]

    np.testing.assert_allclose(speeds, rpm, rtol=0.05)


def test_the_rotors_stand_still_before_px4_arms(tmp_path, monkeypatch):
    """Before PX4 arms, the rotors stand still.

    Given `astro_max_base` and the PX4 link's fake streaming disarmed actuator controls, the message PX4 SITL sends
    before it arms, every control 0 and no armed flag in its mode, when the run steps 2 s, then every rotor's
    speed reads 0.
    """
    monkeypatch.setattr(fake, "HOVER", 0.0)  # the fake streams its one command on every channel, with mode 0
    registry = nx.Registry({"NexusPx4SitlAPI": Px4Fake})
    with nx.Sim("astro_max_base", scene="empty", device="cpu", registry=registry, run_dir=tmp_path / "run") as sim:
        sim.start(timeout=10.0)
        sim.sleep(HELD_S)
        fastest = float(_rotor_speeds(sim).max())

    assert fastest == pytest.approx(0.0, abs=1.0)
