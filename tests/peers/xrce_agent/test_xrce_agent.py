"""The Micro XRCE-DDS agent peer: the container it starts, on the port of the run's instance, which PX4 dials,
and that its stop removes it.

The docker daemon is the system boundary: a stand-in records what the peers ask of it, and touches nothing real.
"""

import pytest
from docker.errors import NotFound

import nexus_uav._src.peers.containers as containers
from nexus_uav._src.peers.pins import PeerPin
from nexus_uav._src.peers.px4_sitl.runner import Px4Sitl
from nexus_uav._src.peers.xrce_agent.runner import XrceAgent


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    """A stand-in docker daemon that holds every image: the containers the peers ran, each its keywords, and the
    names they removed. Home is a test folder, so a start empties no real instance's store.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runs, removed = [], []

    class Container:
        def __init__(self, name):
            self.name = name

        def remove(self, force=False):
            removed.append(self.name)

        def stop(self, timeout=None):
            pass

    class Images:
        def get(self, tag):
            return object()

    class Containers:
        def run(self, image, **kwargs):
            runs.append({"image": image, **kwargs})
            return Container(kwargs.get("name")) if kwargs.get("detach", True) else b""

        def get(self, name):
            if name in [r.get("name") for r in runs] and name not in removed:
                return Container(name)
            raise NotFound(name)

        def list(self, **kwargs):
            return []

    class Client:
        images = Images()
        containers = Containers()

    client = Client()
    monkeypatch.setattr(containers, "client", lambda: client)
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    return runs, removed


def test_a_declared_agent_listens_on_the_port_of_the_runs_instance_which_px4_dials_and_stops_with_the_run(
    daemon, tmp_path
):
    """A run whose vehicle declares the agent starts it on the port of the run's instance and stops it with the
    run, and PX4 dials that port.

    Given the agent peer and the PX4 peer of one run, each on instance 2, when they start, then the agent's command
    listens on UDP 8890 and PX4's environment holds `PX4_UXRCE_DDS_PORT` 8890. When the agent stops, then the
    daemon removes its container.
    """
    runs, removed = daemon
    tree = tmp_path / "px4"
    tree.mkdir()
    (tree / "Makefile").write_text("px4_sitl:\n")
    agent_pin = PeerPin(ref=f"eProsima/Micro-XRCE-DDS-Agent@{'a' * 40}", source="a test")
    agent = XrceAgent(pin=agent_pin, instance=2, log_dir=tmp_path / "run")
    px4 = Px4Sitl(
        pin=PeerPin(ref=str(tree), source="a test"), airframe="astro_max", instance=2, log_dir=tmp_path / "run"
    )

    agent.start()
    px4.start()
    agent.stop()

    launches = [r for r in runs if r.get("detach", True)]
    assert (
        launches[0]["command"],
        launches[1]["environment"].get("PX4_UXRCE_DDS_PORT"),
        removed,
    ) == (["MicroXRCEAgent", "udp4", "-p", "8890"], "8890", [launches[0]["name"]])
