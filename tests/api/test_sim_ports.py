"""The run's port map, ``Sim.ports``: where a script reads the address of a link that leaves the run.

The run owns every address. The builder numbers PX4's links from the Software In The Loop (SITL)
instance the run claims, builds the Hardware In The Loop (HIL) link's end inside the run, and names
the offboard link, whose end a script opens outside the run, in the map: its port, over the
User Datagram Protocol (UDP), and the MAVLink system id the client addresses. The build's assembly
step returns the loop over stand-in core components, as ``tests/peers/px4_sitl/test_px4_peer.py``
does, so the tests need no physics, no PX4 tree and no container: the map is a fact of the build,
read through ``Sim``.
"""

import hashlib

import pytest

import nexus_sim._src.build.launch as launch_mod
from nexus_sim._src.api.sim import Sim
from nexus_sim._src.config import Catalog, LaunchConfig
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import SimTime
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake

# --- the stand-in core components, as tests/core builds the loop -----------------------------------


class _Clock:
    dt = 0.004

    def __init__(self):
        self._t = SimTime()

    def now(self):
        return self._t

    def advance(self):
        self._t = SimTime(self._t.sim_time + self.dt, self._t.step_index + 1)
        return self._t

    def throttle(self):
        pass


class _Physics:
    base_body = "body_frd"

    def reset(self):
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        return state

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]


class _Actuator:
    def forces(self, controls, state):
        pass

    def stages(self):
        return [Stage("forces", "device", lambda tick: self.forces(None, tick.state))]


@pytest.fixture
def assembly(monkeypatch, tmp_path):
    """The build's assembly step returns the loop over the stand-in components, around the controller
    and the peers the build made; the home folder sits in the test's folder, so an instance lock or a
    console log lands there.
    """

    def assemble(label, cfg, **kw):
        run = {
            k: kw[k] for k in ("peers", "ports") if k in kw
        }  # what the build made, handed through as the real assembly does
        return Orchestrator(
            clock=_Clock(),
            physics=_Physics(),
            actuator=_Actuator(),
            sensors=[],
            controller=kw["controller"],
            exchange_timeout=0.5,
            preroll_timeout=kw.get("preroll_timeout", 2.0),
            **run,
        )

    monkeypatch.setattr(launch_mod, "build_orchestrator", assemble)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


# --- a catalog vehicle that declares PX4 and its SITL peer ----------------------------------------

_DECLARED = (
    '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
    'def Xform "vehicle"\n{\n'
    '    def Scope "Controller" (\n        prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]\n    )\n'
    '    {\n        string nexus:airframe = "astro_max"\n    }\n}\n'
)

# A layer that drops the PX4 SITL peer's declaration from the controller's scope, so the run attaches to an
# autopilot started elsewhere.
_DROP_PX4_SITL = (
    '#usda 1.0\n\nover "vehicle"\n{\n'
    '    over "Controller" (\n        delete apiSchemas = ["NexusPx4SitlAPI"]\n    )\n    {\n    }\n}\n'
)


class _Px4OnInstance2:
    """A PX4 SITL peer that claims instance 2 and starts no process: the real peer's shape on the peer
    contract, which the port map follows, with no container behind it.
    """

    @staticmethod
    def claim_instance():
        return 2, None

    def __init__(self, **run):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def alive(self):
        return True


def _run(tmp_path, *, layer: str | None = None, peers: dict | None = None) -> Orchestrator:
    """Build a run of a catalog vehicle that declares PX4 and its SITL peer, over `layer` when the caller
    passes one, with the PX4 SITL peer sent to the class `peers` names.
    """
    blob = tmp_path / "vehicle.usda"
    blob.write_text(_DECLARED)
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    catalog = Catalog.from_dict(
        {
            "vehicles": {"astro": {"usd": {"url": blob.as_uri(), "sha256": sha, "filename": blob.name}}},
            "scenes": {"empty": {}},
        }
    )
    spec = {"vehicle": "astro", "scene": "empty"}
    if layer is not None:
        path = tmp_path / "override.usda"
        path.write_text(layer)
        spec["layer"] = str(path)
    launch = LaunchConfig.from_dict(spec)
    return launch_mod.build_from_launch(
        launch, catalog=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0, peers=peers
    )


def test_the_runs_port_map_names_the_offboard_link_from_the_px4_instance_the_run_claims(assembly, tmp_path):
    """The run's port map names the offboard link from the PX4 instance the run claims.

    Given a `Sim` over a PX4 vehicle whose run claims instance 2, when a script reads the run's port
    map, then the offboard link's entry names UDP port 14542 and MAVLink system id 3.
    """
    with Sim.from_orchestrator(_run(tmp_path, peers={"px4_sitl": _Px4OnInstance2})) as sim:
        link = sim.ports["offboard"]

    assert (link["protocol"], link["port"], link["system_id"]) == ("udp", 14542, 3)


def test_a_run_attached_to_a_px4_started_elsewhere_lists_instance_0s_offboard_link(assembly, tmp_path):
    """A run attached to a PX4 started elsewhere lists instance 0's offboard link.

    Given a `Sim` over a PX4 vehicle whose override layer drops the peer declaration, when a script
    reads the run's port map, then the offboard link's entry names UDP port 14540 and system id 1.
    """
    with Sim.from_orchestrator(_run(tmp_path, layer=_DROP_PX4_SITL)) as sim:
        link = sim.ports["offboard"]

    assert (link["protocol"], link["port"], link["system_id"]) == ("udp", 14540, 1)


@pytest.mark.xdist_group("px4_ports")  # binds or dials PX4's ports, with the other tests that do
def test_on_a_fake_px4_run_the_offboard_lookup_fails_at_once_and_names_the_fake(assembly, tmp_path):
    """On a fake PX4 run the offboard lookup fails at once and names the fake.

    Given a `Sim` over the fake PX4, when a script reads the offboard link from the run's port map,
    then it raises before any step, with a message that names the fake PX4.
    """
    with Sim.from_orchestrator(_run(tmp_path, peers={"px4_sitl": Px4Fake})) as sim:
        with pytest.raises(LookupError, match="fake PX4"):
            sim.ports["offboard"]
