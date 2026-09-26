"""The goto_policy example: its policy resolution, the hosted policy from the catalog's base or the
``--policy`` override, with no GPU and no network since the base is a ``file://`` directory; and
its flight, which repeats bit for bit on the Newton CPU backend.
"""

import hashlib
import re

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("newton")

import nexus as na
from nexus._src.build.assembly import build_scenario
from nexus._src.build.launch import resolve_to_vehicle_builder
from nexus._src.config import LaunchConfig
from nexus.examples.controllers.policy.assembly import build_policy_orchestrator
from nexus.examples.controllers.policy.goto import flight


def _catalog_with_base(tmp_path, monkeypatch):
    """A project catalog in the working directory whose ``assets.base`` is a ``file://`` directory,
    and an asset cache under ``tmp_path``. Returns the base directory.
    """
    base = tmp_path / "blobs"
    base.mkdir()
    (tmp_path / "nexus.registry.yaml").write_text(f"assets:\n  base: {base.as_uri()}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))
    return base


def test_hosted_policy_is_fetched_from_the_catalog_base(tmp_path, monkeypatch):
    """The example fetches the hosted policy from the catalog's `assets.base` at
    `<base>/assets/policies/<name>-<sha>.pt`.
    """
    data = b"a torchscript policy"
    sha = hashlib.sha256(data).hexdigest()
    base = _catalog_with_base(tmp_path, monkeypatch)
    blob = base / "assets" / "policies" / f"goto_policy-{sha}.pt"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)

    out = flight._resolve_policy(None, asset={"name": "goto_policy", "sha256": sha})

    got = hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert (str(tmp_path / "cache") in out, got) == (True, sha)


def test_unreachable_hosted_policy_exits_with_the_policy_hint(tmp_path, monkeypatch):
    """A hosted policy the base can't serve exits with the `--policy` hint, not a traceback."""
    _catalog_with_base(tmp_path, monkeypatch)  # an empty base: nothing to fetch

    with pytest.raises(SystemExit) as exc:
        flight._resolve_policy(None)

    message = str(exc.value)
    assert ("--policy" in message, "train.py" in message) == (True, True)


def test_policy_override_flies_that_file_and_skips_the_fetch(tmp_path, monkeypatch):
    """`--policy <file>` flies that file and skips the fetch."""
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"a fresh local export")
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))

    out = flight._resolve_policy(str(policy))

    assert (out, (tmp_path / "cache").exists()) == (str(policy), False)


def test_policy_override_that_is_not_a_file_exits_naming_it(tmp_path):
    """`--policy <not-a-file>` exits naming the path."""
    missing = tmp_path / "nope.pt"

    with pytest.raises(SystemExit, match=re.escape("nope.pt")):
        flight._resolve_policy(str(missing))


def _fly_tour(policy: str, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Fly the example's tour with *policy* for *steps* control steps on the CPU backend, as the
    example does, and return the recorded positions and ``xyzw`` quaternions of the base body.
    """
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base"))
    orch = build_policy_orchestrator(cfg, policy_path=policy, vehicle_builder=vb, max_steps=steps)
    with na.Sim.from_orchestrator(orch) as sim:
        sim.operator.set_mission(flight.WAYPOINTS)
        sim.run()
    traj = sim.physics[sim.base_body].history()
    return np.array([s.position for s in traj]), np.array([s.quat_xyzw for s in traj])


@pytest.mark.usefixtures("warp_cpu")  # the build's force_cpu sets the device; the scope puts it back
def test_a_policy_flies_the_same_path_every_time(torchscript_policy):
    """A policy flies the same path every time: given one exported policy, when the tour flies twice,
    then the recorded `est_pos` and `est_quat_xyzw` of the two flights are equal element for element.
    """
    policy = torchscript_policy()

    pos_1, quat_1 = _fly_tour(policy, steps=200)
    pos_2, quat_2 = _fly_tour(policy, steps=200)

    assert (len(pos_1) > 0, np.array_equal(pos_1, pos_2), np.array_equal(quat_1, quat_2)) == (True, True, True)
