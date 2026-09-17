"""Where the Kit launch finds its checkout.

The launch reads the compose file out of ``docker/`` and bind-mounts the repo as ``NEXUS_DIR``;
neither is a property of the process cwd. Deriving both from the module's own path is what makes
``nexus run`` work from a subdirectory with nothing exported.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nexus._src.cli import detect

ROOT = Path(__file__).resolve().parents[2]


def test_the_checkout_is_derived_from_this_module_not_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path(detect.repo_root())
    assert root == ROOT
    assert (root / "nexus").is_dir()
    assert (root / "docker" / "docker-compose.yml").is_file(), "the launch reads the compose file from here"


# Mount sources that must ALREADY exist: the two checkouts and the daemon socket. A cache mount is
# the opposite case -- docker creating it on first use is what should happen.
MUST_EXIST = ("NEXUS_DIR", "PX4_DIR", "docker.sock")


def test_mounts_that_must_already_exist_refuse_to_create_themselves():
    """Docker CREATES a missing bind-mount source, root-owned (the reasoning is in
    docker-compose.yml). A missing SOCKET is worse: the path becomes a root-owned directory where
    the daemon's socket belongs.
    """
    svc = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text())["services"]["isaacsim"]
    guarded = []
    for v in svc["volumes"]:
        source = v["source"] if isinstance(v, dict) else v.split(":")[0]
        if not any(name in source for name in MUST_EXIST):
            continue
        assert isinstance(v, dict), f"{source} needs the long syntax to set create_host_path"
        assert v["bind"]["create_host_path"] is False, source
        guarded.append(source)
    assert any("NEXUS_DIR" in s for s in guarded), "the workspace mount is this service's reason to exist"


def test_auto_runtime_resolves_against_the_registry_the_flag_names(tmp_path, monkeypatch):
    """An explicit registry beats discovery on the command line too: `--runtime auto` probes the
    vehicle Universal Scene Description (USD) file to pick a runtime, and it must read the catalog
    `--registry` names rather than the one it would otherwise find.
    """
    pytest.importorskip("pxr")
    usd = tmp_path / "plain.usda"  # a vehicle with no RTX sensor prims: the standalone runtime
    usd.write_text('#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\ndef Xform "vehicle" {}\n')
    registry = tmp_path / "mine.yaml"
    registry.write_text(
        "vehicles:\n"
        "  - name: my_quad\n"
        f'    usd: {{ url: "{usd.as_uri()}", sha256: {hashlib.sha256(usd.read_bytes()).hexdigest()} }}\n'
        "scenes:\n  empty: {}\n"
        "defaults: { vehicle: my_quad, scene: empty }\n"
    )
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))  # no writes into the checkout's cache
    args = SimpleNamespace(runtime="auto", vehicle="my_quad", registry=str(registry))

    assert detect.resolve_runtime(args) == "standalone"
