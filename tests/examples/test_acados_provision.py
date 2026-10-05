"""How the acados example provisions acados: the pinned commit, built in the folder the example owns."""

from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path

from nexus.examples.controllers.acados_nmpc import provision

PINNED = "f22001ac39773eb9988b9f04b4d38378818d10d6"


def _fake_machine(monkeypatch) -> list[list[str]]:
    """Stand in for git, CMake and the network at the process boundary, and record each command."""
    ran: list[list[str]] = []

    def run(cmd, **kwargs):
        ran.append(cmd)
        if cmd[:2] == ["cmake", "--build"]:  # the install step leaves the library in the tree
            lib = Path(cmd[2]).parent / "lib"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "libacados.so").touch()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: Path(dest).write_bytes(b"renderer"))
    return ran


def test_the_default_acados_folder_is_named_for_the_pinned_commit(monkeypatch, tmp_path):
    """With no `ACADOS_SOURCE_DIR`, the example owns one folder per pinned commit under the home cache."""
    monkeypatch.delenv("ACADOS_SOURCE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert provision.acados_dir() == tmp_path / ".cache" / "nexus" / "acados" / PINNED


def test_provisioning_an_empty_folder_fetches_the_pinned_commit(monkeypatch, tmp_path):
    """Given an empty `ACADOS_SOURCE_DIR`, provisioning fetches the commit `acados.ref` pins."""
    monkeypatch.setenv("ACADOS_SOURCE_DIR", str(tmp_path))
    ran = _fake_machine(monkeypatch)

    provision.provision()

    fetch = ["git", "-C", str(tmp_path), "fetch", "-q", "--depth", "1", "https://github.com/acados/acados.git", PINNED]
    assert fetch in ran


def test_provisioning_leaves_an_executable_renderer_in_the_tree(monkeypatch, tmp_path):
    """Given an empty `ACADOS_SOURCE_DIR`, provisioning leaves the Tera renderer executable in `bin/`."""
    monkeypatch.setenv("ACADOS_SOURCE_DIR", str(tmp_path))
    _fake_machine(monkeypatch)

    provision.provision()

    assert os.access(tmp_path / "bin" / "t_renderer", os.X_OK)


def test_provisioning_a_built_tree_runs_no_command(monkeypatch, tmp_path):
    """Provisioning is idempotent: given a tree that holds the library and the renderer, it runs nothing."""
    monkeypatch.setenv("ACADOS_SOURCE_DIR", str(tmp_path))
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libacados.so").touch()
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "t_renderer").touch(mode=0o755)
    ran = _fake_machine(monkeypatch)

    provision.provision()

    assert ran == []
