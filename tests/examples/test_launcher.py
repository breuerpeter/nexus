"""The `python -m nexus.examples <name>` launcher, which mirrors NVIDIA Newton's."""

from __future__ import annotations

import importlib.util

import pytest

from nexus import examples


def test_get_examples_names():
    assert set(examples.get_examples()) == {
        "pid",
        "sampling_mpc",
        "acados_nmpc",
        "px4_sitl",
        "px4_sitl_manual",
        "px4_mission",
        "goto_policy",
        "gain_tuning",
        "mass_recovery",
    }


@pytest.mark.parametrize("module", examples.get_examples().values())
def test_every_example_module_resolves(module):
    # The dotted path must import, since dirs are namespace packages after the underscore rename, and the
    # file must exist in the package: this is what `runpy.run_module` relies on at launch.
    assert importlib.util.find_spec(module) is not None


def test_list_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["nexus.examples", "--list"])
    with pytest.raises(SystemExit) as exc:
        examples.main()
    assert exc.value.code == 0
    assert "sampling_mpc" in capsys.readouterr().out


def test_unknown_example_exits_one(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nexus.examples", "nope"])
    with pytest.raises(SystemExit) as exc:
        examples.main()
    assert exc.value.code == 1


def test_known_example_dispatches_with_passthrough_args(monkeypatch):
    calls = {}

    def fake_run_module(target, run_name, alter_sys=False):
        import sys

        calls["target"] = target
        calls["run_name"] = run_name
        calls["argv"] = list(sys.argv)

    monkeypatch.setattr("runpy.run_module", fake_run_module)
    monkeypatch.setattr("sys.argv", ["nexus.examples", "sampling_mpc", "--record"])
    examples.main()

    assert calls["target"] == "nexus.examples.controllers.sampling_mpc.obstacle_slalom"
    assert calls["run_name"] == "__main__"
    # The example gets itself as argv[0] with only its own args after the name.
    assert calls["argv"] == [calls["target"], "--record"]
