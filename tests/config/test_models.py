"""LaunchConfig schema: defaults, parsing from dict/YAML, strictness, and setters."""

import pytest
from pydantic import ValidationError

from nexus._src.config import LaunchConfig


def test_empty_launch_config_defaults():
    lc = LaunchConfig()
    assert lc.vehicle is None
    assert lc.control.kind == "px4-sitl"
    assert lc.runtime.device == "auto"  # resolved per runtime, CUDA when present; explicit "cpu" = bit-exact
    assert lc.runtime.determinism == "bit-exact"
    assert lc.runtime.max_steps is None
    assert lc.scene is None


def test_from_dict_nested():
    lc = LaunchConfig.from_dict(
        {
            "vehicle": "astro_max_fpv",
            "scene": "seattle-waterfront",
            "control": {"kind": "px4-sitl"},
            "runtime": {"device": "cuda:0", "seed": 7},
        }
    )
    assert lc.vehicle == "astro_max_fpv"
    assert lc.scene == "seattle-waterfront"
    assert lc.control.kind == "px4-sitl"
    assert lc.runtime.device == "cuda:0" and lc.runtime.seed == 7


def test_from_yaml(tmp_path):
    p = tmp_path / "launch.yaml"
    p.write_text("vehicle: astro_max_fpv\ncontrol: {kind: px4-sitl}\n")
    lc = LaunchConfig.from_yaml(p)
    assert lc.vehicle == "astro_max_fpv"
    assert lc.control.kind == "px4-sitl"


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError):  # extra="forbid" -> ValidationError
        LaunchConfig.from_dict({"vehicel": "astro_max_fpv"})  # typo


def test_bad_control_kind_rejected():
    with pytest.raises(ValidationError):
        LaunchConfig.from_dict({"control": {"kind": "nonsense"}})


def test_setters_chain():
    lc = LaunchConfig().set_vehicle("astro_max_fpv").set_scene("empty").set_control("px4-sitl")
    assert lc.vehicle == "astro_max_fpv"
    assert lc.scene == "empty"
    assert lc.control.kind == "px4-sitl"


def test_control_kind_rejects_unknown():
    """px4-sitl is the only control kind: the schema accepted px4-hitl but nobody implemented it, see GH #72."""
    with pytest.raises(ValidationError):
        LaunchConfig.from_dict({"control": {"kind": "px4-hitl"}})
    with pytest.raises(ValidationError):
        LaunchConfig().set_control("px4-hitl")


def test_non_px4_control_kinds_rejected():
    # PX4 is the one first-class controller: the old in-process kinds are examples now,
    # self-assembled via Sim.from_orchestrator, and the schema rejects them.
    for kind in ("policy", "builtin", "sampling-mpc", "acados", "external"):
        with pytest.raises(ValidationError):
            LaunchConfig.from_dict({"control": {"kind": kind}})
