"""LaunchConfig schema: defaults, parsing from dict/YAML, strictness, and setters."""

import pytest
from pydantic import ValidationError

from nexus._src.config import LaunchConfig


def test_empty_launch_config_defaults():
    lc = LaunchConfig()
    assert lc.vehicle is None
    assert lc.runtime.device == "auto"  # resolved per runtime, CUDA when present; explicit "cpu" = bit-exact
    assert lc.runtime.determinism == "bit-exact"
    assert lc.runtime.max_steps is None
    assert lc.scene is None


def test_from_dict_nested():
    lc = LaunchConfig.from_dict(
        {
            "vehicle": "astro_max_fpv",
            "scene": "seattle-waterfront",
            "runtime": {"device": "cuda:0", "seed": 7},
        }
    )
    assert lc.vehicle == "astro_max_fpv"
    assert lc.scene == "seattle-waterfront"
    assert lc.runtime.device == "cuda:0" and lc.runtime.seed == 7


def test_from_yaml(tmp_path):
    p = tmp_path / "launch.yaml"
    p.write_text("vehicle: astro_max_fpv\n")
    lc = LaunchConfig.from_yaml(p)
    assert lc.vehicle == "astro_max_fpv"


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError):  # extra="forbid" -> ValidationError
        LaunchConfig.from_dict({"vehicel": "astro_max_fpv"})  # typo


def test_setters_chain():
    lc = LaunchConfig().set_vehicle("astro_max_fpv").set_scene("empty")
    assert lc.vehicle == "astro_max_fpv"
    assert lc.scene == "empty"


def test_the_environment_launch_key_is_rejected():
    """The `environment` launch key and the receipt's `environment` field go, since nothing reads them."""
    with pytest.raises(ValidationError):
        LaunchConfig.from_dict({"environment": {"wind": {}}})
