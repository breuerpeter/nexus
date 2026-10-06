"""LaunchConfig schema: defaults, parsing from dict/YAML, strictness, and setters."""

import pytest
from pydantic import ValidationError

from nexus_sim._src.config import LaunchConfig


def test_empty_launch_config_defaults():
    lc = LaunchConfig()
    assert lc.vehicle is None
    assert lc.runtime.device == "auto"  # resolved per runtime, CUDA when present; explicit "cpu" = bit-exact
    assert lc.runtime.max_steps is None
    assert lc.scene is None


def test_from_dict_nested():
    lc = LaunchConfig.from_dict(
        {
            "vehicle": "astro_max_fpv",
            "scene": "seattle-waterfront",
            "runtime": {"device": "cuda", "seed": 7},
        }
    )
    assert lc.vehicle == "astro_max_fpv"
    assert lc.scene == "seattle-waterfront"
    assert lc.runtime.device == "cuda" and lc.runtime.seed == 7


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


@pytest.mark.parametrize(
    ("field", "launch"),
    [
        ("substeps", {"runtime": {"substeps": 4}}),
        ("determinism", {"runtime": {"determinism": "bit-exact"}}),
        ("sensors", {"sensors": {"imu": {"rate": 250}}}),
        ("ulog", {"output": {"ulog": True}}),
        ("video", {"output": {"video": False}}),
        ("run_id", {"output": {"run_id": "r1"}}),
    ],
)
def test_a_launch_that_names_a_field_no_run_applies_fails_to_load(field, launch):
    """A launch that names `runtime.substeps`, `runtime.determinism`, `sensors`, `output.ulog`,
    `output.video` or `output.run_id` fails to load, and the error names the field.

    Given a launch dict that names one of the six fields, when `LaunchConfig.from_dict` loads it, then
    it raises a validation error whose text names that field.
    """
    with pytest.raises(ValidationError, match=field):
        LaunchConfig.from_dict(launch)


def test_a_launch_that_names_a_device_ordinal_fails_to_load():
    """A launch that names a `runtime.device` other than `auto`, `cpu` or `cuda` fails to load, and the
    error names the three values.

    Given a launch dict with `runtime.device: cuda:1`, when `LaunchConfig.from_dict` loads it, then it
    raises a validation error whose text names `auto`, `cpu` and `cuda`.
    """
    with pytest.raises(ValidationError, match="'auto', 'cpu' or 'cuda'"):
        LaunchConfig.from_dict({"runtime": {"device": "cuda:1"}})
