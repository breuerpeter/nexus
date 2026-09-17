"""The public vocabulary of ``core/schema.py``: ``Controls`` names the seam it crosses, not a vehicle type."""

import dataclasses

from nexus import Controls


def test_controls_carries_one_per_actuator_command_vector():
    """`Controls` carries one per-actuator command vector, named for what the controller emits: given
    `from nexus import Controls`, when a caller lists its dataclass fields, then there is one, not
    `motors`, and the class docstring says actuator and never rotor or motor.
    """
    names = [f.name for f in dataclasses.fields(Controls)]
    doc = (Controls.__doc__ or "").lower()
    problems = []
    if len(names) != 1:
        problems.append(f"{len(names)} fields, not one: {names}")
    if "motors" in names:
        problems.append("the field is still named motors")
    if "actuator" not in doc:
        problems.append("the docstring does not say actuator")
    for word in ("rotor", "motor"):
        if word in doc:
            problems.append(f"the docstring says {word}")
    assert not problems, problems
