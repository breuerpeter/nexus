"""A component reads only its own schema's attributes, #381: what it needs from another component arrives as a
signal. The exemption list in the core, ``SIBLINGS_EXEMPT``, names the classes that still read another schema's
attributes through ``siblings``, and #386 The ESC emits a terminal voltage and the motor is Newton's DC motor
empties it.
"""

from importlib.metadata import entry_points


def _shipped() -> list[type]:
    """Every class the shipped package registers in the `nexus.registry` group."""
    return [ep.load() for ep in entry_points(group="nexus.registry") if ep.value.startswith("nexus_uav.")]


def test_the_siblings_exemption_is_empty_and_no_shipped_class_declares_siblings():
    """The ESC reads nothing of its propeller's or its motor's, and no shipped class reads another schema's
    attributes: the exemption list is empty.

    Given the core's exemption list and the registry's shipped classes, when a test reads them, then the list is
    empty and no class declares `siblings`.
    """
    from nexus_uav._src.core.components import SIBLINGS_EXEMPT

    declares = [cls.__name__ for cls in _shipped() if getattr(cls, "siblings", None)]
    assert (tuple(SIBLINGS_EXEMPT), declares) == ((), [])
