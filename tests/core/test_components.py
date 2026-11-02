"""A component reads only its own schema's attributes, #381: what it needs from another component arrives as a
signal. The one class that still reads another schema's attributes through ``siblings`` is the shipped ESC, named
in ``SIBLINGS_EXEMPT`` until #386 The ESC emits a terminal voltage and the motor is Newton's DC motor empties it.
"""

import importlib


def _class(path: str) -> type:
    """The class at `path`, `module.Class`."""
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def test_the_siblings_exemption_names_the_shipped_esc_alone_and_each_class_it_names_declares_siblings():
    """The exemption is one named list in the core that holds the shipped ESC alone, so #386 empties it.

    Given the core's exemption list, when a test reads it, then it names the shipped ESC's class alone, and each
    class it names declares `siblings`.
    """
    from nexus_uav._src.core.components import SIBLINGS_EXEMPT

    declares = [bool(getattr(_class(path), "siblings", None)) for path in SIBLINGS_EXEMPT]
    assert (tuple(SIBLINGS_EXEMPT), declares) == (("nexus_uav._src.vehicle.commands.esc.Esc",), [True])
