"""The component Protocols of ``core/interfaces.py``: the old actuator contract stays for the examples' ``Rotors``."""

from nexus_sim._src.core import interfaces
from nexus_sim.examples._lib.rotors import Rotors

MARKERS = ("forces_wp", "stages")


def _declares(protocol: type, name: str) -> bool:
    """Whether the Protocol class itself declares ``name``, as a method or an annotated attribute."""
    return name in vars(protocol) or name in getattr(protocol, "__annotations__", {})


def test_the_old_actuator_contract_stays_in_core_for_the_examples_rotors():
    """The old actuator contract, `Actuator` in `core/interfaces.py`, stays for the examples' single-body
    `Rotors`: given the tree, when a caller imports `Actuator` from `nexus_sim._src.core.interfaces`, then it
    declares `forces_wp` and `stages`, `Rotors` satisfies it, and it carries no pairing marker, since the
    guard that read one went with the articulated actuator.
    """
    problems = []
    contract = getattr(interfaces, "Actuator", None)
    if contract is None:
        problems.append("core/interfaces.py declares no Actuator")
    else:
        problems += [f"Actuator declares no {n}" for n in MARKERS if not _declares(contract, n)]
        problems += [f"Rotors has no {n}" for n in MARKERS if not hasattr(Rotors, n)]
        if _declares(contract, "requires_articulated"):
            problems.append("Actuator still declares requires_articulated")
    assert not problems, problems
