"""The component Protocols of ``core/interfaces.py``: one actuator contract, declared where its consumer is."""

import typing

from nexus._src.core import interfaces
from nexus._src.vehicle.actuators import ArticulatedRotors, base

MARKERS = ("forces", "requires_articulated", "capturable")


def _declares(protocol: type, name: str) -> bool:
    """Whether the Protocol class itself declares ``name``, as a method or an annotated attribute."""
    return name in vars(protocol) or name in getattr(protocol, "__annotations__", {})


def test_one_actuator_contract_in_core_carrying_the_markers():
    """One actuator contract, `Actuator` in `core/interfaces.py`, carrying `requires_articulated` and
    `capturable`; the pairing guard alone stays in `vehicle/actuators/base.py`: given the tree, when
    a caller imports `Actuator` from `nexus._src.core.interfaces`, then it declares `forces`,
    `requires_articulated` and `capturable`, the shipped `ArticulatedRotors` satisfies it, no
    `ActuatorModel` exists, and `vehicle/actuators/base.py` declares no Protocol and keeps
    `check_actuator_model_pairing`.
    """
    problems = []
    contract = getattr(interfaces, "Actuator", None)
    if contract is None:
        problems.append("core/interfaces.py declares no Actuator")
    else:
        problems += [f"Actuator declares no {n}" for n in MARKERS if not _declares(contract, n)]
        problems += [f"ArticulatedRotors has no {n}" for n in MARKERS if not hasattr(ArticulatedRotors, n)]
    if hasattr(interfaces, "ActuatorModel"):
        problems.append("core/interfaces.py still declares ActuatorModel")
    protocols = [
        n
        for n, m in vars(base).items()
        if isinstance(m, type) and typing.Protocol in m.__mro__ and m.__module__ == base.__name__
    ]
    if protocols:
        problems.append(f"vehicle/actuators/base.py still declares {protocols}")
    if not callable(getattr(base, "check_actuator_model_pairing", None)):
        problems.append("vehicle/actuators/base.py lost check_actuator_model_pairing")
    assert not problems, problems
