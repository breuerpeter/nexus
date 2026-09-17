"""The actuator↔model pairing guard.

The actuator contract is :class:`~nexus._src.core.interfaces.Actuator`, declared beside its consumer, the
orchestrator. What stays here is the guard the assemblies call before wiring one: the shipped
:class:`~nexus._src.vehicle.actuators.articulated.ArticulatedRotors` drives real joints, so it declares
``requires_articulated = True`` and needs motors on ``model.actuators``, which the vehicle's Universal Scene
Description (USD) authors as ``NewtonActuator`` prims.
"""

from __future__ import annotations


def check_actuator_model_pairing(actuator, model) -> None:
    """Raise if the actuator needs an articulated model, motors on ``model.actuators``, but the loaded
    model has none. The shipped :class:`~nexus._src.vehicle.actuators.articulated.ArticulatedRotors`
    declares ``requires_articulated = True``, so an unauthored vehicle USD fails here, loudly, before the
    run starts; an actuator that sums its wrench onto one body declares ``False`` and pairs with any model.
    """
    if getattr(actuator, "requires_articulated", False) and not getattr(model, "actuators", None):
        raise ValueError(
            f"{type(actuator).__name__} requires an articulated model with motors on "
            "model.actuators; the loaded model has none."
        )
