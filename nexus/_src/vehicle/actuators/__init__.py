"""Actuator component: the core actuation seam.

The core actuator is :class:`~nexus._src.vehicle.actuators.articulated.ArticulatedRotors`: rotor
motors as ``newton.actuators`` authored in Universal Scene Description (USD). Each is a
``NewtonActuator`` prim, a ``ControllerPID`` velocity servo plus the ``ClampingDCMotor`` envelope,
driving the real rotor joints, with the airflow-aware propeller aero of
:mod:`~nexus._src.vehicle.actuators.propeller` as the ``body_f`` kernel. Ω is a solver-integrated joint
state: physical motor lag plus saturation, spinning props at no extra cost. ``Controls.command`` is
always ``nr`` normalized per-rotor commands.

The joint-agnostic single-body actuator, ``Rotors`` or ``RigidBodyRotors`` with a summed base wrench
and first-order Ω lag, lives with the examples that collapse the model to a single body, in
``nexus/examples/_lib/rotors.py``. Core keeps only the shared geometry and propeller pieces,
:mod:`layout` and :mod:`propeller`, and the pairing guard.
"""

from .articulated import ArticulatedRotors, aero_from_omega
from .base import check_actuator_model_pairing
from .layout import RPM_PER_RADS, find_rotor_joints, quat_to_R
from .propeller import propeller_force

__all__ = [
    "RPM_PER_RADS",
    "ArticulatedRotors",
    "aero_from_omega",
    "check_actuator_model_pairing",
    "find_rotor_joints",
    "propeller_force",
    "quat_to_R",
]
