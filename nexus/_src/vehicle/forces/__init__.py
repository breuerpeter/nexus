"""The force seam's components: each turns the current state into body wrenches and adds them to
``state.body_f``.

A force element runs once per physics substep, after the command stages and before the physics ``step``
stage, and adds to the shared ``body_f`` buffer, never assigns: the physics ``clear`` stage zeroes it once
per tick, so two elements on one body both act, whatever their order. The shipped one is the propellers',
:class:`~nexus._src.vehicle.forces.propellers.Propellers`, each rotor's thrust, H-force and reaction drag
from its solver-integrated speed. Airframe drag, wind or ground effect would be one more class here.
"""

from .propellers import Propeller, Propellers, propeller_force

__all__ = ["Propeller", "Propellers", "propeller_force"]
