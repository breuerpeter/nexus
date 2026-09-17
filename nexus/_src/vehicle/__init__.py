"""The sensors, the actuators and the controllers a vehicle carries on its base body.

One tier per role, each an ordinary package: ``sensors`` measure the vehicle's state, ``actuators``
move it, ``controllers`` command it from on board. The operator is not here: it commands the vehicle
from outside, a ground station on its own link, so it stays a peer of this package.
"""
