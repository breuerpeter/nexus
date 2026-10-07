"""The sensors, the rotor chain and the controllers a vehicle carries on its base body.

One tier per seam, each an ordinary package: ``sensors`` measure the vehicle's state, ``commands`` turn
the controller's command into Newton's control inputs, ``forces`` add body wrenches from the state, and
``controllers`` command the vehicle from on board. Newton's own actuators, the rotor motors the vehicle's
Universal Scene Description (USD) file authors, sit between the command elements and the force elements, and
physics steps them. ``rotors`` holds the rotor geometry those tiers share. The guidance isn't here: the run
owns it, not the vehicle.
"""
