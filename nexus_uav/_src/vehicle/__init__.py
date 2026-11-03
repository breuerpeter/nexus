"""The sensors, the rotor chain and the controllers a vehicle carries on its base body.

One tier per role, each an ordinary package: ``sensors`` measure the vehicle's state, ``commands`` turn
the controller's command into the signal the chain takes on, ``actuators`` turn that signal into torque on a
joint, ``forces`` add body wrenches from the state, and ``controllers`` command the vehicle from on board.
``rotors`` holds the rotor geometry those tiers share. The guidance isn't here: the run
owns it, not the vehicle.
"""
