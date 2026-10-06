"""The command seam's components: each turns the controller's command into Newton's control inputs.

A command stage runs once per physics substep, after the physics ``clear`` stage and before the force
stages, and writes the targets and feedforward the model's ``newton.Control`` holds; physics then steps
every Newton actuator the vehicle declares before its solver. The shipped one is the rotors',
:class:`~nexus_sim._src.vehicle.commands.rotors.RotorCommand`: the controller's normalized per-rotor commands
to each rotor motor's speed target and drag feedforward. A component that writes another actuator's
target, a gimbal servo's or an Electronic Speed Controller (ESC) input stage, is one more class here.
"""

from .rotors import RotorCommand

__all__ = ["RotorCommand"]
