"""newton-runtime-isaacsim: the Isaac Sim runtime, a renderer around the one physics.

The alternative to runtime-standalone, which the vehicle Universal Scene Description (USD) selects:
RTX sensor prims route here. It hosts the *same* neutral component set in-process in the Isaac Sim
Kit app, including the *same* ``NewtonPhysics``, nexus's ModelBuilder and solvers, Kit never
simulates, and drives the *same* core ``Orchestrator``. The *only* Isaac-specific seam is the
renderer in :mod:`nexus._src.rendering` and the RTX sensors under
:mod:`nexus._src.vehicle.sensors`: a Kit stage composed purely for RTX, posed from Newton
Air's model each rendered frame. The controller still serves the PX4 Hardware In The Loop (HIL) link on
TCP:4560; PX4 lockstep still paces the loop.

``run_isaacsim`` is the entrypoint: it boots ``SimulationApp``, then assembles and runs. The
top-level imports stay light, no ``isaacsim``/``omni``, so this package imports fine outside
the Kit env; the heavy bits load lazily inside ``run_isaacsim`` once the app has booted.
"""

from .runtime import run_isaacsim

__all__ = ["run_isaacsim"]
