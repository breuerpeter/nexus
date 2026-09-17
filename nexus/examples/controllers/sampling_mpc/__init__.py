"""Sampling + gradient **diffsim Model Predictive Control (MPC)** on the Controller seam: a
receding-horizon controller that samples noisy control trajectories and refines them by
back-propagating a Signed Distance Field (SDF) obstacle cost through a batched differentiable
rollout, ``SolverSemiImplicit``. First-class alongside the Proportional Integral Derivative (PID),
PX4, acados Nonlinear Model Predictive Control (NMPC), and trained-policy controllers.
"""

from .controller import SamplingMPCController

__all__ = ["SamplingMPCController"]
