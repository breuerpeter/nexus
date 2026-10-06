"""Analytic real-time Nonlinear Model Predictive Control (NMPC) via **acados**, the Controller seam: the
formulation: a 13-state quad, a differential-flatness reference, and Sequential Quadratic
Programming (SQP) with Real Time Iteration (RTI). First-class alongside the Proportional Integral
Derivative (PID), PX4, sampling Model Predictive Control (MPC), and trained-policy controllers.
acados/casadi/ruckig are optional, code-generated dependencies, lazy-imported inside the controller's
methods.
"""

from .controller import AcadosNMPCController

__all__ = ["AcadosNMPCController"]
