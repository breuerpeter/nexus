"""MinSnapReference: a min-snap polynomial trajectory planner for the operator, Plane 5.

This is the higher-fidelity sibling of the ruckig :class:`~nexus.examples._lib.reference.FlatnessReference`.
Where ruckig chains per-axis, jerk-limited legs, so the velocity *direction* swings sharply at each
waypoint the path flies through, a min-snap polynomial minimizes ∫‖d⁴p/dt⁴‖² over the whole path
subject to the waypoint constraints: it rounds the corners, so the velocity direction turns gently.
That smoothness is what makes **nose-first** flight track cleanly: a
dedicated smooth yaw polynomial gives a yaw reference *and* an ω_z feed-forward, ``state.w.z()=yaw'``,
both bounded by construction, so the Nonlinear Model Predictive Control (NMPC) turns the nose to follow
the path without a spiky rate reference fighting position tracking.

Same public interface as ``FlatnessReference``: ``set_start``, ``duration``, ``reference_path``, and
``flat_state_at``, so the operator + the acados NMPC use either planner interchangeably. Pure host
numpy, with no ruckig, casadi or acados; the operator stays solver-agnostic.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from nexus.examples._lib.reference import GRAVITY, _quat_from_z_yaw


class _Polynomial:
    """A single scalar polynomial ``p(t)=Σ cᵢ·τⁱ`` (τ = (t−t₀)/T) whose free coefficients give the least
    weighted-derivative cost ``cᵀHc`` subject to equality constraints ``Ac=b``.
    ``weights[k]`` weights the derivative of order ``k+1``, so ``(0,0,0,1)`` is min-snap,
    a weight on the fourth derivative; the constraints pin the position + selected derivatives at given
    times. Solved as the Karush-Kuhn-Tucker (KKT) system ``[[2H, Aᵀ],[A, 0]]·[c; λ] = [0; b]``.
    """

    def __init__(self, order: int, weights, t_offset: float, duration: float):
        self.order = int(order)
        n = self.order + 1
        self.t_offset = float(t_offset)
        self.t_scale = 1.0 / float(duration)
        self.weights = np.asarray(weights, dtype=np.float64)
        i = np.arange(n)
        self._exp = np.maximum(i[:, None] - i[None, :], 0)  # exponents[i,d] = max(i−d, 0)
        self._alpha = np.zeros((n, n))  # alpha[i,d] = i!/(i−d)!  (0 for i<d)
        for d in range(n):
            for ii in range(d, n):
                self._alpha[ii, d] = factorial(ii) / factorial(ii - d)
        self._rows: list[np.ndarray] = []
        self._b: list[float] = []
        self._c = np.full(n, np.nan)

    def _tau(self, t: float) -> float:
        return self.t_scale * (t - self.t_offset)

    def _basis(self, t: float, d: int) -> np.ndarray:
        """Row of the linear map ``c → pᵈ(t)``, the time-domain ``d``-th derivative: ``t_scaleᵈ·αᵢ,d·τ^eᵢ,d``."""
        tau = self._tau(t)
        return (self.t_scale**d) * self._alpha[:, d] * np.power(tau, self._exp[:, d])

    def add_constraint(self, t: float, d: int, value: float) -> None:
        """Pin the ``d``-th time-derivative at time ``t`` to ``value``; ``d=0`` pins the position."""
        self._rows.append(self._basis(t, d))
        self._b.append(float(value))

    def _createH(self) -> np.ndarray:
        n = self.order + 1
        H = np.zeros((n, n))
        for k, w in enumerate(self.weights):
            if w <= 0.0:
                continue
            d = k + 1  # weights[k] weights the derivative of order k+1
            a = self._alpha[:, d]
            e = self._exp[:, d]
            denom = np.maximum(e[:, None] + e[None, :] + 1.0, 1.0)
            H += (self.t_scale ** (2 * d)) * w * (a[:, None] * a[None, :]) / denom
        return H

    def solve(self) -> None:
        A = np.array(self._rows)
        b = np.array(self._b)
        H = self._createH()
        n, m = H.shape[0], A.shape[0]
        S = np.block([[2.0 * H, A.T], [A, np.zeros((m, m))]])
        s = np.concatenate([np.zeros(n), b])
        try:
            x = np.linalg.solve(S, s)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(S, s, rcond=None)[0]
        self._c = x[:n]

    def __call__(self, t: float, d: int = 0) -> float:
        return float(self._basis(t, d) @ self._c)


class MinSnapReference:
    """The polynomial reference: a per-axis min-snap position polynomial through the waypoints
    + a smooth, nose-first, min-jerk yaw polynomial, lifted to the full flat quad state via differential
    flatness. The operator builds it and feeds it to the NMPC via ``accept_setpoint``, wrapped in a
    ``ReferenceTrajectory``.
    """

    def __init__(
        self, waypoints, *, mass, cruise_speed: float = 1.4, order: int | None = None, forward_heading: bool = True,
    ):  # fmt: skip
        """Configure the planner; the polynomial fit happens lazily on first use, after :meth:`set_start`.

        Args:
            waypoints: Ordered positions the path flies through, a sequence of x, y, z triples, excluding
                the start.
            mass: Vehicle mass [kg]; scales the specific thrust into the collective-thrust feedforward.
            cruise_speed: Average path speed [m/s] used to assign the per-waypoint times, ∝ segment length,
                which sets the total trajectory duration, comparable to the ruckig plan's.
            order: Polynomial order per axis. Defaults to ``n_waypoints + 8``, at least 11, so the min-snap
                cost always keeps 4 free Degrees of Freedom (DOF) beyond the ``n_waypoints + 1 + 4``
                equality constraints: order 11 on a 3-waypoint path, generalized. At the exact
                constraint count the fit degenerates to plain interpolation; below it, the KKT system has
                more equations than unknowns.
            forward_heading: Fly **nose-first**: the yaw polynomial faces the direction of travel, the
                chord tangent, at each waypoint. ``False`` holds a fixed 0 heading.
        """
        self.waypoints = np.array([[float(x) for x in wp_] for wp_ in waypoints], dtype=np.float64)
        self.mass = float(mass)
        self.cruise_speed = float(cruise_speed)
        self.order = int(order) if order is not None else max(11, len(self.waypoints) + 8)
        self.forward_heading = bool(forward_heading)
        self._p0 = None
        self._polys = None  # (px, py, pz)
        self._total_dur = 0.0

    def set_start(self, p0) -> None:
        """Anchor the trajectory at the start position, the vehicle's pose on the first tick."""
        self._p0 = np.asarray(p0, dtype=np.float64)

    @property
    def duration(self) -> float:
        """Total trajectory duration [s]; fits the plan on first access."""
        if self._polys is None:
            self._build()
        return self._total_dur

    def _waypoint_times(self, pts: np.ndarray) -> np.ndarray:
        """Per-waypoint times ∝ cumulative segment length / cruise speed, so the vehicle flies the whole
        path at a roughly constant speed, and the total duration ≈ the ruckig plan's.
        """
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg) / max(self.cruise_speed, 1e-6)])
        return t

    def _build(self) -> None:
        pts = np.vstack([self._p0, self.waypoints])  # start, wp1, …, goal
        times = self._waypoint_times(pts)
        self._total_dur = float(times[-1])
        T = self._total_dur

        # Per-axis min-snap position polynomial: position at every waypoint + rest, v=a=0, at both ends.
        self._polys = []
        for ax in range(3):
            poly = _Polynomial(self.order, (0.0, 0.0, 0.0, 1.0), t_offset=0.0, duration=T)  # min-snap
            for i, t in enumerate(times):
                poly.add_constraint(t, 0, pts[i, ax])  # fly THROUGH every waypoint
            for d in (1, 2):  # start + goal at rest: zero velocity + acceleration
                poly.add_constraint(0.0, d, 0.0)
                poly.add_constraint(T, d, 0.0)
            poly.solve()
            self._polys.append(poly)

    def _pos_vel_acc_jerk(self, t: float):
        if self._polys is None:
            self._build()
        t = min(max(t, 0.0), self._total_dur)  # clamp: hold at the goal; the poly diverges outside [0, T]
        p = np.array([poly(t, 0) for poly in self._polys])
        v = np.array([poly(t, 1) for poly in self._polys])
        a = np.array([poly(t, 2) for poly in self._polys])
        j = np.array([poly(t, 3) for poly in self._polys])
        return p, v, a, j

    def reference_path(self, n: int = 300) -> np.ndarray:
        """Sample the planned position path for visualization / logging: ``(n, 3)``, start → goal."""
        if self._polys is None:
            self._build()
        return np.array([self._pos_vel_acc_jerk(float(t))[0] for t in np.linspace(0.0, self._total_dur, n)])

    def _yaw_and_rate(self, t: float) -> tuple[float, float]:
        """Nose-first yaw straight from the smooth min-snap velocity:
        ``yaw = atan2(v_y, v_x)`` and ``ω_z = d/dt yaw = (v_x·a_y − v_y·a_x)/‖v_h‖²``.
        min-snap turns the velocity direction gently, so this yaw + its rate stay bounded, with no
        ruckig-style spikes. Sampled at ``t`` clamped just inside the endpoints, where the velocity ramps
        from/to rest so its direction, the travel heading toward wp1 / the final-approach heading, is
        well-defined instead of the 0/0 at rest.
        """
        if not self.forward_heading:
            return 0.0, 0.0
        teps = min(0.3, 0.15 * self._total_dur)
        tc = min(max(t, teps), self._total_dur - teps)
        _, v, a, _ = self._pos_vel_acc_jerk(tc)
        yaw = float(np.arctan2(v[1], v[0]))
        rate = (v[0] * a[1] - v[1] * a[0]) / (v[0] * v[0] + v[1] * v[1] + 0.25)  # denom floor guards low speed
        rate = float(np.clip(rate, -3.0, 3.0)) if 0.0 <= t <= self._total_dur else 0.0
        return yaw, rate

    def flat_state_at(self, t: float, yaw: float | None = None):
        """Lift the polynomial at ``t`` to the flat quad state ``(pos, quat_wxyz, vel, omega_body,
        collective_thrust)``. Attitude from the thrust vector +
        the yaw polynomial; body-rate roll/pitch from the body-frame jerk and ω_z from the yaw rate, the
        feed-forward that makes nose-first track.
        """
        p, v, a, j = self._pos_vel_acc_jerk(t)
        if yaw is None:
            yaw, yaw_rate = self._yaw_and_rate(t)
        else:
            yaw, yaw_rate = float(yaw), 0.0
        thrust_vec = a + np.array([0.0, 0.0, GRAVITY])  # specific thrust == a − g_vec, with g_vec = (0, 0, −g)
        c = np.linalg.norm(thrust_vec)
        z_b = thrust_vec / (c + 1e-9)
        quat = _quat_from_z_yaw(z_b, yaw)
        # bodyrate: body_jerk = R(q)ᵀ·jerk → ω_x=−body_jerk_y/thrust, ω_y=body_jerk_x/thrust; ω_z = yaw rate.
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b) + 1e-9
        x_b = np.cross(y_b, z_b)
        rot = np.column_stack([x_b, y_b, z_b])  # world ← body
        body_jerk = rot.T @ j
        omega = np.array([-body_jerk[1] / (c + 1e-9), body_jerk[0] / (c + 1e-9), yaw_rate])
        return p, quat, v, omega, self.mass * c
