"""FlatnessReference: the operator's trajectory planner, Plane 5.

This lives in the **operator** plane, not the controller: the operator turns mission intent, waypoints,
into a smooth, jerk-limited tracking reference and hands it to a tracking controller, the acados
Nonlinear Model Predictive Control (NMPC), as a ``ReferenceTrajectory`` setpoint. ruckig plans a
C²-continuous position trajectory through the waypoints, as chained legs, and differential flatness lifts it to the full quad state, attitude and body rate, + the
collective-thrust feedforward the NMPC tracks. The planner sits apart from the Model Predictive
Control (MPC), on the operator side of the seam.

Pure host Python, ruckig + numpy; no casadi or acados, so the operator stays solver-agnostic.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81


def _quat_from_z_yaw(z_body: np.ndarray, yaw: float) -> np.ndarray:
    """Minimal-tilt quaternion, wxyz, whose body +z is ``z_body`` and whose heading is ``yaw``: the
    standard differential-flatness attitude of Mellinger & Kumar. The thrust axis fixes roll/pitch, the
    yaw setpoint fixes the remaining Degree Of Freedom (DOF).
    """
    z_b = z_body / (np.linalg.norm(z_body) + 1e-9)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b) + 1e-9
    x_b = np.cross(y_b, z_b)
    rot = np.column_stack([x_b, y_b, z_b])  # body axes as columns
    # rotation matrix → quaternion, wxyz, numerically stable branch on the trace
    tr = np.trace(rot)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k]) * 2.0
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (rot[j, i] + rot[i, j]) / s
        q[k] = (rot[k, i] + rot[i, k]) / s
        w = (rot[k, j] - rot[j, k]) / s
        x, y, z = q
    out = np.array([w, x, y, z])
    return out / (np.linalg.norm(out) + 1e-9)


class FlatnessReference:
    """The ruckig reference: ruckig plans a jerk-limited position trajectory through the waypoints,
    chained C2-continuous legs, and differential flatness lifts it to a full state reference, attitude and
    body rate, plus the collective-thrust feedforward the NMPC tracks. Built by the **operator** and fed
    to the controller via ``accept_setpoint(ReferenceTrajectory(reference=ref))``.
    """

    def __init__(
        self, waypoints, *, mass, max_velocity=2.0, max_acceleration=3.0, max_jerk=15.0, cruise_frac=0.5,
        forward_heading: bool = False, yaw_speed_eps: float = 0.3, yaw_lookahead: float = 0.4,
    ):  # fmt: skip
        """Configure the planner; the plan builds lazily on first use, after :meth:`set_start`.

        Args:
            waypoints: Ordered flythrough positions ``[(x, y, z), ...]``, excluding the start.
            mass: Vehicle mass [kg]. Scales the specific thrust into the collective-thrust feedforward.
            max_velocity: Per-axis velocity limit [m/s] for the ruckig plan.
            max_acceleration: Per-axis acceleration limit [m/s²].
            max_jerk: Per-axis jerk limit [m/s³], the C²-continuity bound.
            cruise_frac: Fraction of ``max_velocity`` used as the flythrough speed at intermediate
                waypoints.
            forward_heading: Fly **nose-first**: the yaw reference points along the horizontal velocity,
                ``atan2(v_y, v_x)``, a velocity heading mode, so the drone turns to face where
                it's going. ``False`` holds a fixed 0-yaw heading, and the drone translates without turning.
            yaw_speed_eps: Horizontal speed [m/s] below which the velocity direction is ill-defined, at the
                start, the goal, or hover: the heading then borrows the nearest well-defined moving-heading
                along the trajectory, so the yaw reference stays smooth through the near-zero-speed
                endpoints.
            yaw_lookahead: Pure-pursuit lookahead [s] for the nose-first heading: the nose aims at the
                position this far ahead on the path, smoothing the yaw over the ruckig flythrough swings.
        """
        self.waypoints = np.array([[float(x) for x in wp_] for wp_ in waypoints], dtype=np.float64)
        self.mass = float(mass)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.cruise_frac = float(cruise_frac)
        self.forward_heading = bool(forward_heading)
        self._yaw_speed_eps = float(yaw_speed_eps)
        self._yaw_lookahead = float(yaw_lookahead)  # [s] pure-pursuit lookahead, smooths the nose-first yaw
        self._p0 = None
        self._segments = None
        self._total_dur = 0.0

    def set_start(self, p0) -> None:
        """Anchor the trajectory at the start position, the vehicle's pose on the first tick.

        Args:
            p0: The start position ``(x, y, z)``; the first leg runs from here to ``waypoints[0]``.
        """
        self._p0 = np.asarray(p0, dtype=np.float64)

    @property
    def duration(self) -> float:
        """Total trajectory duration [s]; builds the plan on first access. The operator uses it to end
        the run after the trajectory completes, + a settle hold.
        """
        if self._segments is None:
            self._build()
        return self._total_dur

    def _build(self) -> None:
        from ruckig import InputParameter, Result, Ruckig, Trajectory

        pts = np.vstack([self._p0, self.waypoints])  # start, wp1, …, goal
        cruise = self.cruise_frac * self.max_velocity
        vels = [np.zeros(3)]
        for i in range(1, len(pts) - 1):
            chord = pts[i + 1] - pts[i - 1]  # flythrough tangent at the intermediate waypoint
            vels.append(cruise * chord / (np.linalg.norm(chord) + 1e-9))
        vels.append(np.zeros(3))  # at rest at the final goal

        otg = Ruckig(3)
        self._segments, t_accum = [], 0.0
        for i in range(len(pts) - 1):
            inp = InputParameter(3)
            inp.current_position, inp.current_velocity, inp.current_acceleration = list(pts[i]), list(vels[i]), [0.0] * 3  # fmt: skip
            inp.target_position, inp.target_velocity, inp.target_acceleration = list(pts[i + 1]), list(vels[i + 1]), [0.0] * 3  # fmt: skip
            inp.max_velocity, inp.max_acceleration, inp.max_jerk = [self.max_velocity] * 3, [self.max_acceleration] * 3, [self.max_jerk] * 3  # fmt: skip
            traj = Trajectory(3)
            if otg.calculate(inp, traj) != Result.Working:
                raise RuntimeError(f"ruckig failed to plan reference leg {i}")
            self._segments.append((t_accum, traj))
            t_accum += traj.duration
        self._total_dur = t_accum

    def _reference_at(self, t: float):
        """Position, velocity, and acceleration at time ``t`` since launch; holds at the goal afterwards."""
        if self._segments is None:
            self._build()
        t = min(max(t, 0.0), self._total_dur)
        for t0, traj in self._segments:
            if t <= t0 + traj.duration:
                p, v, a = traj.at_time(t - t0)
                return np.asarray(p), np.asarray(v), np.asarray(a)
        t0, traj = self._segments[-1]
        p, v, a = traj.at_time(traj.duration)
        return np.asarray(p), np.asarray(v), np.asarray(a)

    def reference_path(self, n: int = 300) -> np.ndarray:
        """Sample the planned position path for visualization / logging.

        Args:
            n: Number of evenly time-spaced samples over the trajectory duration.

        Returns:
            np.ndarray: An ``(n, 3)`` array of reference positions from start to goal.
        """
        if self._segments is None:
            self._build()
        return np.array([self._reference_at(float(t))[0] for t in np.linspace(0.0, self._total_dur, n)])

    def _heading_at(self, t: float) -> float:
        """Nose-first heading: aim the nose at a **position lookahead** on the path, pure-pursuit style:
        ``atan2(p(t+T)_y − p(t)_y, ...x)``. This is a *smoothed* velocity direction: the ruckig flythrough
        plan swings the instantaneous velocity ~100° at
        each leg boundary, a sharp yaw the NMPC can't track without sacrificing position, but the direction
        to a point ``_yaw_lookahead`` seconds ahead turns gently. Where the lookahead displacement is too
        short to define a heading, at the goal or in hover, it borrows the nearest well-defined lookahead
        along the trajectory, so the yaw reference stays smooth + finite.
        """
        min_disp = self._yaw_speed_eps * self._yaw_lookahead  # min horizontal displacement for a defined heading
        p0 = self._reference_at(t)[0]
        for dt_probe in (0.0, 0.1, 0.2, 0.35, -0.1, -0.25, -0.5, -0.9):
            p1 = self._reference_at(t + dt_probe + self._yaw_lookahead)[0]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if dx * dx + dy * dy > min_disp * min_disp:
                return float(np.arctan2(dy, dx))
        return 0.0  # fully static trajectory → default heading

    def _heading_and_rate(self, t: float) -> tuple[float, float]:
        """The velocity-aligned heading + its rate ``dyaw/dt``, a wrap-safe central difference: the yaw
        setpoint + the ``ω_z`` feed-forward for nose-first tracking.
        """
        yaw = self._heading_at(t)
        h = 1e-2
        d = self._heading_at(t + h) - self._heading_at(t - h)
        d = np.arctan2(np.sin(d), np.cos(d))  # shortest signed angle; handles the ±π wrap
        return yaw, d / (2.0 * h)

    def flat_state_at(self, t: float, yaw: float | None = None):
        """Lift the position reference at ``t`` to the full flat state: returns
        ``(pos, quat_wxyz, vel, omega_body, collective_thrust)``. Attitude and body rate follow from the
        thrust vector ``a + g·ẑ`` and its derivative, the jerk, the standard quadrotor flatness map. The yaw
        is the **nose-first** velocity-aligned heading, + its ``ω_z`` rate, when ``forward_heading``; pass an
        explicit ``yaw`` to override, then ``ω_z = 0``, or it defaults to 0 with ``forward_heading=False``.
        """
        p, v, a = self._reference_at(t)
        if yaw is None and self.forward_heading:
            # Nose-first: aim at a position lookahead, which is smooth, + feed the yaw rate forward as ω_z:
            # ω_z is the yaw rate. The lookahead is what
            # makes the ff usable: the min-snap yaw is a smooth polynomial with bounded ẏaw, and the
            # pure-pursuit lookahead likewise smooths the ruckig heading here, so ω_z stays bounded instead
            # of spiking at the flythrough velocity swings.
            yaw, yaw_rate = self._heading_and_rate(t)
        else:
            yaw = 0.0 if yaw is None else float(yaw)
            yaw_rate = 0.0
        g_vec = np.array([0.0, 0.0, GRAVITY])
        thrust_vec = a + g_vec  # specific thrust the trajectory demands; points along body +z
        c = np.linalg.norm(thrust_vec)
        z_b = thrust_vec / (c + 1e-9)
        quat = _quat_from_z_yaw(z_b, yaw)
        # body rate from jerk: ż_b = ω × z_b, so the in-plane jerk component gives ω_x, ω_y; ω_z is the
        # yaw-rate feed-forward, ≈ the world yaw rate for near-level flight, good enough for the NMPC to track.
        h = 1e-3
        _, _, a_p = self._reference_at(t + h)
        _, _, a_m = self._reference_at(t - h)
        jerk = (a_p - a_m) / (2.0 * h)
        z_dot = (jerk - np.dot(jerk, z_b) * z_b) / (c + 1e-9)  # component of ż_b ⟂ z_b
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b) + 1e-9
        x_b = np.cross(y_b, z_b)
        omega = np.array([-np.dot(z_dot, y_b), np.dot(z_dot, x_b), yaw_rate])  # roll/pitch from jerk, yaw-rate ff
        return p, quat, v, omega, self.mass * c
