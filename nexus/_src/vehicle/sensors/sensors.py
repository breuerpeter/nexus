"""Sensor plugins: Inertial Measurement Unit (IMU), magnetometer, barometer and Global Positioning
System (GPS), per architecture.md §2, all **Warp-native**.

Each sensor's math is a ``@wp.kernel`` over the live ``newton.State``'s device arrays,
``body_q`` / ``body_qd``, the zero-copy Warp accessors, and its noise is the Warp Random Number
Generator (RNG), ``wp.rand``, so the sensor region is capturable into a CUDA graph and uniform with
the rest of the device step: no host NumPy / ``math`` in the per-tick path. The kernels replicate the
canonical frame math from :mod:`nexus._src.transform` **verbatim**, including the bridge's known
North East Down (NED) axis inconsistency, preserved for PX4 parity; see that module's warning.

Each tick reads the result back once into the host :class:`Measurement` dataclass, which the PX4
``Controller`` serialises to MAVLink: the one host boundary, an external process, exactly the
"captured region = device step minus the controller" split of architecture.md §5. Each sensor splits
that into :meth:`sample_wp`, which launches the kernel into its device buffer, the "Warp Measurement
buffer" with no readback, so it joins a CUDA graph, and :meth:`read`, the single D2H into the host
``Measurement`` after replay; ``sample`` = both, the eager path.

**Determinism, re-baselined onto the Warp RNG.** Noise comes from ``wp.rand_init(seed, step*16 + axis)``
with a per-sensor seed from :meth:`SeedTree.seed_for` and the per-tick ``step`` index: an
independent, reproducible noise field per sensor, the same bits run to run. This intentionally
replaces the bridge's single shared ``random.Random`` sequence, the SeedTree forward design.
"""

from __future__ import annotations

import warp as wp

from nexus._src.recording.sensor import SensorRecorder


@wp.func
def _noise(seed: int, step: int, axis: int, sigma: float) -> float:
    """Reproducible Gaussian noise for one (sensor seed, tick, axis): ``sigma * N(0,1)``."""
    r = wp.rand_init(seed, step * 16 + axis)
    return sigma * wp.randn(r)


@wp.kernel
def increment_step(step: wp.array(dtype=int)):
    """Advance the per-sensor tick counter. Launched inside a captured region so the noise field
    varies per **graph replay**; without this the baked-at-capture step freezes the sensor stream,
    which PX4's Extended Kalman Filter (EKF) flags as a stuck sensor: no GPS/position fusion, so it
    won't arm.
    """
    step[0] = step[0] + 1


@wp.kernel
def imu_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    gravity_world: wp.vec3,
    r_com_to_mount: wp.vec3,  # Center Of Mass (COM)->mount offset, body frame; zero = COM-mounted
    dt: float,
    first: int,
    seed: int,
    step: wp.array(dtype=int),
    sigma_acc: float,
    sigma_gyro: float,
    prev_lin: wp.array(dtype=wp.vec3),
    prev_ang: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),  # [xacc,yacc,zacc, xgyro,ygyro,zgyro, qw,qx,qy,qz]
):
    st = step[0]
    q = wp.transform_get_rotation(body_q[0])
    vlin = wp.spatial_top(body_qd[0])  # world, at COM
    vang = wp.spatial_bottom(body_qd[0])  # world
    if first == 1:  # first tick: seed the finite-diff so acceleration starts at zero
        prev_lin[0] = vlin
        prev_ang[0] = vang
    acc_com = (vlin - prev_lin[0]) / dt
    alpha = (vang - prev_ang[0]) / dt
    prev_lin[0] = vlin
    prev_ang[0] = vang
    # lever arm: a_mount = a_com + alpha x r + omega x (omega x r), r = R(q)*r_body
    r_w = wp.quat_rotate(q, r_com_to_mount)
    acc_mount = acc_com + wp.cross(alpha, r_w) + wp.cross(vang, wp.cross(vang, r_w))
    grav_b = wp.quat_rotate_inv(q, gravity_world)
    acc_b = wp.quat_rotate_inv(q, acc_mount)
    gyro_b = wp.quat_rotate_inv(q, vang)
    out[0] = acc_b[0] - grav_b[0] + _noise(seed, st, 0, sigma_acc)  # specific force, body Forward Right Down (FRD)
    out[1] = acc_b[1] - grav_b[1] + _noise(seed, st, 1, sigma_acc)
    out[2] = acc_b[2] - grav_b[2] + _noise(seed, st, 2, sigma_acc)
    out[3] = gyro_b[0] + _noise(seed, st, 3, sigma_gyro)
    out[4] = gyro_b[1] + _noise(seed, st, 4, sigma_gyro)
    out[5] = gyro_b[2] + _noise(seed, st, 5, sigma_gyro)
    out[6] = q[3]  # quat wire order [w, x, y, z] from [x, y, z, w]
    out[7] = q[0]
    out[8] = q[1]
    out[9] = q[2]


@wp.kernel
def mag_kernel(
    body_q: wp.array(dtype=wp.transform),
    mag_ned: wp.vec3,
    offset: wp.vec3,
    seed: int,
    step: wp.array(dtype=int),
    sigma: wp.vec3,
    out: wp.array(dtype=float),  # [xmag, ymag, zmag]
):
    st = step[0]
    q = wp.transform_get_rotation(body_q[0])
    # NED -> world is a PROPER rotation: X=N, Y=-E, which is west, Z=-D; the frame contract lives in
    # ``nexus._src.transform``. The old Y=+E map was a reflection, GH #61.
    mag_world = wp.vec3(mag_ned[0], -mag_ned[1], -mag_ned[2])
    mag_b = wp.quat_rotate_inv(q, mag_world)
    out[0] = mag_b[0] + offset[0] + _noise(seed, st, 0, sigma[0])
    out[1] = mag_b[1] + offset[1] + _noise(seed, st, 1, sigma[1])
    out[2] = mag_b[2] + offset[2] + _noise(seed, st, 2, sigma[2])


@wp.kernel
def baro_kernel(
    body_q: wp.array(dtype=wp.transform),
    pressure_msl: float,
    seed: int,
    step: wp.array(dtype=int),
    sigma: float,
    out: wp.array(dtype=float),  # [abs_pressure, pressure_alt]
):
    st = step[0]
    alt = wp.transform_get_translation(body_q[0])[2]  # z up in sim
    out[0] = pressure_msl * wp.pow(1.0 - 2.25577e-5 * alt, 5.25588) + _noise(seed, st, 0, sigma)
    out[1] = alt + _noise(seed, st, 1, sigma)


@wp.kernel
def gps_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    ref_lat: wp.float64,
    ref_lon: wp.float64,
    ref_alt: wp.float64,
    inv_lon_scale: wp.float64,  # 1 / (111000 * cos(radians(ref_lat)))
    out: wp.array(dtype=wp.float64),  # [lat, lon, alt, vn, ve, vd, ground_speed]
):
    p = wp.transform_get_translation(body_q[0])
    v = wp.spatial_top(body_qd[0])  # world linear vel
    # lat/lon/alt in float64: a float32 ``ref_lat (~47.6) + small offset`` would lose ~1e-5 deg,
    # about 1 m, which PX4 then quantizes to degE7, so the geodetic mapping must be double precision.
    # World -> geodetic uses the same axis map as everything else: north = +x, east = -y, down =
    # -z, the frame contract in ``nexus._src.transform``. Position and velocity must share
    # it; when they disagreed, PX4's EKF reset in a loop and east guidance ran away, GH #61.
    out[0] = ref_lat + wp.float64(p[0]) / wp.float64(111000.0)
    out[1] = ref_lon - wp.float64(p[1]) * inv_lon_scale
    out[2] = ref_alt + wp.float64(p[2])
    out[3] = wp.float64(v[0])  # vn
    out[4] = wp.float64(-v[1])  # ve, since world +y = West
    out[5] = wp.float64(-v[2])  # vd
    out[6] = wp.float64(wp.sqrt(v[0] * v[0] + v[1] * v[1]))  # ground speed


class ImuSensor(SensorRecorder):
    """Accelerometer, which reads specific force, + gyro, body FRD: Warp kernel + Warp RNG. Owns
    the earlier-tick velocities, persistent device arrays, for the finite-difference
    acceleration / angular acceleration, and supports a mount at ``mount_offset`` on a body whose COM
    is ``com``, the ``alpha x r + omega x (omega x r)`` lever-arm term; both default to the origin.
    """

    capturable = True
    name = "imu"  # sim.sensors key, the flat instance name
    fields = ("xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro", "qw", "qx", "qy", "qz")  # _out layout

    def __init__(
        self,
        seedtree,
        dt: float,
        mount_offset=(0.0, 0.0, 0.0),
        com=(0.0, 0.0, 0.0),
        acc_noise: float = 0.02,
        gyro_noise: float = 0.02,
    ):
        self.seed = seedtree.seed_for("imu")
        self.dt = float(dt)
        self.r_com_to_mount = wp.vec3(
            float(mount_offset[0] - com[0]), float(mount_offset[1] - com[1]), float(mount_offset[2] - com[2])
        )
        self.acc_noise = float(acc_noise)
        self.gyro_noise = float(gyro_noise)
        self._prev_lin = wp.zeros(1, dtype=wp.vec3)
        self._prev_ang = wp.zeros(1, dtype=wp.vec3)
        self._out = wp.zeros(10, dtype=float)
        self._step = wp.zeros(1, dtype=int)  # per-tick counter, incremented in-graph so the noise varies
        self._first = True

    def sample_wp(self, state, env, t) -> None:
        """Launch the IMU kernel into the device buffer ``self._out``, with NO host readback, so it joins
        a captured region. Increments the device step counter first so the noise field varies per
        replay. For a captured region, call once eagerly first to seed the finite-diff ``prev``
        velocities, then capture with ``first`` already 0.
        """
        wp.launch(increment_step, dim=1, inputs=(self._step,))
        wp.launch(
            imu_kernel,
            dim=1,
            inputs=(
                state.body_q,
                state.body_qd,
                wp.vec3(*env.gravity_world),
                self.r_com_to_mount,
                self.dt,
                1 if self._first else 0,
                self.seed,
                self._step,
                self.acc_noise,
                self.gyro_noise,
                self._prev_lin,
                self._prev_ang,
            ),
            outputs=(self._out,),
        )
        self._first = False

    def read(self, out) -> None:
        """One D2H of the device buffer into the host ``Measurement``, the PX4-controller boundary."""
        r = self._out.numpy()
        out.xacc, out.yacc, out.zacc = float(r[0]), float(r[1]), float(r[2])
        out.xgyro, out.ygyro, out.zgyro = float(r[3]), float(r[4]), float(r[5])
        out.quat_wxyz = (float(r[6]), float(r[7]), float(r[8]), float(r[9]))
        out.rollspeed, out.pitchspeed, out.yawspeed = out.xgyro, out.ygyro, out.zgyro

    def sample(self, state, env, t, out) -> None:
        self.sample_wp(state, env, t)
        self.read(out)


class MagSensor(SensorRecorder):
    capturable = True
    name = "mag"
    fields = ("xmag", "ymag", "zmag")

    def __init__(self, seedtree, mag_offset=(0.0, 0.0, 0.0), noise=(0.02, 0.02, 0.03)):
        # ``noise`` is the per-axis Gaussian sigma in Gauss. The default, 0.02/0.02/0.03, is the
        # bridge value; note it's ~10x a real magnetometer and, against PX4's strict per-sample World
        # Magnetic Model (WMM) strength check, intermittently trips the "magnetic interference" check,
        # so the PX4 Hardware In The Loop (HIL) path passes a lower sigma.
        self.seed = seedtree.seed_for("mag")
        self.offset = wp.vec3(*[float(x) for x in mag_offset])
        self.sigma = wp.vec3(*[float(x) for x in noise])
        self._out = wp.zeros(3, dtype=float)
        self._step = wp.zeros(1, dtype=int)

    def sample_wp(self, state, env, t) -> None:
        wp.launch(increment_step, dim=1, inputs=(self._step,))
        wp.launch(
            mag_kernel,
            dim=1,
            inputs=(state.body_q, wp.vec3(*env.mag_ned), self.offset, self.seed, self._step, self.sigma),
            outputs=(self._out,),
        )

    def read(self, out) -> None:
        r = self._out.numpy()
        out.xmag, out.ymag, out.zmag = float(r[0]), float(r[1]), float(r[2])

    def sample(self, state, env, t, out) -> None:
        self.sample_wp(state, env, t)
        self.read(out)


class BaroSensor(SensorRecorder):
    capturable = True
    name = "baro"
    fields = ("abs_pressure", "pressure_alt")

    def __init__(self, seedtree, noise: float = 0.02):
        self.seed = seedtree.seed_for("baro")
        self.noise = float(noise)
        self._out = wp.zeros(2, dtype=float)
        self._step = wp.zeros(1, dtype=int)

    def sample_wp(self, state, env, t) -> None:
        wp.launch(increment_step, dim=1, inputs=(self._step,))
        wp.launch(
            baro_kernel,
            dim=1,
            inputs=(state.body_q, float(env.air_pressure_msl), self.seed, self._step, self.noise),
            outputs=(self._out,),
        )
        self._temperature = env.temperature

    def read(self, out) -> None:
        r = self._out.numpy()
        out.abs_pressure, out.pressure_alt = float(r[0]), float(r[1])
        out.temperature = getattr(self, "_temperature", 25.0)

    def sample(self, state, env, t, out) -> None:
        self.sample_wp(state, env, t)
        self.read(out)


class GpsSensor(SensorRecorder):
    capturable = True
    name = "gps"
    fields = ("lat", "lon", "alt", "vn", "ve", "vd", "ground_speed")  # _out is float64 for lat/lon precision

    def __init__(self, ref_lat: float, ref_lon: float, ref_alt: float, fix_type: int = 3):
        import math

        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        self.ref_alt = float(ref_alt)
        self.inv_lon_scale = 1.0 / (111000.0 * math.cos(math.radians(ref_lat)))
        self.fix_type = int(fix_type)
        self._out = wp.zeros(7, dtype=wp.float64)

    def sample_wp(self, state, env, t) -> None:
        wp.launch(
            gps_kernel,
            dim=1,
            inputs=(state.body_q, state.body_qd, self.ref_lat, self.ref_lon, self.ref_alt, self.inv_lon_scale),
            outputs=(self._out,),
        )

    def read(self, out) -> None:
        r = self._out.numpy()
        out.gps_valid = True
        out.lat_deg, out.lon_deg, out.alt_m = float(r[0]), float(r[1]), float(r[2])
        out.vn, out.ve, out.vd = float(r[3]), float(r[4]), float(r[5])
        out.ground_speed = float(r[6])
        out.fix_type = self.fix_type

    def sample(self, state, env, t, out) -> None:
        self.sample_wp(state, env, t)
        self.read(out)
