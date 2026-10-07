"""Sensors: Inertial Measurement Unit (IMU), magnetometer, barometer and Global Positioning
System (GPS), all **Warp-native**.

Each sensor's math is a ``@wp.kernel`` over the live ``newton.State``'s device arrays,
``body_q`` / ``body_qd``, the zero-copy Warp accessors, and its noise is the Warp Random Number
Generator (RNG), ``wp.rand``, so the sensor stages join a CUDA graph and uniform with
the rest of the device step: no host NumPy / ``math`` in the per-tick path. The kernels replicate the
canonical frame math from :mod:`nexus_sim._src.transform` **verbatim**.

Each tick reads the result back once into the host :class:`Measurement` dataclass, which the PX4
``Controller`` serialises to MAVLink for its peer, an external process: exactly the
"captured region = device step minus the controller" split. Each sensor splits
that into :meth:`sample_wp`, which launches the kernel into its device buffer, the "Warp Measurement
buffer" with no readback, so it joins a CUDA graph, and :meth:`read`, the single D2H into the host
``Measurement`` after replay; ``sample`` = both, the eager path.

**Determinism, re-baselined onto the Warp RNG.** Noise comes from ``wp.rand_init(seed, step*16 + axis)``
with a per-sensor seed, which the builder derives from the run's seed and the sensor's prim, and the
per-tick ``step`` index: an
independent, reproducible noise field per sensor, the same bits run to run. Each sensor draws
its own sequence, not one shared ``random.Random`` stream.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.recording.sensor import SensorRecorder


class DeviceSensor(SensorRecorder):
    """A sensor whose work is one device stage, named after the sensor, over ``sample_wp``."""

    def stages(self) -> list[Stage]:
        return [Stage(self.name, "device", lambda tick: self.sample_wp(tick.state, tick.t))]


def _at_body_origin(run, kind: str) -> None:
    """Fail a sensor that models no mount offset when its prim sits off its body's origin.

    Raises:
        ValueError: The prim's translation isn't zero; the message names the prim.
    """
    if tuple(run.mount) != (0.0, 0.0, 0.0):
        raise ValueError(
            f"{run.path}: a {kind} models no mount offset, and this prim sits {tuple(run.mount)} from its "
            "body's origin; author it at the origin"
        )


def _rigid_mount(run, kind: str) -> None:
    """Fail a sensor whose prim's transform scales, shears or mirrors it: a mount only moves and turns.

    Raises:
        ValueError: The prim's transform isn't rigid; the message names the prim.
    """
    if run.prim is None:
        return
    from pxr import UsdGeom

    m = np.array(UsdGeom.Xformable(run.prim).GetLocalTransformation())[:3, :3]
    if not np.allclose(m @ m.T, np.eye(3), atol=1e-6) or np.linalg.det(m) < 0.0:
        raise ValueError(
            f"{run.path}: a {kind} mount only moves and turns, and this prim's transform scales, shears or "
            "mirrors it; author a translation and a rotation alone"
        )


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
    body: int,  # the model body the sensor rides
    r_com_to_mount: wp.vec3,  # Center Of Mass (COM) to mount, in the body's axes
    q_mount: wp.quat,  # the mount's rotation in the body's axes
    dt: float,
    first: int,
    seed: int,
    step: wp.array(dtype=int),
    sigma_acc: float,
    sigma_gyro: float,
    prev_lin: wp.array(dtype=wp.vec3),
    prev_ang: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),  # [xacc,yacc,zacc, xgyro,ygyro,zgyro], in the mount's axes
):
    st = step[0]
    q = wp.transform_get_rotation(body_q[body])
    vlin = wp.spatial_top(body_qd[body])  # world, at COM
    vang = wp.spatial_bottom(body_qd[body])  # world
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
    q_sensor = q * q_mount  # the mount's axes in the world
    acc_s = wp.quat_rotate_inv(q_sensor, acc_mount - gravity_world)  # specific force
    gyro_s = wp.quat_rotate_inv(q_sensor, vang)
    out[0] = acc_s[0] + _noise(seed, st, 0, sigma_acc)
    out[1] = acc_s[1] + _noise(seed, st, 1, sigma_acc)
    out[2] = acc_s[2] + _noise(seed, st, 2, sigma_acc)
    out[3] = gyro_s[0] + _noise(seed, st, 3, sigma_gyro)
    out[4] = gyro_s[1] + _noise(seed, st, 4, sigma_gyro)
    out[5] = gyro_s[2] + _noise(seed, st, 5, sigma_gyro)


@wp.kernel
def mag_kernel(
    body_q: wp.array(dtype=wp.transform),
    mag_ned: wp.vec3,
    offset: wp.vec3,
    seed: int,
    step: wp.array(dtype=int),
    sigma: wp.vec3,
    body: int,  # the model body the sensor rides
    out: wp.array(dtype=float),  # [xmag, ymag, zmag]
):
    st = step[0]
    q = wp.transform_get_rotation(body_q[body])
    # NED -> world is a PROPER rotation: X=N, Y=-E, which is west, Z=-D; the frame contract lives in
    # ``nexus_sim._src.transform``. The old Y=+E map was a reflection, GH #61.
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
    body: int,  # the model body the sensor rides
    out: wp.array(dtype=float),  # [abs_pressure, pressure_alt]
):
    st = step[0]
    alt = wp.transform_get_translation(body_q[body])[2]  # z up in sim
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
    body: int,  # the model body the sensor rides
    out: wp.array(dtype=wp.float64),  # [lat, lon, alt, vn, ve, vd, ground_speed]
):
    p = wp.transform_get_translation(body_q[body])
    v = wp.spatial_top(body_qd[body])  # world linear vel
    # lat/lon/alt in float64: a float32 ``ref_lat (~47.6) + small offset`` would lose ~1e-5 deg,
    # about 1 m, which PX4 then quantizes to degE7, so the geodetic mapping must be double precision.
    # World -> geodetic uses the same axis map as everything else: north = +x, east = -y, down =
    # -z, the frame contract in ``nexus_sim._src.transform``. Position and velocity must share
    # it; when they disagreed, PX4's EKF reset in a loop and east guidance ran away, GH #61.
    out[0] = ref_lat + wp.float64(p[0]) / wp.float64(111000.0)
    out[1] = ref_lon - wp.float64(p[1]) * inv_lon_scale
    out[2] = ref_alt + wp.float64(p[2])
    out[3] = wp.float64(v[0])  # vn
    out[4] = wp.float64(-v[1])  # ve, since world +y = West
    out[5] = wp.float64(-v[2])  # vd
    out[6] = wp.float64(wp.sqrt(v[0] * v[0] + v[1] * v[1]))  # ground speed


class ImuSensor(DeviceSensor):
    """Accelerometer, which reads specific force, and gyroscope, in the axes of the sensor's mount:
    Warp kernel + Warp RNG.

    The mount is the sensor's prim under the body it rides: its translation and its rotation. The
    accelerometer adds the lever arm ``alpha x r + omega x (omega x r)``, where ``r`` runs from the body's
    center of mass to the mount, as Newton's ``SensorIMU`` does.

    The accelerometer finite-differences the body's velocity over the control tick, so it reports the
    mean acceleration over the tick before, half a tick late, on purpose: PX4's estimator
    integrates delta velocity over each sample interval, and a tick mean is that interval's average.
    Newton's ``SensorIMU`` reads the solver's acceleration at the tick's end instead.

    The sensor reports no attitude: the vehicle's true state isn't a sensor reading.

    Args:
        run: The run's values: its seed, the tick, the site's gravity, the body and the mount.
        acc_noise: Standard deviation of the accelerometer's white noise, m/s^2.
        gyro_noise: Standard deviation of the gyroscope's white noise, rad/s.
        rate: The declared sample rate, hertz. The sensor samples every tick whatever it says.

    Raises:
        ValueError: The prim's transform scales, shears or mirrors the mount; the message names the prim.
    """

    name = "imu"  # sim.sensors key, the flat instance name
    fields = ("xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro")  # _out layout

    def __init__(self, run, acc_noise: float = 0.02, gyro_noise: float = 0.02, rate: float = 0.0):
        _rigid_mount(run, "inertial measurement unit")
        self.seed = run.seed
        self.dt = float(run.dt)
        self.body = int(run.body)
        self.rate = float(rate)
        # The site's gravity, along world -Z, the same value the physics applies; the accelerometer
        # reports specific force, so at rest it reads minus this.
        self.gravity_world = wp.vec3(0.0, 0.0, -float(run.site.gravity))
        self.r_com_to_mount = wp.vec3(*[float(m) - float(c) for m, c in zip(run.mount, run.com, strict=True)])
        self.q_mount = wp.quat(*[float(x) for x in run.rotation])
        self.acc_noise = float(acc_noise)
        self.gyro_noise = float(gyro_noise)
        self._prev_lin = wp.zeros(1, dtype=wp.vec3)
        self._prev_ang = wp.zeros(1, dtype=wp.vec3)
        self._out = wp.zeros(6, dtype=float)
        self._step = wp.zeros(1, dtype=int)  # per-tick counter, incremented in-graph so the noise varies
        self._first = True

    def sample_wp(self, state, t) -> None:
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
                self.gravity_world,
                self.body,
                self.r_com_to_mount,
                self.q_mount,
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

    def sample(self, state, t, out) -> None:
        self.sample_wp(state, t)
        self.read(out)


class MagSensor(DeviceSensor):
    name = "mag"
    fields = ("xmag", "ymag", "zmag")

    def __init__(self, run, offset=(0.0, 0.0, 0.0), noise=(0.02, 0.02, 0.03), rate: float = 0.0):
        # The field is the site's, a North East Down (NED) vector in gauss, resolved at build.
        # ``noise`` is the per-axis Gaussian sigma in Gauss. The default, 0.02/0.02/0.03, is the
        # bridge value; note it's ~10x a real magnetometer and, against PX4's strict per-sample World
        # Magnetic Model (WMM) strength check, intermittently trips the "magnetic interference" check,
        # so the shipped vehicles author a lower sigma.
        _at_body_origin(run, "magnetometer")
        self.seed = run.seed
        self.body = int(run.body)
        self.rate = float(rate)
        self.offset = tuple(float(x) for x in offset)
        self.noise = tuple(float(x) for x in noise)
        self.mag_ned = wp.vec3(*[float(x) for x in run.site.mag_ned])
        self._offset = wp.vec3(*self.offset)
        self._sigma = wp.vec3(*self.noise)
        self._out = wp.zeros(3, dtype=float)
        self._step = wp.zeros(1, dtype=int)

    def sample_wp(self, state, t) -> None:
        wp.launch(increment_step, dim=1, inputs=(self._step,))
        wp.launch(
            mag_kernel,
            dim=1,
            inputs=(state.body_q, self.mag_ned, self._offset, self.seed, self._step, self._sigma, self.body),
            outputs=(self._out,),
        )

    def read(self, out) -> None:
        r = self._out.numpy()
        out.xmag, out.ymag, out.zmag = float(r[0]), float(r[1]), float(r[2])

    def sample(self, state, t, out) -> None:
        self.sample_wp(state, t)
        self.read(out)


class BaroSensor(DeviceSensor):
    name = "baro"
    fields = ("abs_pressure", "pressure_alt")

    def __init__(self, run, noise: float = 0.02, rate: float = 0.0):
        # The site's air pressure at mean sea level, hPa, and its temperature, degrees Celsius.
        _at_body_origin(run, "barometer")
        self.seed = run.seed
        self.body = int(run.body)
        self.rate = float(rate)
        self.pressure_msl = float(run.site.pressure_msl)
        self.temperature = float(run.site.temperature)
        self.noise = float(noise)
        self._out = wp.zeros(2, dtype=float)
        self._step = wp.zeros(1, dtype=int)

    def sample_wp(self, state, t) -> None:
        wp.launch(increment_step, dim=1, inputs=(self._step,))
        wp.launch(
            baro_kernel,
            dim=1,
            inputs=(state.body_q, self.pressure_msl, self.seed, self._step, self.noise, self.body),
            outputs=(self._out,),
        )

    def read(self, out) -> None:
        r = self._out.numpy()
        out.abs_pressure, out.pressure_alt = float(r[0]), float(r[1])
        out.temperature = self.temperature

    def sample(self, state, t, out) -> None:
        self.sample_wp(state, t)
        self.read(out)


class GpsSensor(DeviceSensor):
    name = "gps"
    fields = ("lat", "lon", "alt", "vn", "ve", "vd", "ground_speed")  # _out is float64 for lat/lon precision

    def __init__(self, run, fix_type: int = 3, rate: float = 0.0):
        import math

        # The reference is the site's geodetic origin, where the world's origin sits on Earth.
        _at_body_origin(run, "GPS receiver")
        self.body = int(run.body)
        self.rate = float(rate)
        self.ref_lat = float(run.site.lat)
        self.ref_lon = float(run.site.lon)
        self.ref_alt = float(run.site.alt)
        self.inv_lon_scale = 1.0 / (111000.0 * math.cos(math.radians(self.ref_lat)))
        self.fix_type = int(fix_type)
        self._out = wp.zeros(7, dtype=wp.float64)

    def sample_wp(self, state, t) -> None:
        wp.launch(
            gps_kernel,
            dim=1,
            inputs=(
                state.body_q,
                state.body_qd,
                self.ref_lat,
                self.ref_lon,
                self.ref_alt,
                self.inv_lon_scale,
                self.body,
            ),
            outputs=(self._out,),
        )

    def read(self, out) -> None:
        r = self._out.numpy()
        out.gps_valid = True
        out.lat_deg, out.lon_deg, out.alt_m = float(r[0]), float(r[1]), float(r[2])
        out.vn, out.ve, out.vd = float(r[3]), float(r[4]), float(r[5])
        out.ground_speed = float(r[6])
        out.fix_type = self.fix_type

    def sample(self, state, t, out) -> None:
        self.sample_wp(state, t)
        self.read(out)
