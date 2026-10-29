"""Sensors: Inertial Measurement Unit (IMU), magnetometer, barometer and Global Positioning
System (GPS), all **Warp-native**, each on a site of Newton's model.

A sensor's class takes the model and its site, the label of the site its prim loaded as, then its schema's
attributes as keywords, as Newton's own sensors do. It reads the body it rides, its mount and the body's
center of mass from the model's arrays, and the world's values, gravity, the magnetic field, the pressure
at sea level, the temperature and the geodetic origin, from the model's world attributes, which the scene's
``PhysicsScene`` authors. Nothing samples those per tick.

Each sensor's math is a ``@wp.kernel`` over the live ``newton.State``'s device arrays,
``body_q`` / ``body_qd``, the zero-copy Warp accessors, and its noise is the Warp Random Number
Generator (RNG), ``wp.rand``, so the sensor stages join a CUDA graph and uniform with
the rest of the device step: no host NumPy / ``math`` in the per-tick path. The kernels replicate the
canonical frame math from :mod:`nexus_uav._src.transform` **verbatim**.

Each sensor's :meth:`sample_wp` launches its kernel, which writes the sample, stamped with the tick's sim
time, to the sensor's output signal on the device, with no readback, so it joins a CUDA graph. The kernel
writes a sample only on a tick that the sensor's declared rate makes due, and the check runs on the device
too, so between two samples the signal holds the last one, and its time marks a new one. A reader takes the
sample there: an estimator, or the PX4 controller, which serialises it to MAVLink for its peer in a host
stage.

**Determinism, re-baselined onto the Warp RNG.** Noise comes from ``wp.rand_init(seed, step*16 + axis)``
with a per-sensor seed, which the builder derives from the run's seed and the sensor's prim and passes as
the keyword ``seed``, and the sample's index ``step``, which the sensor counts on the device, so each graph
replay draws fresh noise: an independent, reproducible noise field per sensor, the same bits run to run.
Each sensor draws its own sequence, not one shared ``random.Random`` stream, and a tick with no new sample
draws none.
"""

from __future__ import annotations

import warp as wp
from newton import ShapeFlags

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.schema import BaroSample, GpsSample, ImuSample, MagSample
from nexus_uav._src.core.signals import Signal

from .rate import checked_rate, due

STANDARD_GRAVITY = 9.80665  # m/s^2: the g a datasheet states an accelerometer's range in, not the world's gravity


def site_index(model, site: str) -> int:
    """The index of the site of `model` labelled `site`, the path of the prim Newton loaded it from.

    Raises:
        ValueError: No shape of the model carries the label, or the shape is no site; the message names
            the label.
    """
    labels = [str(label) for label in model.shape_label]
    if site not in labels:
        raise ValueError(
            f"{site}: no site of the model has this label; a sensor's prim is a geometry prim with the sensor's "
            "schema applied, which Newton loads as a site"
        )
    index = labels.index(site)
    if not model.shape_flags.numpy()[index] & ShapeFlags.SITE:
        raise ValueError(f"{site}: the shape with this label is no site; apply the sensor's schema to it")
    return index


class DeviceSensor(Component):
    """A sensor whose work is one device stage, named after the sensor, over ``sample_wp``. The stage reads
    the tick's sim time, the signals ``time`` and ``time_ns``, and writes the sensor's sample, stamped with it, to its output
    signal ``out`` on a tick that its rate makes due. On any other tick the kernel writes nothing, so the
    signal holds the last sample. The sensor records nothing itself: the Recorder keeps the signal's history.

    Args:
        output: The name of the output signal, such as ``imu``.
        sample: The output's type, a Warp struct whose first field is the sample's time.
        model: Newton's model, which holds the sensor's site and the world's values.
        site: The label of the sensor's site: the path of the prim that declares it.
        rate: How often the sensor gives a new sample, hertz: 0 gives one every tick, as does a rate faster
            than the tick rate.

    Raises:
        ValueError: The model has no site of that label, the site sits on no body, or the rate is negative;
            the message names the site.
    """

    def __init__(self, output: str, sample, model, site: str, rate: float = 0.0):
        self.time = Signal("time", wp.float64, shape=(1,))
        self.time_ns = Signal("time_ns", wp.int64, shape=(1,))
        self.out = Signal(output, sample, shape=(1,))
        self.model = model
        self.path = site
        self.site = site_index(model, site)
        self.body = int(model.shape_body.numpy()[self.site])
        if self.body < 0:
            raise ValueError(f"{site}: the site sits on no body; put the sensor's prim under the body it rides")
        self.rate = checked_rate(rate, site)
        self._start = wp.zeros(1, dtype=wp.int64)  # the sim time of the first sample, nanoseconds
        self._taken = wp.zeros(1, dtype=int)  # how many samples the sensor has taken

    def stages(self) -> list[Stage]:
        return [
            Stage(
                self.name,
                "device",
                lambda tick: self.sample_wp(tick.state, tick.t),
                reads=(self.time, self.time_ns),
                writes=(self.out,),
            )
        ]


def _at_body_origin(sensor: DeviceSensor, kind: str) -> None:
    """Fail a sensor that models no mount offset when its site sits off its body's origin.

    Raises:
        ValueError: The site's translation isn't zero; the message names the site.
    """
    mount = tuple(float(x) for x in sensor.model.shape_transform.numpy()[sensor.site][:3])
    if any(abs(x) > 1e-6 for x in mount):  # the importer's composed transform carries rounding
        raise ValueError(
            f"{sensor.path}: a {kind} models no mount offset, and this prim sits {mount} from its body's origin; "
            "author it at the origin"
        )


@wp.func
def _noise(seed: int, step: int, axis: int, sigma: float) -> float:
    """Reproducible Gaussian noise for one (sensor seed, sample, axis): ``sigma * N(0,1)``."""
    r = wp.rand_init(seed, step * 16 + axis)
    return sigma * wp.randn(r)


@wp.func
def _due(now: wp.int64, rate: wp.float64, start: wp.array(dtype=wp.int64), taken: wp.array(dtype=int)) -> bool:
    """Whether a sample is due at sim time ``now``, in nanoseconds, by every sensor's rule, :func:`.rate.due`; a
    due sample counts toward the next one's due time.

    The first sample's time and the count ``taken`` live on the device, so a graph replay counts too, and the count
    indexes each sample's noise. A count baked at capture would freeze the sensor stream, which PX4's Extended
    Kalman Filter (EKF) flags as a stuck sensor: no GPS/position fusion, so it won't arm.
    """
    k = taken[0]
    if not due(now - start[0], k, rate):
        return False
    if k == 0:
        start[0] = now
    taken[0] = k + 1
    return True


@wp.func
def _world(body_world: wp.array(dtype=int), body: int) -> int:
    """The world the body is in, which indexes the model's world values; a body in no world reads the first."""
    return wp.max(body_world[body], 0)


@wp.kernel
def imu_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    gravity: wp.array(dtype=wp.vec3),  # the model's gravity, per world
    body_world: wp.array(dtype=int),
    body_com: wp.array(dtype=wp.vec3),  # each body's center of mass, in its axes
    shape_transform: wp.array(dtype=wp.transform),  # each shape's transform in its body's axes
    site: int,  # the sensor's site, a shape of the model
    body: int,  # the model body the sensor rides
    seed: int,
    sigma_acc: float,
    sigma_gyro: float,
    acc_limit: float,  # the accelerometer's full-scale range, m/s^2: each axis reads at most this either way
    prev_lin: wp.array(dtype=wp.vec3),
    prev_ang: wp.array(dtype=wp.vec3),
    prev_t: wp.array(dtype=wp.float64),  # the sim time of the last sample
    time: wp.array(dtype=wp.float64),
    time_ns: wp.array(dtype=wp.int64),
    rate: wp.float64,
    start: wp.array(dtype=wp.int64),
    taken: wp.array(dtype=int),
    sample: wp.array(dtype=ImuSample),
):
    if not _due(time_ns[0], rate, start, taken):
        return
    st = taken[0]
    q = wp.transform_get_rotation(body_q[body])
    vlin = wp.spatial_top(body_qd[body])  # world, at COM
    vang = wp.spatial_bottom(body_qd[body])  # world
    if st == 1:  # the first sample: seed the finite difference so acceleration starts at zero
        prev_lin[0] = vlin
        prev_ang[0] = vang
        prev_t[0] = time[0]
    # The time since the last sample, over which the velocity changed. A sample at the time of the last one
    # reads no acceleration.
    interval = wp.float32(time[0] - prev_t[0])
    acc_com = wp.vec3(0.0)
    alpha = wp.vec3(0.0)
    if interval > 0.0:
        acc_com = (vlin - prev_lin[0]) / interval
        alpha = (vang - prev_ang[0]) / interval
    prev_lin[0] = vlin
    prev_ang[0] = vang
    prev_t[0] = time[0]
    mount = shape_transform[site]
    # lever arm: a_mount = a_com + alpha x r + omega x (omega x r), r = R(q)*r_body
    r_w = wp.quat_rotate(q, wp.transform_get_translation(mount) - body_com[body])
    acc_mount = acc_com + wp.cross(alpha, r_w) + wp.cross(vang, wp.cross(vang, r_w))
    q_sensor = q * wp.transform_get_rotation(mount)  # the mount's axes in the world
    g = gravity[_world(body_world, body)]
    acc_s = wp.quat_rotate_inv(q_sensor, acc_mount - g)  # specific force
    gyro_s = wp.quat_rotate_inv(q_sensor, vang)
    s = ImuSample()
    s.time = time[0]
    # A real part clips its noisy output, so each axis saturates after the noise.
    s.accel = wp.vec3(
        wp.clamp(acc_s[0] + _noise(seed, st, 0, sigma_acc), -acc_limit, acc_limit),
        wp.clamp(acc_s[1] + _noise(seed, st, 1, sigma_acc), -acc_limit, acc_limit),
        wp.clamp(acc_s[2] + _noise(seed, st, 2, sigma_acc), -acc_limit, acc_limit),
    )
    s.gyro = wp.vec3(
        gyro_s[0] + _noise(seed, st, 3, sigma_gyro),
        gyro_s[1] + _noise(seed, st, 4, sigma_gyro),
        gyro_s[2] + _noise(seed, st, 5, sigma_gyro),
    )
    sample[0] = s


@wp.kernel
def mag_kernel(
    body_q: wp.array(dtype=wp.transform),
    magnetic_field: wp.array(dtype=wp.vec3),  # the model's field in the world's axes, per world
    body_world: wp.array(dtype=int),
    offset: wp.vec3,
    seed: int,
    sigma: wp.vec3,
    body: int,  # the model body the sensor rides
    time: wp.array(dtype=wp.float64),
    time_ns: wp.array(dtype=wp.int64),
    rate: wp.float64,
    start: wp.array(dtype=wp.int64),
    taken: wp.array(dtype=int),
    sample: wp.array(dtype=MagSample),
):
    if not _due(time_ns[0], rate, start, taken):
        return
    st = taken[0]
    q = wp.transform_get_rotation(body_q[body])
    # The field is in the world's axes, X north, Y west, Z up, a PROPER rotation of North East Down;
    # the frame contract lives in ``nexus_uav._src.transform``. The old Y=+E map was a reflection, GH #61.
    mag_b = wp.quat_rotate_inv(q, magnetic_field[_world(body_world, body)])
    s = MagSample()
    s.time = time[0]
    s.field = wp.vec3(
        mag_b[0] + offset[0] + _noise(seed, st, 0, sigma[0]),
        mag_b[1] + offset[1] + _noise(seed, st, 1, sigma[1]),
        mag_b[2] + offset[2] + _noise(seed, st, 2, sigma[2]),
    )
    sample[0] = s


@wp.kernel
def baro_kernel(
    body_q: wp.array(dtype=wp.transform),
    sea_level_pressure: wp.array(dtype=float),  # the model's pressure at mean sea level, hPa, per world
    temperature: wp.array(dtype=float),  # the model's air temperature, degrees Celsius, per world
    body_world: wp.array(dtype=int),
    seed: int,
    sigma: float,
    body: int,  # the model body the sensor rides
    time: wp.array(dtype=wp.float64),
    time_ns: wp.array(dtype=wp.int64),
    rate: wp.float64,
    start: wp.array(dtype=wp.int64),
    taken: wp.array(dtype=int),
    sample: wp.array(dtype=BaroSample),
):
    if not _due(time_ns[0], rate, start, taken):
        return
    st = taken[0]
    world = _world(body_world, body)
    alt = wp.transform_get_translation(body_q[body])[2]  # z up in sim
    s = BaroSample()
    s.time = time[0]
    s.pressure = sea_level_pressure[world] * wp.pow(1.0 - 2.25577e-5 * alt, 5.25588) + _noise(seed, st, 0, sigma)
    s.altitude = alt + _noise(seed, st, 1, sigma)
    s.temperature = temperature[world]
    sample[0] = s


@wp.kernel
def gps_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    geodetic_origin: wp.array(dtype=wp.vec3d),  # the model's origin: latitude, longitude, altitude, per world
    body_world: wp.array(dtype=int),
    body: int,  # the model body the sensor rides
    fix_type: int,
    time: wp.array(dtype=wp.float64),
    time_ns: wp.array(dtype=wp.int64),
    rate: wp.float64,
    start: wp.array(dtype=wp.int64),
    taken: wp.array(dtype=int),
    sample: wp.array(dtype=GpsSample),
):
    if not _due(time_ns[0], rate, start, taken):
        return
    origin = geodetic_origin[_world(body_world, body)]
    ref_lat = origin[0]
    ref_lon = origin[1]
    ref_alt = origin[2]
    p = wp.transform_get_translation(body_q[body])
    v = wp.spatial_top(body_qd[body])  # world linear vel
    # lat/lon/alt in float64: a float32 ``ref_lat (~47.6) + small offset`` would lose ~1e-5 deg,
    # about 1 m, which PX4 then quantizes to degE7, so the geodetic mapping must be double precision.
    # World -> geodetic uses the same axis map as everything else: north = +x, east = -y, down =
    # -z, the frame contract in ``nexus_uav._src.transform``. Position and velocity must share
    # it; when they disagreed, PX4's EKF reset in a loop and east guidance ran away, GH #61.
    metres_per_degree = wp.float64(111000.0)
    inv_lon_scale = wp.float64(1.0) / (metres_per_degree * wp.cos(wp.radians(ref_lat)))
    s = GpsSample()
    s.time = time[0]
    s.lat = ref_lat + wp.float64(p[0]) / metres_per_degree
    s.lon = ref_lon - wp.float64(p[1]) * inv_lon_scale
    s.alt = ref_alt + wp.float64(p[2])
    s.velocity = wp.vec3(v[0], -v[1], -v[2])  # north, east and down: world +y is west
    s.ground_speed = wp.sqrt(v[0] * v[0] + v[1] * v[1])
    s.fix_type = fix_type
    sample[0] = s


class ImuSensor(DeviceSensor):
    """Accelerometer, which reads specific force, and gyroscope, in the axes of the sensor's site:
    Warp kernel + Warp RNG.

    The mount is the sensor's site on the body it rides: its translation and its rotation in the body's
    axes, from the model. The accelerometer adds the lever arm ``alpha x r + omega x (omega x r)``, where
    ``r`` runs from the body's center of mass to the site, as Newton's ``SensorIMU`` does.

    The accelerometer finite-differences the body's velocity over the sim time since its last sample, so it
    reports the mean acceleration over the sample interval before, half an interval late, on purpose:
    PX4's estimator integrates delta velocity over each sample interval, and the interval's mean is that
    integral over its length. Newton's ``SensorIMU`` reads the solver's acceleration at the tick's end instead.

    Each accelerometer axis saturates at the full-scale range, as a real part does: a ground strike past it
    reads the range's edge with the strike's sign.

    The sensor reports no attitude: the vehicle's true state isn't a sensor reading.

    Args:
        model: Newton's model, which holds the site, the body's center of mass and gravity.
        site: The label of the sensor's site: the path of the prim that declares it.
        seed: The sensor's noise seed, which the builder derives from the run's seed and the prim.
        acc_noise: Standard deviation of the accelerometer's white noise, m/s^2.
        acc_range: The accelerometer's full-scale range, in standard gravity, 9.80665 m/s^2: each axis reads
            at most this either way.
        gyro_noise: Standard deviation of the gyroscope's white noise, rad/s.
        rate: How often the sensor gives a new sample, hertz: 0 gives one every tick, as does a rate faster
            than the tick rate.

    Raises:
        ValueError: The model has no site of that label, the rate is negative, or the range is zero or less;
            the message names the site.
    """

    name = "imu"  # the instance name, which the builder sets from the prim for a declared sensor

    def __init__(
        self,
        model,
        site: str,
        *,
        seed: int = 0,
        acc_noise: float = 0.02,
        acc_range: float = 16.0,
        gyro_noise: float = 0.02,
        rate: float = 0.0,
    ):
        super().__init__("imu", ImuSample, model, site, rate)
        if acc_range <= 0:
            raise ValueError(f"{site}: the accelerometer's range is {acc_range} g; author a range above 0")
        self.seed = int(seed)
        self.acc_noise = float(acc_noise)
        self.acc_range = float(acc_range)
        self.gyro_noise = float(gyro_noise)
        self._acc_limit = self.acc_range * STANDARD_GRAVITY  # the range in m/s^2, which the kernel clamps to
        self._prev_lin = wp.zeros(1, dtype=wp.vec3)
        self._prev_ang = wp.zeros(1, dtype=wp.vec3)
        self._prev_t = wp.zeros(1, dtype=wp.float64)  # the sim time of the last sample

    def sample_wp(self, state, t) -> None:
        """Launch the IMU kernel into the sample signal's buffer, with NO host readback, so it joins a
        captured region. The kernel counts its samples on the device, so the noise field varies per
        replay, and its first sample seeds the finite difference, so a captured region replays with the
        earlier velocities set.
        """
        wp.launch(
            imu_kernel,
            dim=1,
            inputs=(
                state.body_q,
                state.body_qd,
                self.model.gravity,
                self.model.body_world,
                self.model.body_com,
                self.model.shape_transform,
                self.site,
                self.body,
                self.seed,
                self.acc_noise,
                self.gyro_noise,
                self._acc_limit,
                self._prev_lin,
                self._prev_ang,
                self._prev_t,
                self.time.buffer,
                self.time_ns.buffer,
                self.rate,
                self._start,
                self._taken,
            ),
            outputs=(self.out.buffer,),
        )


class MagSensor(DeviceSensor):
    """Magnetometer at the origin of the body it rides: the model's magnetic field in the body's axes, with
    an offset and white noise.

    Args:
        model: Newton's model, which holds the site and the field.
        site: The label of the sensor's site: the path of the prim that declares it.
        seed: The sensor's noise seed.
        offset: A constant offset on each body axis, gauss.
        noise: Standard deviation of the white noise on each body axis, gauss. The default,
            0.02/0.02/0.03, is the bridge value, about ten times a real magnetometer's, and against PX4's
            strict per-sample World Magnetic Model (WMM) strength check it trips the "magnetic interference"
            check now and then, so the shipped vehicles author a lower sigma.
        rate: How often the sensor gives a new sample, hertz.

    Raises:
        ValueError: The model has no site of that label, the site sits off its body's origin, or the rate is
            negative; the message names the site.
    """

    name = "mag"

    def __init__(
        self, model, site: str, *, seed: int = 0, offset=(0.0, 0.0, 0.0), noise=(0.02, 0.02, 0.03), rate: float = 0.0
    ):
        super().__init__("mag", MagSample, model, site, rate)
        _at_body_origin(self, "magnetometer")
        self.seed = int(seed)
        self.offset = tuple(float(x) for x in offset)
        self.noise = tuple(float(x) for x in noise)
        self._offset = wp.vec3(*self.offset)
        self._sigma = wp.vec3(*self.noise)

    def sample_wp(self, state, t) -> None:
        wp.launch(
            mag_kernel,
            dim=1,
            inputs=(
                state.body_q,
                self.model.nexus.magnetic_field,
                self.model.body_world,
                self._offset,
                self.seed,
                self._sigma,
                self.body,
                self.time.buffer,
                self.time_ns.buffer,
                self.rate,
                self._start,
                self._taken,
            ),
            outputs=(self.out.buffer,),
        )


class BaroSensor(DeviceSensor):
    """Barometer at the origin of the body it rides: the model's pressure at sea level brought to the body's
    altitude through the standard atmosphere, with white noise, and the model's temperature.

    Args:
        model: Newton's model, which holds the site, the pressure at sea level and the temperature.
        site: The label of the sensor's site: the path of the prim that declares it.
        seed: The sensor's noise seed.
        noise: Standard deviation of the white noise on the pressure, hPa, and on the altitude, metres.
        rate: How often the sensor gives a new sample, hertz.

    Raises:
        ValueError: The model has no site of that label, the site sits off its body's origin, or the rate is
            negative; the message names the site.
    """

    name = "baro"

    def __init__(self, model, site: str, *, seed: int = 0, noise: float = 0.02, rate: float = 0.0):
        super().__init__("baro", BaroSample, model, site, rate)
        _at_body_origin(self, "barometer")
        self.seed = int(seed)
        self.noise = float(noise)

    def sample_wp(self, state, t) -> None:
        wp.launch(
            baro_kernel,
            dim=1,
            inputs=(
                state.body_q,
                self.model.nexus.sea_level_pressure,
                self.model.nexus.temperature,
                self.model.body_world,
                self.seed,
                self.noise,
                self.body,
                self.time.buffer,
                self.time_ns.buffer,
                self.rate,
                self._start,
                self._taken,
            ),
            outputs=(self.out.buffer,),
        )


class GpsSensor(DeviceSensor):
    """GPS receiver at the origin of the body it rides: the body's position and velocity from the model's
    geodetic origin, where the world's origin sits on Earth.

    Args:
        model: Newton's model, which holds the site and the origin.
        site: The label of the sensor's site: the path of the prim that declares it.
        seed: The sensor's noise seed, which the receiver doesn't draw on.
        fix_type: The fix the receiver reports, a MAVLink ``GPS_FIX_TYPE`` value: 3 is a 3D fix.
        rate: How often the sensor gives a new sample, hertz.

    Raises:
        ValueError: The model has no site of that label, the site sits off its body's origin, or the rate is
            negative; the message names the site.
    """

    name = "gps"

    def __init__(self, model, site: str, *, seed: int = 0, fix_type: int = 3, rate: float = 0.0):
        super().__init__("gps", GpsSample, model, site, rate)
        _at_body_origin(self, "GPS receiver")
        self.fix_type = int(fix_type)

    def sample_wp(self, state, t) -> None:
        wp.launch(
            gps_kernel,
            dim=1,
            inputs=(
                state.body_q,
                state.body_qd,
                self.model.nexus.geodetic_origin,
                self.model.body_world,
                self.body,
                self.fix_type,
                self.time.buffer,
                self.time_ns.buffer,
                self.rate,
                self._start,
                self._taken,
            ),
            outputs=(self.out.buffer,),
        )
