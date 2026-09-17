"""The *one* canonical transform between the world frame and North East Down (NED) / Forward Right
Down (FRD), shared by every sensor.

This module is the single home for the frame conventions the bridge once
scattered across ``mavlink_interface``. ``world_to_body`` = R(q)^-1·v unifies the world<->body
*rotation*, which the Inertial Measurement Unit (IMU), gyro and mag use, and the world<->NED
*axis map* below is the one definition the Warp sensor kernels mirror.

The frame contract: the Newton world is right-handed with z UP: X = North,
Y = West, Z = Up. So world to NED is n = +x, e = -y, d = -z, a proper
rotation satisfying n x e = d.

The bridge this code came from encoded that map three mutually inconsistent ways:
Global Positioning System (GPS) velocity said Y=West, ``ve=-vy``, while the magnetometer and the
GPS lat/lon mapping said Y=East, which is a reflection, not a rotation. Only the velocity map
was right. The IMU chain is what pins the proper frame, because aligning the velocity map
to the mirrored pair instead diverges on every axis, so GH #61 corrected the GPS position and the
magnetometer to match it.

Newton world frame: Forward Left Up (FLU) / Z-up. Body frame: FRD.
"""

from __future__ import annotations

import math

import warp as wp

GRAVITY_WORLD = wp.vec3(0.0, 0.0, -9.81)


def quat_xyzw(body_q_row) -> wp.quat:
    """Warp quat [x, y, z, w] from a body_q row [px,py,pz, qx,qy,qz,qw]."""
    return wp.quat(
        float(body_q_row[3]),
        float(body_q_row[4]),
        float(body_q_row[5]),
        float(body_q_row[6]),
    )


def quat_from_xyzw(q4) -> wp.quat:
    """Warp quat from a neutral ``[x, y, z, w]`` 4-vector, the Newton-native orientation convention."""
    return wp.quat(float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3]))


def world_to_body(q: wp.quat, v_world) -> wp.vec3:
    """The canonical world->body rotation: R(q)^-1 · v_world. Shared by all sensors."""
    return wp.quat_rotate_inv(q, wp.vec3(float(v_world[0]), float(v_world[1]), float(v_world[2])))


def body_to_world(q: wp.quat, v_body) -> wp.vec3:
    """The canonical body->world rotation: R(q) · v_body."""
    return wp.quat_rotate(q, wp.vec3(float(v_body[0]), float(v_body[1]), float(v_body[2])))


def lever_arm_accel(q: wp.quat, omega_world, alpha_world, r_com_to_mount_body) -> wp.vec3:
    """World-frame acceleration of a body-fixed mount point relative to the Center Of Mass (COM):
    ``α×r + ω×(ω×r)``, where ``r`` is the COM→mount offset given in the *body* frame,
    so a sensor offset from the COM reads the lever-arm term. Returns a zero vector
    when the mount sits at the COM. The ``body_qd`` linear velocity is world-frame at the COM.
    """
    w = wp.vec3(float(omega_world[0]), float(omega_world[1]), float(omega_world[2]))
    a = wp.vec3(float(alpha_world[0]), float(alpha_world[1]), float(alpha_world[2]))
    r = body_to_world(q, r_com_to_mount_body)
    return wp.cross(a, r) + wp.cross(w, wp.cross(w, r))


def quat_wxyz(q: wp.quat) -> tuple[float, float, float, float]:
    """Warp xyzw quat -> MAVLink wire order [w, x, y, z]."""
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def world_vel_to_ned(v_world) -> tuple[float, float, float]:
    """Newton world linear velocity -> NED [m/s]: ``(vx, -vy, -vz)``, because world +y is West."""
    return (float(v_world[0]), -float(v_world[1]), -float(v_world[2]))


def mag_ned_to_world(mag_ned) -> wp.vec3:
    """NED magnetic field -> Newton world frame: ``(n, -e, -d)``, with X=N, Y=W, Z=Up."""
    return wp.vec3(float(mag_ned[0]), -float(mag_ned[1]), -float(mag_ned[2]))


def gps_from_local(ref_lat: float, ref_lon: float, ref_alt: float, pos) -> tuple[float, float, float]:
    """Local Newton-world position [m] -> (lat_deg, lon_deg, alt_m AMSL).

    Flat-earth small-angle mapping, no projection lib. World +y is *west*, so it
    subtracts from the longitude.
    """
    lat = ref_lat + float(pos[0]) / 111000.0
    lon = ref_lon - float(pos[1]) / (111000.0 * math.cos(math.radians(ref_lat)))
    alt = ref_alt + float(pos[2])
    return lat, lon, alt
