"""The unified **PropellerModel**, ``propeller.propeller_force``: the body-frame propeller force from a
motor speed Ω + the rotor's in-plane inflow ``(vx, vy)``: the airflow-aware ``kf·Ω²`` − forward-flight
loss + H-force. These pin the invariants the unified summed-wrench actuator relies on, model-free, on Warp
CPU with no asset: static thrust = ``kf·Ω²`` = ``ct·rpm²``; the forward-flight loss reduces axial thrust by
``aero_h·|v_hor|²``; the H-force opposes the in-plane inflow scaling with Ω; and at the shipped
``aero_h = aero_hforce = 0`` the model is exactly the quasi-static ``(0, 0, kf·Ω²)``.

PYTEST_DONT_REWRITE: this module defines a ``@wp.kernel``; pytest's assertion rewriting feeds warp
1.14's codegen an exec()-compiled module, which it refuses as "directly evaluating Warp code defined
as a string." The marker opts the module out, since kernel asserts here don't need introspection.
"""

import math

import numpy as np
import warp as wp

from nexus._src.vehicle.actuators import RPM_PER_RADS, propeller_force

wp.set_device("cpu")

CT = 3.463e-6  # N/min²
KF = CT * RPM_PER_RADS**2  # thrust = kf·Ω² == ct·rpm², with Ω in rad/s
OMEGA = 250.0  # rad/s
THRUST0 = CT * (OMEGA * RPM_PER_RADS) ** 2  # static thrust, scalar [N]


@wp.kernel
def _prop(
    omega: float, vx: float, vy: float, kf: float, aero_h: float, aero_hforce: float, out: wp.array(dtype=wp.vec3)
):
    out[0] = propeller_force(omega, vx, vy, kf, aero_h, aero_hforce)  # (h_x, h_y, thrust_mag)


def _run(omega, vx, vy, *, aero_h=0.0, aero_hforce=0.0):
    out = wp.zeros(1, dtype=wp.vec3)
    wp.launch(_prop, dim=1, inputs=(float(omega), float(vx), float(vy), KF, float(aero_h), float(aero_hforce)), outputs=(out,))  # fmt: skip
    return out.numpy()[0]  # (h_x, h_y, thrust_mag)


def test_static_thrust_is_kf_omega_squared():
    """At zero inflow the propeller returns (0, 0, kf·Ω²) = (0, 0, ct·rpm²), the quasi-static map."""
    h_x, h_y, thrust = _run(OMEGA, 0.0, 0.0)
    assert math.isclose(thrust, THRUST0, rel_tol=1e-5), (thrust, THRUST0)
    assert h_x == 0.0 and h_y == 0.0


def test_airflow_off_is_quasi_static():
    """With aero_h = aero_hforce = 0, the shipped astro-max, any inflow leaves (0, 0, kf·Ω²) unchanged."""
    h_x, h_y, thrust = _run(OMEGA, 12.0, -7.0, aero_h=0.0, aero_hforce=0.0)
    assert math.isclose(thrust, THRUST0, rel_tol=1e-6)
    assert h_x == 0.0 and h_y == 0.0


def test_forward_flight_reduces_thrust():
    """In-plane inflow reduces axial thrust by aero_h·|v_hor|², the forward-flight thrust loss."""
    vx, vy, aero_h = 8.0, 6.0, 0.1
    _h_x, _h_y, thrust = _run(OMEGA, vx, vy, aero_h=aero_h)
    expected = THRUST0 - aero_h * (vx * vx + vy * vy)
    assert thrust < THRUST0
    assert math.isclose(thrust, expected, rel_tol=1e-4), (thrust, expected)


def test_forward_flight_loss_clamps_at_zero():
    """The loss never drives thrust negative: an inflow of 1e3 zeroes it, not below."""
    _h_x, _h_y, thrust = _run(OMEGA, 1e3, 0.0, aero_h=1.0)
    assert thrust == 0.0


def test_hforce_opposes_inplane_velocity():
    """The H-force opposes the in-plane inflow and equals −aero_hforce·|Ω|·v, component-wise."""
    vx, vy, k = 5.0, -3.0, 1e-4
    h_x, h_y, thrust = _run(OMEGA, vx, vy, aero_hforce=k)
    assert math.isclose(h_x, -k * OMEGA * vx, rel_tol=1e-4), h_x
    assert math.isclose(h_y, -k * OMEGA * vy, rel_tol=1e-4), h_y
    assert math.isclose(thrust, THRUST0, rel_tol=1e-5)  # axial thrust unchanged, since aero_h = 0


def test_hforce_scales_with_rotor_speed():
    """The H-force scales with |Ω|, so doubling |Ω| doubles |H|, and uses |Ω|, so it's sign-independent."""
    k, vx = 1e-4, 4.0
    h1 = _run(100.0, vx, 0.0, aero_hforce=k)[0]
    h2 = _run(200.0, vx, 0.0, aero_hforce=k)[0]
    h_neg = _run(-100.0, vx, 0.0, aero_hforce=k)[0]
    assert math.isclose(h2, 2.0 * h1, rel_tol=1e-5)
    assert math.isclose(h_neg, h1, rel_tol=1e-5)  # |Ω| ⇒ spin direction doesn't flip the H-force
    assert np.sign(h1) == -np.sign(vx)  # opposes +x inflow
