"""Physical-parameter recovery, system-ID, by differentiable simulation: recover an unknown airframe **mass**
from an open-loop trajectory.

Self-contained example: the articulated astro-max, the plant every run flies, climbs under one fixed command to
every rotor, its Newton motors spinning the rotor joints and its propellers lifting it, under
``SolverSemiImplicit``; a heavier airframe climbs less. Starting from a wrong mass guess, gradient descent on the
airframe's mass, which the rollout writes into the model on the tape, minimizes the trajectory
Mean Squared Error (MSE) against a reference rollout flown at the true mass, recovering the mass to <5%.

The companion ``gain_tuning.py`` is the other differentiable-design gate, controller auto-tuning.

Run:

    uv run -m nexus_uav.examples mass_recovery   # asserts the recovered mass

Runs on Compute Unified Device Architecture (CUDA) when present, where each gradient replays from one captured
graph, else on the CPU backend, much slower.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import nexus_uav as nx
from nexus_uav.examples._lib import RunClock, dump_stats
from nexus_uav.examples._lib.articulated import ArticulatedRollout, CapturedGradient

GRAVITY = 9.81
LIFT = 1.3  # the fixed command lifts this many times the vehicle's nominal weight


@wp.kernel
def set_airframe_mass(
    mass: wp.array(dtype=float),
    body: int,
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
):
    """Write the airframe's mass, the recovered leaf, into the model the solver reads."""
    body_mass[body] = mass[0]
    body_inv_mass[body] = 1.0 / mass[0]


@wp.kernel
def traj_mse(
    body_q: wp.array(dtype=wp.transform),
    body: int,
    ref: wp.array(dtype=wp.vec3),
    tick: int,
    cost: wp.array(dtype=float),
):
    p = wp.transform_get_translation(body_q[body])
    d = p - ref[tick]
    wp.atomic_add(cost, 0, wp.dot(d, d))


def recover_mass(
    *,
    vehicle: str = "astro_max_base",
    true_mass_scale: float = 1.4,
    init_guess_scale: float = 0.7,
    n_steps: int = 200,
    dt: float = 0.004,
    iters: int = 120,
    lr: float = 0.05,
):
    """Known-parameter recovery: recover the airframe's mass from an open-loop trajectory.

    One fixed command spins every rotor to the speed that lifts the vehicle's nominal weight times the lift factor; a
    heavier airframe climbs less. Starting from a wrong mass guess, descend the gradient of the trajectory MSE
    against a reference rollout flown at the true mass. Returns the convergence history and each rotor's speed at
    the end of the last rollout.
    """
    from nexus_uav._src.build.launch import resolve_vehicle_usd
    from nexus_uav._src.config import LaunchConfig
    from nexus_uav._src.vehicle.rotors import RPM_PER_RADS

    vehicle_usd, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle(vehicle).set_scene("empty"))
    rollout = ArticulatedRollout(vehicle_usd, ticks=n_steps, dt=dt, start=(0.0, 0.0, 0.5))
    model, base = rollout.model, rollout.base
    nominal_mass = float(model.body_mass.numpy()[base])
    true_mass = nominal_mass * true_mass_scale

    rotors = vehicle_usd.rotors()  # in channel order, each with its own thrust map
    kf = np.array([rotor.propeller["ct"] for rotor in rotors]) * RPM_PER_RADS**2
    omega_max = np.array([rotor.rpm_max for rotor in rotors]) / RPM_PER_RADS
    speed = np.sqrt(LIFT * rollout.mass * GRAVITY / len(rotors) / kf)  # rad/s, each rotor's
    command = wp.array([(speed / omega_max).tolist()], dtype=float)

    def forward(mass: wp.array, cost: wp.array | None = None, ref: wp.array | None = None, record=None):
        wp.launch(set_airframe_mass, dim=1, inputs=(mass, base), outputs=(model.body_mass, model.body_inv_mass))
        for t in range(n_steps):
            rollout.tick(t, command)
            state = rollout.state_at(t + 1)
            if cost is not None:
                wp.launch(traj_mse, dim=1, inputs=(state.body_q, base, ref, t), outputs=(cost,))
            if record is not None:
                record.append(state.body_q.numpy()[base, :3].copy())

    # reference trajectory at the true mass
    rec: list = []
    rollout.reset()
    forward(wp.array([true_mass], dtype=float), record=rec)
    ref = wp.array(np.asarray(rec), dtype=wp.vec3)

    # recover: descend the airframe's mass from a wrong guess
    mass = wp.array([nominal_mass * init_guess_scale], dtype=float, requires_grad=True)
    cost = wp.zeros(1, dtype=float, requires_grad=True)

    def taped():
        cost.zero_()
        forward(mass, cost=cost, ref=ref)

    gradient = CapturedGradient(taped, mass, cost)
    opt = wp.optim.Adam([mass], lr=lr * nominal_mass)
    history = []
    for _ in range(iters):
        rollout.reset()
        loss, grad = gradient()
        opt.step([grad])
        mass.assign(np.clip(mass.numpy(), 0.1 * nominal_mass, 10.0 * nominal_mass))
        history.append({"mass": float(mass.numpy()[0]), "loss": loss})
    return {
        "true_mass": float(true_mass),
        "recovered_mass": history[-1]["mass"],
        "history": history,
        "rotor_speeds": rollout.rotor_speeds(n_steps).tolist(),
    }


def main():
    # The run's time, net of the kernels Warp compiles into a cold cache, is what CI's time gate reads.
    with RunClock() as clock:
        # The system-ID math gate, on the articulated plant: CUDA when present, else the CPU backend.
        if wp.is_cuda_available():
            wp.set_device("cuda:0")
        mr = recover_mass(true_mass_scale=1.4, init_guess_scale=0.7, n_steps=200, iters=150, lr=0.05)
        err = abs(mr["recovered_mass"] - mr["true_mass"]) / mr["true_mass"]
        nx.logger.info(
            f"mass recovery: true={mr['true_mass']:.3f} recovered={mr['recovered_mass']:.3f} ({err * 100:.2f}%)"
        )
        # Evaluation artifacts first, before the gate can raise, since a failed run must still leave its
        # stats for diagnosis; no flight, so the stats JSON only.
        dump_stats(
            "mass_recovery",
            {
                "mass_rel_err": round(err, 4),
                "true_mass_kg": round(mr["true_mass"], 4),
                "recovered_mass_kg": round(mr["recovered_mass"], 4),
                "rotor_speed_min_rad_per_sec": round(min(mr["rotor_speeds"]), 1),
                **clock.stats(),
            },
        )
        assert err < 0.05, "mass not recovered"
        nx.logger.info("OK: mass recovered from an open-loop trajectory (differentiable system-ID)")


if __name__ == "__main__":
    main()
