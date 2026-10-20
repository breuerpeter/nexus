"""The rotor chain of many copies of one vehicle, the plant the RL task trains on, steps each copy as a run steps
one vehicle: each copy takes its own commands, and its rotors' torque and forces are those the core's rotor chain
gives the vehicle alone.

Real builds on the Warp CPU backend of the fixture quad of ``tests/vehicle/quad.py``. Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import newton
import warp as wp

from nexus_uav._src.physics.motors import Motors
from nexus_uav._src.physics.vehicle import VehicleUsd
from nexus_uav._src.scene.world import GRAVITY
from nexus_uav._src.vehicle.commands import Escs
from nexus_uav._src.vehicle.forces import Propellers
from nexus_uav._src.vehicle.rotors import rotor_of
from nexus_uav.examples._lib.rotor_chain import RotorChain, vehicle_rotors
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")

DT = 0.004  # [s]
SPEED = 300.0  # every rotor's speed [rad/s]
PLACES = ((0.0, 0.0, 2.0), (3.0, 0.0, 2.0))  # where each copy starts [m]
COMMANDS = ((0.2, 0.4, 0.6, 0.8), (0.9, 0.7, 0.5, 0.3))  # each copy's rotor commands


def _model(path, places):
    """A model with one copy of the vehicle at each of ``places``, every rotor at 300 rad/s: the model, its state
    and its control.
    """
    builder = newton.ModelBuilder()
    for place in places:
        VehicleUsd({"usd_path": str(path), "spawn_pos": place}).build(builder)
    builder.gravity = -GRAVITY
    model = builder.finalize()
    joints = [rotor.joint for rotor in VehicleUsd({"usd_path": str(path)}).rotors()]
    dofs = vehicle_rotors(model, joints)[0].ravel()  # every copy's rotors
    state = model.state()
    qd = state.joint_qd.numpy()
    qd[dofs] = SPEED
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, model.control()


def _alone(path, place, command):
    """The core's rotor chain on the vehicle alone at ``place`` under ``command``: its rotors' torque and their
    bodies' wrenches.
    """
    rotors = VehicleUsd({"usd_path": str(path)}).rotors()
    model, state, control = _model(path, [place])
    found = [rotor_of(model, rotor.body) for rotor in rotors]  # each rotor's dof, coord, body and airframe
    dofs, bodies, base = [f[0] for f in found], [f[2] for f in found], found[0][3]
    values = {key: [rotor.propeller[key] for rotor in rotors] for key in ("ct", "cd", "aero_h", "aero_hforce")}
    Escs(
        rotor_vel_dofs=wp.array([dofs], dtype=wp.int32),
        channels=[rotor.channel for rotor in rotors],
        ct=values["ct"],
        cd=values["cd"],
        rpm_max=[rotor.rpm_max for rotor in rotors],
    ).write(wp.array([command], dtype=float), control)
    Propellers(
        rotor_vel_dofs=wp.array(dofs, dtype=wp.int32),
        rotor_bodies=wp.array(bodies, dtype=wp.int32),
        rotor_bases=wp.array([base] * len(found), dtype=wp.int32),
        **values,
    ).add(state)
    Motors(model).step(state, control, DT)
    return control.joint_f.numpy()[dofs], state.body_f.numpy()[bodies]


def test_each_copy_flies_its_rotor_chain_as_the_core_flies_the_vehicle_alone(tmp_path):
    """Each copy of the vehicle flies its rotor chain as the core flies the vehicle alone.

    Given two copies of the quad 3 m apart, every rotor at 300 rad/s, each copy under its own commands, when the
    rotor chain steps them once, then each copy's rotor torques and rotor body wrenches equal those the core's
    command element, propellers and motors give the vehicle alone under its commands.
    """
    path = tmp_path / "quad.usda"
    quad.author(path)
    model, state, control = _model(path, PLACES)
    chain = RotorChain(model, VehicleUsd({"usd_path": str(path)}).rotors())

    chain.step(state, control, wp.array(COMMANDS, dtype=float), DT)

    for copy, (place, command) in enumerate(zip(PLACES, COMMANDS, strict=True)):
        torque, wrench = _alone(path, place, command)
        np.testing.assert_allclose(control.joint_f.numpy()[chain.rotor_dofs[copy]], torque, rtol=1e-6)
        np.testing.assert_allclose(state.body_f.numpy()[chain.rotor_bodies[copy]], wrench, rtol=1e-6, atol=1e-9)


def test_a_rotor_joint_the_model_does_not_hold_fails_naming_each_joints_count(tmp_path):
    """A declared rotor joint the model doesn't hold fails the search, naming how often the model holds each joint.

    Given a model with two copies of the quad, when the search looks for its four rotor joints and a fifth no copy
    has, then it raises and names the fifth joint's path below the root prim with a count of 0.
    """
    path = tmp_path / "quad.usda"
    quad.author(path)
    model, _state, _control = _model(path, PLACES)
    joints = [*(rotor.joint for rotor in VehicleUsd({"usd_path": str(path)}).rotors()), f"{quad.ROOT}/no_such_joint"]

    with pytest.raises(ValueError, match=r"'/no_such_joint': 0"):
        vehicle_rotors(model, joints)
