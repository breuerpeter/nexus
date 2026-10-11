"""The plant's outputs: a `PoseTwist` per body and a struct per joint width, which the physics' `state` stage
writes from Newton's arrays, #275. Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.build.assembly import build_scenario
from nexus_sim._src.core.schema import PoseTwist
from nexus_sim._src.physics import NewtonPhysics, VehicleUsd
from nexus_sim._src.physics.outputs import joint_struct
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")


def _physics(tmp_path, **author):
    """The physics of the fixture quad authored with `author`, built on the CPU device."""
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    usd = VehicleUsd({"usd_path": str(quad.author(tmp_path / "quad.usda", **author))})
    return NewtonPhysics(vehicle_usd=usd, cfg=cfg)


def _state_stage(physics):
    """The physics' `state` stage, its outputs given the buffers a run's wiring gives them."""
    stage = next(st for st in physics.stages() if st.name == "state")
    for signal in stage.writes:
        signal.buffer = signal._allocate()
    return stage


def test_a_joint_struct_holds_q_and_qd_of_the_joints_widths_and_one_type_serves_each_width():
    """A joint struct holds `q` and `qd` as vectors of the joint's widths, and one type serves each width.

    Given the widths of a free joint, 7 and 6, when `joint_struct` runs twice for them, then both are one type
    whose numpy dtype holds `q (7,)` and `qd (6,)`, float32, and the struct of a revolute joint, 1 and 1, is
    another type.
    """
    free, again, revolute = joint_struct(7, 6), joint_struct(7, 6), joint_struct(1, 1)
    dtype = np.dtype(free.numpy_dtype())
    fields = {name: (str(dtype[name].base), dtype[name].shape) for name in dtype.names}

    assert (free is again, free is revolute, fields) == (
        True,
        False,
        {"q": ("float32", (7,)), "qd": ("float32", (6,))},
    )


def test_the_plant_writes_one_pose_twist_per_body_and_one_struct_per_joint_named_by_their_labels(tmp_path):
    """The plant writes one `PoseTwist` signal per body, named `body/<label>`, and one struct signal per joint
    with a coordinate, named `joints/<label>`, each keyed by the model's label.

    Given the fixture quad with a gimbal joint, when its physics builds, then its `state` stage writes
    `body/body`, `body/gimbal` and `body/rotor_0` to `body/rotor_3` as `PoseTwist`, and `joints/joint_1`,
    the free base joint, `joints/gimbal_joint` and `joints/rotor_0_joint` to `joints/rotor_3_joint` as
    structs of `q` and `qd`, and `outputs.by_label` finds the base body's signal by its prim's path.
    """
    with wp.ScopedDevice("cpu"):
        physics = _physics(tmp_path, gimbal=True)
    state = next(st for st in physics.stages() if st.name == "state")
    written = {s.name: s for s in state.writes}
    bodies = sorted(name for name, s in written.items() if s.type is PoseTwist)
    joints = {name: np.dtype(s.type.numpy_dtype())["q"].shape for name, s in written.items() if s.type is not PoseTwist}

    assert (bodies, joints, physics.outputs.by_label["/Vehicle/body"] is written["body/body"]) == (
        ["body/body", "body/gimbal", "body/rotor_0", "body/rotor_1", "body/rotor_2", "body/rotor_3"],
        {
            "joints/joint_1": (7,),
            "joints/gimbal_joint": (1,),
            "joints/rotor_0_joint": (1,),
            "joints/rotor_1_joint": (1,),
            "joints/rotor_2_joint": (1,),
            "joints/rotor_3_joint": (1,),
        },
        True,
    )


def test_the_state_stage_writes_each_bodys_pose_and_twist_and_each_joints_coordinates_from_the_state(tmp_path):
    """The `state` stage writes each body's pose and twist and each joint's coordinates and velocities from
    the physics state it takes.

    Given the fixture quad settled on the ground, when the `state` stage runs over the settled state, then the
    base body's signal holds the state's position, orientation and twist of body 0, to the bit, and the free
    joint's signal holds the first 7 coordinates and 6 velocities of `joint_q` and `joint_qd`.
    """
    with wp.ScopedDevice("cpu"):
        physics = _physics(tmp_path)
        state = physics.reset()
        stage = _state_stage(physics)
        stage.run(type("Tick", (), {"state": state, "dt": 0.004})())
        written = {s.name: s for s in stage.writes}
        body, free = written["body/body"].read()[0], written["joints/joint_1"].read()[0]
        q, qd = state.body_q.numpy()[0], state.body_qd.numpy()[0]
        jq, jqd = state.joint_q.numpy(), state.joint_qd.numpy()

    assert (
        body["position"].tolist(),
        body["orientation"].tolist(),
        body["linear_velocity"].tolist(),
        body["angular_velocity"].tolist(),
        free["q"].tolist(),
        free["qd"].tolist(),
    ) == (q[:3].tolist(), q[3:].tolist(), qd[:3].tolist(), qd[3:].tolist(), jq[:7].tolist(), jqd[:6].tolist())
