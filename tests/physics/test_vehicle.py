"""VehicleUsd loads a Universal Scene Description (USD) asset into a Newton ModelBuilder: the standalone
USD vehicle path.

Real end-to-end on the Warp CPU backend: author a minimal USD, a rigid body with a mass, build it
through VehicleUsd, finish the model, and assert the body + mass came through. Needs USD support,
usd-core + newton-usd-schemas, newton's importers deps minus open3d; skipped if pxr is missing.
"""

import pytest

pytest.importorskip("pxr")
pytest.importorskip("newton")

pytestmark = pytest.mark.usefixtures("warp_cpu")


def _author_min_usd(path: str) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Cube.Define(stage, "/Vehicle/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.5)
    stage.GetRootLayer().Save()


def test_usd_builder_loads_body_and_mass(tmp_path):
    import newton

    from nexus_sim._src.physics.vehicle import VehicleUsd

    usd = str(tmp_path / "mini.usda")
    _author_min_usd(usd)

    mb = newton.ModelBuilder()
    VehicleUsd({"usd_path": usd}).build(mb)
    model = mb.finalize()

    assert len(model.body_label) == 1
    assert abs(float(model.body_mass.numpy().sum()) - 1.5) < 1e-6


def test_usd_builder_spawns_frd_init_attitude(tmp_path):
    """The vehicle body's authored frame is Forward Right Down (FRD); VehicleUsd must place it with the
    180°-X FRD→world init attitude so it rests upright and the Inertial Measurement Unit (IMU)/mag read
    FRD, the frame PX4 Hardware In The Loop (HIL) expects. Placing it at identity instead leaves the FRD
    body inverted → PX4 reads an upside-down attitude and refuses to arm. Asserts the resting body
    orientation is 180° about X.
    """
    import newton
    import numpy as np

    from nexus_sim._src.physics.vehicle import VehicleUsd

    usd = str(tmp_path / "mini.usda")
    _author_min_usd(usd)

    mb = newton.ModelBuilder()
    VehicleUsd({"usd_path": usd}).build(mb)
    model = mb.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # 180° about X as an xyzw quaternion is [1, 0, 0, 0], up to sign.
    quat_xyzw = state.body_q.numpy()[0, 3:7]
    quat_xyzw *= np.sign(quat_xyzw[0]) or 1.0  # canonicalize sign
    np.testing.assert_allclose(quat_xyzw, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_usd_builder_spawn_att_override(tmp_path):
    """``cfg['spawn_att']`` overrides the default FRD init attitude, for example for a USD authored
    in a different frame: here identity, leaving the body unrotated.
    """
    import newton
    import numpy as np
    import warp as wp

    from nexus_sim._src.physics.vehicle import VehicleUsd

    usd = str(tmp_path / "mini.usda")
    _author_min_usd(usd)

    mb = newton.ModelBuilder()
    VehicleUsd({"usd_path": usd, "spawn_att": wp.quat_identity()}).build(mb)
    model = mb.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    quat_xyzw = state.body_q.numpy()[0, 3:7]
    quat_xyzw *= np.sign(quat_xyzw[3]) or 1.0
    np.testing.assert_allclose(quat_xyzw, [0.0, 0.0, 0.0, 1.0], atol=1e-5)


def test_usd_builder_fixed_base_raises(tmp_path):
    """A USD that pins the base to the world, with no floating base, must fail loudly, not ship a drone
    with zero degrees of freedom; a fixed base link is a common export quirk.
    """
    import newton
    from pxr import Usd, UsdGeom, UsdPhysics

    usd = str(tmp_path / "pinned.usda")
    stage = Usd.Stage.CreateNew(usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Cube.Define(stage, "/Vehicle/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.5)
    joint = UsdPhysics.FixedJoint.Define(stage, "/Vehicle/base_joint")  # world -> base, pins it
    joint.GetBody1Rel().SetTargets([body.GetPrim().GetPath()])
    stage.GetRootLayer().Save()

    from nexus_sim._src.physics.vehicle import VehicleUsd

    with pytest.raises(ValueError, match="floating base"):
        VehicleUsd({"usd_path": usd}).build(newton.ModelBuilder())


def test_usd_builder_missing_path_raises():
    import newton

    from nexus_sim._src.physics.vehicle import VehicleUsd

    with pytest.raises(FileNotFoundError):
        VehicleUsd({}).build(newton.ModelBuilder())


def test_usd_builder_nonexistent_file_raises(tmp_path):
    import newton

    from nexus_sim._src.physics.vehicle import VehicleUsd

    with pytest.raises(FileNotFoundError):
        VehicleUsd({"usd_path": str(tmp_path / "nope.usda")}).build(newton.ModelBuilder())
