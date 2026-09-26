"""The model's gravity is the site's, one value, whatever the vehicle Universal Scene Description (USD)
file authors on its ``PhysicsScene``. Real build on the Warp CPU backend; skipped without pxr or newton.
"""

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("newton")

pytestmark = pytest.mark.usefixtures("warp_cpu")


def _author_vehicle_with_gravity(path: str, magnitude: float) -> None:
    """A one-body vehicle whose ``PhysicsScene`` authors its own gravity magnitude."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/PhysicsScene").CreateGravityMagnitudeAttr(magnitude)
    body = UsdGeom.Cube.Define(stage, "/Vehicle/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.5)
    stage.GetRootLayer().Save()


def test_model_gravity_is_the_sites_whatever_the_vehicle_usd_authors(tmp_path):
    """A run's model gravity is the site's 9.81 whatever the vehicle USD's `PhysicsScene` authors."""
    from nexus._src.physics.builders.usd import USDBuilder
    from nexus._src.physics.physics import NewtonPhysics

    usd = str(tmp_path / "grav5.usda")
    _author_vehicle_with_gravity(usd, 5.0)
    cfg = {"physics": {"dt": 0.004, "solver": "semi_implicit", "contacts": False}}

    physics = NewtonPhysics(vehicle_builder=USDBuilder({"usd_path": usd}, None), cfg=cfg)

    np.testing.assert_allclose(physics.model.gravity.numpy()[0], (0.0, 0.0, -9.81), atol=1e-6)
