r"""Author the ``slalom`` scene Universal Scene Description (USD): three cost-only obstacle pillars
for the sampling Model Predictive Control (MPC) demo.

The pillars are vertical capsules at the mid-point of each leg of the demo's zig-zag, from the
start point through the three waypoints in ``obstacle_slalom.py``. Their semantics are fully
USD-authored: ``UsdPhysics.CollisionAPI`` puts them into the Newton model, since
the generic ``add_usd`` path loads any scene USD, and ``physics:collisionEnabled = false`` makes
them cost-only: present for the Signed Distance Field (SDF) avoidance cost + viz, never producing
contact dynamics. No scene-specific loader code anywhere.

    uv run python scripts/assets/author_slalom_scene.py /tmp/slalom.usdz
then prepare it for upload, which prints the sha, the upload key, and the registry snippet:
    uv run python scripts/assets/prepare_asset_upload.py --scene /tmp/slalom.usdz
upload the file to its printed target and paste the snippet into registry.yaml under scenes.slalom.
"""

from __future__ import annotations

import itertools
import sys

# The demo geometry, which must match the example's waypoints: pillars sit at the mid-point of each leg.
SPAWN = (0.0, 0.0, 2.0)
WAYPOINTS = [(2.5, 1.2, 2.0), (5.0, -1.0, 2.0), (7.5, 0.7, 2.0)]
PILLAR_RADIUS = 0.4
PILLAR_HALF_HEIGHT = 2.5


def pillar_positions() -> list[tuple[float, float, float]]:
    legs = [SPAWN, *WAYPOINTS]
    return [(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 2.0) for a, b in itertools.pairwise(legs)]


def author(out_path: str) -> str:
    import tempfile

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    out_path = str(out_path)
    if not out_path.endswith(".usdz"):
        raise ValueError(f"scene assets are ONE self-contained .usdz, got {out_path!r}")
    from scene_root import author_sky

    tmp_usd = tempfile.mktemp(suffix=".usdc")
    stage = Usd.Stage.CreateNew(tmp_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())  # the one root prim, the runtime's session start-translate target
    author_sky(stage, "/World")  # the scene USD is the single authority on its own lighting, stage 5
    for i, pos in enumerate(pillar_positions()):
        cap = UsdGeom.Capsule.Define(stage, f"/World/pillar_{i}")
        cap.CreateRadiusAttr(float(PILLAR_RADIUS))
        cap.CreateHeightAttr(float(2.0 * PILLAR_HALF_HEIGHT))
        cap.CreateAxisAttr("Z")  # vertical pillar
        cap.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.15, 0.15)])
        UsdGeom.XformCommonAPI(cap).SetTranslate(Gf.Vec3d(*[float(c) for c in pos]))
        api = UsdPhysics.CollisionAPI.Apply(cap.GetPrim())  # -> the Newton model, via generic add_usd
        api.CreateCollisionEnabledAttr(False)  # cost-only: no contact dynamics, USD-authored
    stage.GetRootLayer().Save()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(tmp_usd), out_path):
        raise RuntimeError(f"usdz packaging failed for {tmp_usd}")
    return out_path


if __name__ == "__main__":
    print(author(sys.argv[1] if len(sys.argv) > 1 else "/tmp/slalom.usdz"))
