r"""Author the ``empty`` scene Universal Scene Description (USD): the default flat-ground world, now a real sky-only asset.

Since the scene USD became the single authority on its own lighting, in stage 5 of the runtime
unification, which dropped the runtime's ``_add_sky``, even "no scene" ships a USD: one root
``Xform`` holding the plain lit sky, dome and sun, and no geometry. The ground stays the physics
ground plane; there is deliberately nothing to see below the horizon, exactly as before.

    uv run python scripts/assets/author_empty_scene.py assets/local/empty.usdz
then prepare it for upload, which prints the sha, the upload key, and the registry snippet:
    uv run python scripts/assets/prepare_asset_upload.py --scene assets/local/empty.usdz
"""

from __future__ import annotations

import sys


def author(out_path: str) -> str:
    import tempfile

    from pxr import Sdf, Usd, UsdGeom, UsdUtils
    from scene_root import author_sky

    out_path = str(out_path)
    if not out_path.endswith(".usdz"):
        raise ValueError(f"scene assets are ONE self-contained .usdz, got {out_path!r}")
    tmp_usd = tempfile.mktemp(suffix=".usdc")
    stage = Usd.Stage.CreateNew(tmp_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    author_sky(stage, "/World")
    stage.GetRootLayer().Save()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(tmp_usd), out_path):
        raise RuntimeError(f"usdz packaging failed for {tmp_usd}")
    return out_path


if __name__ == "__main__":
    print(author(sys.argv[1] if len(sys.argv) > 1 else "assets/local/empty.usdz"))
