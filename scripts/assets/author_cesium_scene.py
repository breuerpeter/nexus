"""Author cesium.usd: the Cesium for Omniverse streamed-globe scene, as a shippable Universal Scene Description (USD).

This bakes the STATIC Cesium prim structure that ``RtxFrame._add_cesium_world`` used to build in
code each run: a georeference with a placeholder origin, which the runtime sets from the resolved geodetic
origin, the ``CesiumData``, the Google Photorealistic 3D Tiles ``Tileset``, and the ion server
prim WITHOUT a token. The runtime then just loads this scene and injects the two runtime-only bits:
the ``CESIUM_ION_TOKEN``, a secret never baked into a public asset, and the georef lat/lon.

Kit-only: the Cesium USD schema needs registering at Kit boot through PXR_PLUGINPATH_NAME, which the
Kit image sets, so run it through the command-line tool on a Kit-capable host:

  uv run nexus script scripts/assets/author_cesium_scene.py --out assets/local/cesium.usd

Writes ``cesium.usd`` + a self-contained ``cesium.usdz`` sibling, then round-trips the package to
confirm the ``cesium:*`` schema resolves rather than giving "Empty typeName" and prints the .usdz sha256.
"""

import argparse
import hashlib
import os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True, help="output cesium.usd path (a .usdz sibling is also written)")
p.add_argument("--asset-id", type=int, default=2275207, help="ion asset (2275207 = Google Photorealistic 3D Tiles)")
args = p.parse_args()

# Kit extension modules, imported where they become available; E402 is the price of a
# top-to-bottom script. The cesium.usd schema module only exists once the code below enables the
# extension, so this order is load-bearing.
import omni.kit.app  # noqa: E402

ext_dir = os.environ.get("NEXUS_CESIUM_EXTS", "/cesium-exts")
em = omni.kit.app.get_app().get_extension_manager()
em.add_path(ext_dir)
em.set_extension_enabled_immediate("cesium.omniverse", True)  # pulls cesium.usd.plugins + the schema module

from cesium.usd.plugins.CesiumUsdSchemas import Data as CesiumData  # noqa: E402
from cesium.usd.plugins.CesiumUsdSchemas import Georeference as CesiumGeoreference  # noqa: E402
from cesium.usd.plugins.CesiumUsdSchemas import IonServer as CesiumIonServer  # noqa: E402
from cesium.usd.plugins.CesiumUsdSchemas import Tileset as CesiumTileset  # noqa: E402
from cesium.usd.plugins.CesiumUsdSchemas import Tokens as CesiumTokens  # noqa: E402
from pxr import Usd, UsdGeom, UsdUtils  # noqa: E402

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    out.unlink()
stage = Usd.Stage.CreateNew(str(out))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
# METERS, explicitly: unauthored metersPerUnit falls back to USD's 0.01, centimeters, and the
# scene opens as the renderer's root stage. Cesium places the whole globe in stage units, so a
# cm-scaled stage renders every vehicle pose write at 1/100 of its true motion: the GH #32
# "world doesn't move" bug, which rotation disguised, being unit-free and still working.
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Georeference: placeholder origin; the runtime overwrites lat/lon from the resolved geodetic
# origin, registry default or a launch override. Height stays 0; the ground-align probe converges it.
geo = CesiumGeoreference.Define(stage, "/CesiumGeoreference")
geo.GetGeoreferenceOriginLatitudeAttr().Set(0.0)
geo.GetGeoreferenceOriginLongitudeAttr().Set(0.0)
geo.GetGeoreferenceOriginHeightAttr().Set(0.0)

data = CesiumData.Define(stage, "/Cesium")

tileset = CesiumTileset.Define(stage, "/Cesium/WorldTerrain")
tileset.GetGeoreferenceBindingRel().AddTarget("/CesiumGeoreference")
# Ground-level First-Person View (FPV) tuning, mirroring the old in-code authoring: fog culling off,
# forbid holes, screen-space error 8 as the quality/realtime balance, load the neighborhood ahead of
# time, big tile cache.
tileset.GetEnableFogCullingAttr().Set(False)
tileset.GetForbidHolesAttr().Set(True)
tileset.GetMaximumScreenSpaceErrorAttr().Set(8.0)
tileset.GetPreloadAncestorsAttr().Set(True)
tileset.GetPreloadSiblingsAttr().Set(True)
tileset.GetMaximumCachedBytesAttr().Set(4 * 1024**3)
tileset.GetSourceTypeAttr().Set(CesiumTokens.ion)
tileset.GetIonAssetIdAttr().Set(args.asset_id)
tileset.GetIonServerBindingRel().AddTarget("/CesiumServers/IonOfficial")
# NO IonAccessToken: the runtime injects CESIUM_ION_TOKEN, a secret never baked into a public asset.

server = CesiumIonServer.Define(stage, "/CesiumServers/IonOfficial")
server.GetIonServerUrlAttr().Set("https://ion.cesium.com/")
server.GetIonServerApiUrlAttr().Set("https://api.cesium.com/")
server.GetIonServerApplicationIdAttr().Set(413)
data.GetSelectedIonServerRel().SetTargets(["/CesiumServers/IonOfficial"])

# The scene USD is the single authority on its own lighting, stage 5: the terrain tiles carry
# only aerial-imagery albedo, so light them from the asset. A color dome, self-contained; the
# runtime handler can lay a captured High Dynamic Range Image (HDRI) texture over it on the SESSION
# layer where the container ships one, plus a warm sun. Values match the old in-handler authoring,
# with -20%/-23% trims: the tutorial grade read overexposed.
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

dome = UsdLux.DomeLight.Define(stage, "/CesiumSky")
dome.CreateIntensityAttr(800.0)
dome.CreateColorAttr(Gf.Vec3f(0.55, 0.68, 0.92))
UsdGeom.Xformable(dome.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))  # horizon level, Z-up
sun = UsdLux.DistantLight.Define(stage, "/CesiumSun")
sun.CreateIntensityAttr(2300.0)
sun.CreateAngleAttr(0.5)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.9))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 35.0))

# The photoreal render recipe rides the asset as stage render settings, customLayerData:
# Kit auto-applies them when the scene USD opens as the root stage, with no applier code.
# The Cesium SF tutorial's grade, distances in meters: distance+height fog, which also masks
# the tile-streaming horizon, Iray tonemap, where burnHighlights 1.0 = highlights COMPRESSED and
# lower clips to white, Fast Fourier Transform (FFT) bloom, dome-in-reflections + indirect diffuse GI.
root_layer = stage.GetRootLayer()
cld = dict(root_layer.customLayerData)
cld["renderSettings"] = {
    "rtx:fog:enabled": True,
    "rtx:fog:fogColor": Gf.Vec3f(0.60392, 0.75294, 0.9098),
    "rtx:fog:fogColorIntensity": 0.25,
    "rtx:fog:fogDistanceDensity": 0.0005,
    "rtx:fog:fogStartDist": 1000.0,
    "rtx:fog:fogEndDist": 5000.0,
    "rtx:fog:fogStartHeight": 60.0,
    "rtx:fog:fogHeightDensity": 0.2,
    "rtx:fog:fogHeightFalloff": 0.0,
    "rtx:post:tonemap:op": 7,  # Iray
    "rtx:post:tonemap:irayReinhard:crushBlacks": 0.0,
    "rtx:post:tonemap:irayReinhard:burnHighlights": 1.0,
    "rtx:post:tonemap:irayReinhard:saturation": 1.1,
    "rtx:post:lensFlares:enabled": True,  # FFT bloom
    "rtx:post:lensFlares:flareScale": 0.25,
    "rtx:post:lensFlares:cutoffPoint": Gf.Vec3f(5.0, 5.0, 5.0),
    "rtx:directLighting:domeLight:enabledInReflections": True,
    "rtx:indirectDiffuse:enabled": True,
    "rtx:sceneDb:ambientLightColor": Gf.Vec3f(0.1, 0.1, 0.1),
}
root_layer.customLayerData = cld

stage.SetDefaultPrim(data.GetPrim())
stage.GetRootLayer().Save()

usdz = out.with_suffix(".usdz")
if usdz.exists():
    usdz.unlink()
ok = UsdUtils.CreateNewUsdzPackage(str(out), str(usdz))
h = hashlib.sha256()
with open(usdz, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)

reopened = Usd.Stage.Open(str(usdz))
ts = reopened.GetPrimAtPath("/Cesium/WorldTerrain")
print("=== cesium scene authored ===")
print("usd :", out)
print("usdz:", usdz, "created:", ok)
tn = ts.GetTypeName()
print("Tileset typeName:", tn, "(expect CesiumTilesetPrim, a resolved type, not empty)")
print("IonAssetId      :", CesiumTileset(ts).GetIonAssetIdAttr().Get())
print("Georef present  :", reopened.GetPrimAtPath("/CesiumGeoreference").IsValid())
print("Sky present     :", reopened.GetPrimAtPath("/CesiumSky").IsValid())
print("renderSettings  :", "renderSettings" in reopened.GetRootLayer().customLayerData)
print("SHA256          :", h.hexdigest())
# A resolved Cesium schema gives a non-empty typeName; "Empty typeName" means the schema plugin
# wasn't registered through PXR_PLUGINPATH_NAME, and the asset would be inert.
rc = 0 if (ok and tn and "Cesium" in str(tn)) else 2

raise SystemExit(rc)
