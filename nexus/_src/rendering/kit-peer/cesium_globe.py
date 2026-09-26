"""The Cesium-globe scene: a live-streamed geolocated world, Google Photorealistic 3D Tiles via Cesium
for Omniverse, anchored at the scene's geodetic origin so the First Person View (FPV) camera flies
over the real place the Global Positioning System (GPS) says it's at.

Claimed by content: a scene Universal Scene Description (USD) file carrying a ``CesiumTilesetPrim``.
The scene USD, authored by ``scripts/assets/author_cesium_scene.py``, ships the tileset structure,
which is the georeference plus CesiumData plus the ion tileset plus the culling, quality and cache
tuning; the sky, dome and sun prims; and the photoreal render recipe,
``customLayerData.renderSettings``, which Kit applies when the peer opens the scene as the root
stage, where its absolute ``/Cesium*`` paths compose natively. This handler injects the two
runtime-only bits, the ion token the host sends in the setup message and the resolved georef of lat,
lon and alt, both on the session layer, and runs the live machinery no USD can carry: per-camera
tile-selection viewports and the streaming drain before the flight. The alt is data, the registry
scene's ``geodetic_origin.alt`` or ``--geo lat,lon,alt``: the WGS84 ellipsoidal height of the
surface, applied as the georeference origin height so the street sits at local z=0, with no runtime
ground probing. Tile geometry streams into Fabric, usdrt, and never exists as stage prims.
"""

from __future__ import annotations

import os
import time

# High Dynamic Range Image (HDRI) sky for the streamed globe: a real captured environment map, bundled in the Isaac image.
_STINSON_HDR = (
    "/isaac-sim/extscache/omni.usd.libs-1.0.3+6312fa25.lx64.r.cp312/bin/usd/hdx/resources/textures/StinsonBeach.hdr"
)


def _log(msg: str) -> None:
    print(f"[kit-peer] CesiumGlobe: {msg}", flush=True)


class CesiumGlobe:
    keeps_default_viewport = True  # tile selection reuses the boot viewport

    def __init__(self, token: str | None):
        self._token = (token or "").strip()  # the host's CESIUM_ION_TOKEN, from the setup message
        self._sel: dict = {}  # camera prim path -> its tile-selection camera's transform op
        self._active = False  # compose succeeded → on_ready runs the drain

    @staticmethod
    def matches(usd_path: str) -> bool:
        """A scene USD declaring a Cesium tileset. Needs the Cesium USD schema registered, through
        PXR_PLUGINPATH_NAME at Kit boot, for the ``CesiumTilesetPrim`` typeName to resolve;
        without the schema it reads as a plain scene and returns False.
        """
        from pxr import Usd

        try:
            s = Usd.Stage.Open(usd_path)
            return s is not None and any(p.GetTypeName() == "CesiumTilesetPrim" for p in s.Traverse())
        except Exception:
            return False

    def compose(self, stage, usd_path: str, georef: dict | None, cameras: list[str]) -> None:
        """Inject the runtime-only bits, token and georef, into the already-open scene stage and stand
        up one tile-selection viewport per sensor camera. A missing prerequisite or any failure warns
        and keeps the asset's own sky: the flight goes on, only the tiles are missing.

        Config: the ion token, required; extensions folder from ``NEXUS_CESIUM_EXTS``,
        default ``/cesium-exts``. The georef injection comes before enabling the tile engine: Cesium
        computes ``cesium:ecefToUsdTransform`` once, when it first reads the georeference, and a
        later override doesn't retrigger it.
        """
        token = self._token
        ext_dir = os.environ.get("NEXUS_CESIUM_EXTS", "/cesium-exts")
        checks = (
            (os.path.isdir(ext_dir), f"extensions not at {ext_dir!r} (rebuild the Kit image or set NEXUS_CESIUM_EXTS)"),
            (token, "no CESIUM_ION_TOKEN set"),
            (georef, "no geodetic origin (scene geodetic_origin / gps.init)"),
        )
        problems = [msg for ok, msg in checks if not ok]
        if problems:
            _log(f"unavailable ({'; '.join(problems)}): no tiles (the asset sky remains)")
            return
        try:
            import omni.kit.app

            em = omni.kit.app.get_app().get_extension_manager()
            em.add_path(ext_dir)

            # Raw-attr Set: Kit registers the Cesium schema at boot through PXR_PLUGINPATH_NAME, so
            # the attrs resolve without the extension loaded yet.
            gprim = stage.GetPrimAtPath("/CesiumGeoreference")
            gprim.GetAttribute("cesium:georeferenceOrigin:latitude").Set(float(georef["lat"]))
            gprim.GetAttribute("cesium:georeferenceOrigin:longitude").Set(float(georef["lon"]))
            # The surface's WGS84 ellipsoidal height is data, the registry geodetic_origin.alt or
            # --geo lat,lon,alt, applied directly so the street sits at local z=0. Missing alt →
            # 0 with a warning: the terrain sits at its raw ellipsoidal offset; author the value.
            alt = georef.get("alt")
            if alt is None:
                _log(
                    "no geodetic_origin.alt: georeference height 0 (author the surface's WGS84 "
                    "ellipsoidal height in the registry, or --geo lat,lon,alt)"
                )
            gprim.GetAttribute("cesium:georeferenceOrigin:height").Set(float(alt or 0.0))

            # Inject the ion token, a secret that never goes in the asset, before enabling the tile
            # engine, the same read-once rule as the georef: enabling cesium.omniverse processes the
            # stage at once and the tileset session captures the token it finds then; a later Set
            # doesn't retrigger it. The token-after-enable order loaded every tileset with an empty
            # token: 401 on the ion endpoint and zero tiles streamed, the GH #32 "world doesn't
            # move" recurrences.
            server_prim = stage.GetPrimAtPath("/CesiumServers/IonOfficial")
            server_prim.GetAttribute("cesium:projectDefaultIonAccessToken").Set(token)
            for prim in stage.Traverse():
                if prim.GetTypeName() == "CesiumTilesetPrim":
                    prim.GetAttribute("cesium:ionAccessToken").Set(token)

            em.set_extension_enabled_immediate("cesium.omniverse", True)  # tile engine: reads the origin plus token

            # Cesium's per-frame tick collects cameras from viewport window instances; the headless
            # boot has none, so with only the sensors' offscreen render products no viewport reaches
            # tile selection and nothing ever loads. One tile-selection camera plus viewport per
            # sensor camera: selection must cover every camera's frustum, since a nadir camera views
            # different tiles than the forward one. Selection cameras are free root prims, since the
            # sensors' flown pose is Fabric-only; follow() refreshes their USD pose from the flown
            # pose. Each copies its sensor's intrinsics widened as a guard band that loads tiles
            # ahead. The first viewport reuses the boot's default 'Viewport' window, extra cameras
            # get small windows; Screen Space Error (SSE) tile selection is resolution-independent,
            # measured, so 128x72 selects the same tiles for close to zero render cost. All windows stay
            # visible and updating: Cesium enumerates visible viewport windows each frame.
            import pxr.Usd as Usd
            from omni.kit.viewport.utility import create_viewport_window, get_active_viewport
            from pxr import UsdGeom, UsdLux

            self._sel = {}
            for i, cam_path in enumerate(cameras):
                src = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
                focal = float(src.GetFocalLengthAttr().Get() or 12.0)
                h_ap = float(src.GetHorizontalApertureAttr().Get() or 36.0)
                v_ap = float(src.GetVerticalApertureAttr().Get() or h_ap * 9 / 16)
                sel_path = f"/CesiumTileCam{i or ''}"
                tile_cam = UsdGeom.Camera.Define(stage, sel_path)
                tile_cam.GetFocalLengthAttr().Set(focal)
                # Selection Field Of View (FOV) guard band: +35% apertures. With +15%, fast
                # perspective changes outran the load-ahead margin and opened tile gaps at the frame edge.
                tile_cam.GetHorizontalApertureAttr().Set(h_ap * 1.35)
                tile_cam.GetVerticalApertureAttr().Set(v_ap * 1.35)
                tile_cam.GetClippingRangeAttr().Set(src.GetClippingRangeAttr().Get() or (0.05, 1.0e6))
                xf = UsdGeom.Xformable(tile_cam.GetPrim())
                xf.ClearXformOpOrder()
                op = xf.AddTransformOp()
                # Seed at the sensor's start pose: selection must point at the right place from the
                # first drain frame; identity = origin looking down = nothing selected.
                op.Set(
                    UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                )
                if i == 0:
                    vp = get_active_viewport()
                    vp.camera_path = sel_path
                    try:
                        vp.resolution = (128, 72)  # selection is resolution-independent; render cheap
                    except Exception:
                        pass
                else:
                    win = create_viewport_window(f"CesiumTiles{i}", width=128, height=72)
                    win.viewport_api.camera_path = sel_path
                self._sel[cam_path] = op
            if cameras:
                _log(f"tile selection -> {len(cameras)} viewport(s), one per camera")

            # The asset ships the sky, a color dome plus warm sun; where the image carries a real
            # captured HDRI, lay it over the dome on the session layer: photoreal tiles deserve
            # Image Based Lighting (IBL), not flat blue, but a baked image path never belongs in
            # the asset.
            dome = UsdLux.DomeLight(stage.GetPrimAtPath("/CesiumSky"))
            if dome and os.path.exists(_STINSON_HDR):
                from pxr import Sdf

                dome.GetTextureFileAttr().Set(Sdf.AssetPath(_STINSON_HDR))
                dome.CreateTextureFormatAttr().Set("latlong")
            self._active = True
            _log(
                f"streaming @ ({georef['lat']}, {georef['lon']}) [ion:{usd_path.rsplit('/', 1)[-1]}] "
                "(tiles stream over the drain before the flight)"
            )
        except Exception as exc:
            _log(f"compose failed ({exc!r}): no tiles (the asset sky remains)")

    def follow(self, poses: dict) -> None:
        """Level Of Detail (LOD) follow: refresh every selection camera's USD pose from its sensor's
        flown pose every render; a ~4 Hz decimation let fast perspective changes outrun tile
        selection and open gaps. ``poses`` maps a sensor prim path to its world matrix, a
        ``Gf.Matrix4d``.
        """
        for path, op in self._sel.items():
            m = poses.get(path)
            if m is not None:
                op.Set(m)

    def on_ready(self, app) -> None:
        """Google tiles refine over many frames. Poll Cesium's own statistics until loading
        quiesces, bounded. No flight throttle: capping cesium:mainThreadLoadingTimeLimit, tried at
        3 ms, starves refinement after viewpoint changes.
        """
        if not self._active:
            return
        try:
            from cesium.omniverse.bindings import acquire_cesium_omniverse_interface

            cesium_iface = acquire_cesium_omniverse_interface()
        except Exception:
            cesium_iface = None
        quiet = 0
        loading = None
        elapsed = 0.0
        t_stream0 = time.time()
        for frames_pumped in range(3000):  # noqa: B007  # the log line that follows reports the counter
            app.update()
            loading = None
            if cesium_iface is not None:
                try:
                    st = cesium_iface.get_render_statistics()
                    loading = int(st.tiles_loading_main) + int(st.tiles_loading_worker)
                except Exception:
                    cesium_iface = None
            quiet = quiet + 1 if loading == 0 else 0
            elapsed = time.time() - t_stream0
            # at least 20 s, since refinement requests trickle in, then 60 consecutive idle frames
            if elapsed > 20.0 and (quiet >= 60 or (cesium_iface is None and elapsed > 60.0)):
                break
        loaded = None
        if cesium_iface is not None:
            try:
                st = cesium_iface.get_render_statistics()
                loaded = int(getattr(st, "tiles_loaded", 0)) or int(getattr(st, "tiles_rendered", 0))
            except Exception:
                pass
        if loaded == 0:
            # `loading == 0` also holds when nothing was ever requested: a drain over a tileset that
            # 401'd, with an empty or invalid ion token, streamed nothing and flew the camera through
            # unrefined geometry, the GH #32 "frozen world" recurrences.
            _log(
                "ZERO tiles streamed: the tileset did not load (check this log for ion 401s / "
                "CESIUM_ION_TOKEN validity). The flight goes on over an EMPTY globe."
            )
        else:
            _log(
                f"tiles streamed ({frames_pumped + 1} drain frames, {elapsed:.0f}s, "
                f"loading={loading}, loaded={loaded if loaded is not None else 'n/a'})"
            )
