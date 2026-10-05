"""``sensor:modality`` on a camera prim picks the sensor class, and the two modalities number
their MediaMTX streams separately.

The routing itself needs Kit, so the test here covers the decision it makes: the authored
attribute a vehicle Universal Scene Description (USD) carries, and the stream names that fall out of
it. The rule that matters is the silent default: a vehicle authored before the attribute existed must
keep its exact behaviour, same class and same cam1/cam2 order.
"""

import pytest

pytest.importorskip("pxr")

from nexus._src.vehicle.sensors.rtx_stage import prim_modality


def _stage(tmp_path):
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "vehicle.usda"))
    body = UsdGeom.Xform.Define(stage, "/Vehicle/body")
    for name, modality in (("FpvCam", None), ("Lr1Cam", None), ("IrCam", "ir"), ("EoCam", "eo")):
        cam = UsdGeom.Camera.Define(stage, body.GetPath().AppendChild(name)).GetPrim()
        if modality is not None:
            cam.CreateAttribute("sensor:modality", Sdf.ValueTypeNames.Token, custom=True).Set(modality)
    odd = UsdGeom.Camera.Define(stage, body.GetPath().AppendChild("OddCam")).GetPrim()
    odd.CreateAttribute("sensor:modality", Sdf.ValueTypeNames.Token, custom=True).Set("uv")
    return stage


def test_authored_modality_is_read_and_absent_means_eo(tmp_path):
    stage = _stage(tmp_path)
    at = lambda name: prim_modality(stage.GetPrimAtPath(f"/Vehicle/body/{name}"))  # noqa: E731
    assert at("IrCam") == "ir"
    assert at("EoCam") == "eo"
    assert at("FpvCam") == "eo", "a prim authored before the attribute existed keeps EO behaviour"


def test_an_unknown_modality_falls_back_to_eo(tmp_path):
    stage = _stage(tmp_path)
    assert prim_modality(stage.GetPrimAtPath("/Vehicle/body/OddCam")) == "eo"


# The stream numbering itself, an added IR camera never shifting cam1/cam2, sits inside the Kit-bound
# renderer factory, so ffprobe verifies it in flight rather than here; see the PR test plan.
