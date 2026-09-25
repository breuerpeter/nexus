"""``scripts/assets/prepare_asset_upload.py`` prints the URL a run would fetch, on the shared hosted
layout. A scene needs no glb conversion, so the script runs to the end with no bpy environment.
"""

import hashlib
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "prepare_asset_upload", ROOT / "scripts" / "assets" / "prepare_asset_upload.py"
)
prepare_asset_upload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_asset_upload)


def test_scene_upload_url_keeps_the_usd_layout(tmp_path, capsys):
    """The vehicle and scene USD and GLB URLs do not change: a scene uploads to
    `<base>/assets/usd/scenes/<name>-<sha>.usdz`.
    """
    data = b"a self-contained scene"
    usd = tmp_path / "site.usdz"
    usd.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()

    prepare_asset_upload.main(["--scene", str(usd), "--base", "https://cdn.example/public"])

    assert f"https://cdn.example/public/assets/usd/scenes/site-{sha}.usdz" in capsys.readouterr().out
