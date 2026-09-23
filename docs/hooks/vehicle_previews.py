"""mkdocs hook: inject <model-viewer> asset previews from the vehicle registry.

Replaces `<!-- model-preview: <name> -->` markers in any page with an interactive glTF
preview of the vehicle whose Universal Scene Description (USD) filename stem is `<name>`,
for example `astro_max_fpv`. The glb is the content-addressed `.glb` sibling of the
vehicle's USD on CloudFront, under the flat scheme `usd/vehicles/<name>-<sha>.glb`, so the
URL derives from `registry.yaml`: adding a vehicle there makes its preview available with
no page edits.

Preview-before-publish: if a local glb exists under `assets/local/<name>-<sha>.glb`, where
`scripts/assets/prepare_asset_upload.py` writes it, the hook serves that copy instead, so
`mkdocs serve` shows a freshly generated asset before the upload. The hook injects that
local copy into the built site as `assets/models/<basename>` at build time and writes
nothing into `docs/`. In CI, which has no `assets/local/`, previews fall back to the
CloudFront URL.

`mkdocs.yml` and `docs/javascripts/` wire in the viewer itself, the vendored model-viewer
and Draco decoder; this hook only emits the per-vehicle element and serves local glbs.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from mkdocs.structure.files import File
from mkdocs.utils import get_relative_url

from nexus._src.assets.resolver import hosted_url

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "nexus/_src/config/registry.yaml"
_LOCAL_ASSETS = _REPO / "assets/local"  # where prepare_asset_upload.py drops preview glbs
_SITE_MODELS = "assets/models"  # where the built site serves a local glb
_MARKER = re.compile(r"(?P<indent>[ \t]*)<!--\s*model-preview:\s*(?P<name>[\w.-]+)\s*-->")


def _glb_targets() -> dict[str, tuple[str, str]]:
    """Map asset name -> (cloudfront_glb_url, glb_basename) for every registry vehicle.

    A compact entry stores ``{name, sha256}``; the glb is the content-addressed sibling of the
    vehicle's USD, with the same key and the ``.glb`` suffix, derived against the registry's own base
    through the shared ``hosted_url`` scheme.
    """
    data = yaml.safe_load(_REGISTRY.read_text())
    base = (data.get("assets") or {}).get("base")
    targets: dict[str, tuple[str, str]] = {}
    for vehicle in data.get("vehicles", []):
        usd = vehicle.get("usd") or {}
        name, sha = usd.get("name"), usd.get("sha256")
        if not (name and sha and base):  # an entry with a URL of its own carries no derivable preview
            continue
        targets.setdefault(name, (hosted_url(base, "vehicles", name, sha, "glb"), f"{name}-{sha}.glb"))
    return targets


def _local_glb(basename: str) -> Path:
    return _LOCAL_ASSETS / basename


def on_files(files, config):
    """Serve any locally present preview glb, from `assets/local/*.glb`, at `assets/models/`."""
    for _stem, (_url, basename) in _glb_targets().items():
        local = _local_glb(basename)
        if local.exists():
            files.append(File.generated(config, f"{_SITE_MODELS}/{basename}", abs_src_path=str(local)))
    return files


def _embed(src: str, name: str) -> str:
    return (
        '<div class="model-preview">\n'
        "<model-viewer\n"
        f'  src="{src}"\n'
        f'  alt="Interactive 3D model of {name}"\n'
        '  camera-controls auto-rotate loading="eager"\n'
        '  environment-image="neutral" exposure="1.1"\n'
        '  shadow-intensity="1" shadow-softness="0.8"\n'
        '  style="width:100%;height:480px;border-radius:8px;'
        "background:radial-gradient(circle at 50% 35%,#f5f6f8 0%,#d6dadf 65%,#c2c6cc 100%);\">\n"
        "</model-viewer>\n</div>"
    )


def on_page_markdown(markdown: str, *, page, config, files) -> str:
    if not _MARKER.search(markdown):
        return markdown
    targets = _glb_targets()

    def replace(match: re.Match) -> str:
        indent = match.group("indent")
        name = match.group("name")
        target = targets.get(name)
        if target is None:
            emitted = f"<!-- model-preview: no registry vehicle with USD stem '{name}' -->"
        else:
            glb_url, glb_basename = target
            # Prefer a local copy, for preview-before-publish or offline use; else CloudFront.
            if _local_glb(glb_basename).exists():
                src = get_relative_url(f"{_SITE_MODELS}/{glb_basename}", page.file.url)
            else:
                src = glb_url
            emitted = _embed(src, name)
        # Preserve the marker's leading indentation so a marker nested inside a content tab
        # such as `=== "Base"` stays within the tab's indented block.
        if indent:
            emitted = "\n".join(indent + line if line else line for line in emitted.splitlines())
        return emitted

    return _MARKER.sub(replace, markdown)
