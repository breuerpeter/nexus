"""Content-addressed asset resolver: fetch a remote asset to a verified local cache.

Hosting is content-addressed, the asset's sha256 is in its URL key, which gives three
things at once: CloudFront caches immutable URLs forever, with no invalidations; the hash IS
the version pin, because assets are inputs to a run, so a run references {url, sha256} and is
reproducible; and the same blobs dedup. The runtime path is pure stdlib, urllib plus hashlib,
with no Amazon Web Services (AWS) deps, so it stays light for the standalone runtime and works for a
public-read Content Delivery Network (CDN), anonymous GET, and ``file://`` URLs alike. The one
exception is an ``s3://`` URL, which has no CDN in front of it: boto3 fetches that, so the standard
credential chain applies: the environment, a profile, an instance profile or ``aws sso login``. The
``s3`` extra, ``nexus-uav[s3]``, installs boto3, and a plain install carries no AWS package.

Multi-file Universal Scene Description (USD) assets ship as a single self-contained ``.usdz``, one blob
that ``newton.add_usd`` / Isaac Lab ``UsdFileCfg`` open in place, so one sha256 pins the whole asset.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

# Hosted assets sit on a flat, content-addressed scheme under a catalog's base:
# ``<base>/assets/<kind>/<name>-<sha256>.<ext>``, where ``<kind>`` is the whole folder under
# ``assets/``: ``usd/vehicles`` and ``usd/scenes`` for the self-contained ``.usdz`` files and their
# ``.glb`` previews, ``usd/layers`` for the layers a composed vehicle entry stacks over its base,
# ``policies`` for an example's exported ``.pt`` policy. ``<name>`` is
# human-readable and the sha in the key makes objects immutable, cached forever. This is the single
# source of truth for the scheme; the base is the catalog's own, so where the blobs live is data a
# catalog carries and not a name in this package. The docs preview hook and
# ``scripts/assets/prepare_asset_upload.py`` derive their URLs here.


def hosted_url(base: str, kind: str, name: str, sha256: str, ext: str = "usdz") -> str:
    """URL of a content-addressed hosted asset under *base*. ``kind`` is the folder under
    ``assets/``: ``usd/vehicles``, ``usd/scenes``, ``usd/layers`` or ``policies``.

    The scheme of *base* decides how :func:`fetch` reads it: ``https://`` anonymously, ``file://``
    from disk, and ``s3://`` through boto3's credential chain. An entry that
    carries a full URL of its own never reaches here.
    """
    return f"{base.rstrip('/')}/assets/{kind}/{name}-{sha256}.{ext}"


def default_cache() -> Path:
    """Local content-addressed cache; override with ``$NEXUS_ASSET_CACHE``.

    In a source checkout downloads land in the repo's ``assets/cache/``, gitignored, alongside
    ``assets/local/``; an installed package with no repo tree falls back to ``~/.cache``.
    """
    env = os.environ.get("NEXUS_ASSET_CACHE")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3]
    if (repo / "pyproject.toml").exists():
        return repo / "assets" / "cache"
    return Path.home() / ".cache" / "nexus" / "assets"


def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_s3(url: str, dst: Path, timeout: float) -> None:
    """Download an ``s3://`` asset with boto3, through the standard AWS credential chain.

    boto3 comes from the ``s3`` extra, and only this function imports it, so a plain install, whose
    assets are an ordinary web fetch, never needs it.
    """
    try:
        import boto3
        import botocore.config
        import botocore.exceptions
    except ImportError as e:
        raise RuntimeError(f"fetching {url} needs boto3: install `nexus-uav[s3]`") from e
    parts = urllib.parse.urlparse(url)
    config = botocore.config.Config(connect_timeout=timeout, read_timeout=timeout)
    # A session of its own per fetch: boto3's default session would keep the credentials it found
    # first for the life of the process, past a change to the environment or the profile.
    client = boto3.session.Session().client("s3", config=config)
    try:
        client.download_file(parts.netloc, parts.path.lstrip("/"), str(dst))
    except botocore.exceptions.NoCredentialsError as e:
        raise RuntimeError(
            f"fetching {url} needs AWS credentials for that bucket: the environment, a profile through "
            "AWS_PROFILE, an instance profile, or `aws sso login`"
        ) from e
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise RuntimeError(f"fetching {url} failed: {e}") from e


def fetch(
    url: str,
    sha256: str,
    *,
    cache_dir: str | Path | None = None,
    filename: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Resolve a content-addressed asset to a local path: cache-hit, else download + verify.

    Args:
        url: the asset URL: an ``https://`` URL for anonymous GET, ``file://`` for local/tests, or
            ``s3://``, fetched by boto3 through the AWS credential chain; that needs
            ``nexus-uav[s3]``.
        sha256: expected content hash; a mismatch raises, for integrity plus the version pin.
        cache_dir: cache root; defaults to :func:`default_cache`.
        filename: cached filename; defaults to the URL basename.
        timeout: per-connection socket timeout in seconds; a stalled host raises rather than
            hanging the caller, for example import-time resolution in a training app.

    Returns the cached local path. Idempotent and offline after the first fetch.
    """
    cache = Path(cache_dir) if cache_dir else default_cache()
    name = filename or os.path.basename(urllib.parse.urlparse(url).path) or "asset"
    dst = cache / sha256 / name
    if dst.exists() and sha256_file(dst) == sha256:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dst.parent)
    tmp = Path(tmp_name)
    try:
        if urllib.parse.urlparse(url).scheme == "s3":
            os.close(tmp_fd)
            _download_s3(url, tmp, timeout)
        else:
            with os.fdopen(tmp_fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as resp:
                shutil.copyfileobj(resp, out)
        got = sha256_file(tmp)
        if got != sha256:
            raise ValueError(f"asset hash mismatch for {url}: expected {sha256}, got {got}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def resolve(ref: dict, **kw) -> Path:
    """Resolve a catalog asset ref ``{"url": ..., "sha256": ...}`` to a local path."""
    return fetch(ref["url"], ref["sha256"], filename=ref.get("filename"), **kw)


def stack_layers(
    layers: list[tuple[Path, str]], asset: Path, asset_sha: str, cache_dir: str | Path | None = None
) -> Path:
    """A root file in the asset cache that stacks *layers* over *asset*, a vehicle or a scene, so each
    reader opens the composed asset by one path.

    USD composes the files; the root only lists them, strongest first, and carries the asset's stage
    metadata: its default prim, units, up axis and layer data, such as a scene's render recipe, which a
    stage reads from its root file alone. The root sits under ``layered/<hash>/``, the hash over the
    asset's sha256 and each layer's, so one stack writes one file.

    Args:
        layers: Each layer's local path and sha256, strongest first: a launch's override layer, then a
            composed entry's layers in the entry's order.
        asset: The local path of the asset under them.
        asset_sha: The asset's sha256.
        cache_dir: The cache root; defaults to :func:`default_cache`.

    Returns:
        The root file's path.
    """
    from pxr import Sdf

    key = hashlib.sha256("\n".join([asset_sha, *(sha for _, sha in layers)]).encode()).hexdigest()
    out = Path(cache_dir or default_cache()) / "layered" / key / f"{asset.stem}.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = [*(str(path) for path, _ in layers), str(asset)]
    source = Sdf.Layer.FindOrOpen(str(asset))  # held, so its spec outlives the read
    for info in source.pseudoRoot.ListInfoKeys():
        if not info.startswith("subLayer"):  # its own sublayers and their offsets stay the asset's
            root.pseudoRoot.SetInfo(info, source.pseudoRoot.GetInfo(info))
    root.Export(str(out))
    return out


def stage_hash(path: str | Path) -> str:
    """The sha256 of the stage at *path* with its layer stack flattened into one layer: the content a
    composed vehicle entry resolves to, the same wherever its files sit in a cache.

    A composed entry has no file of its own to hash, so its preview ``.glb`` takes this hash as its key.
    The flattened layer stack is deterministic text; a full flatten numbers its prototypes in no fixed
    order, so it isn't.

    Args:
        path: The stage's root file, such as the one :func:`stack_layers` writes.

    Returns:
        The hex digest.
    """
    from pxr import Usd, UsdUtils

    flat = UsdUtils.FlattenLayerStack(Usd.Stage.Open(str(path)))
    flat.comment = ""  # the flattener names the root file's path in both
    flat.documentation = ""
    return hashlib.sha256(flat.ExportToString().encode()).hexdigest()
