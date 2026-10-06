"""Content-addressed asset resolver: fetch a remote asset to a verified local cache.

Hosting is content-addressed, the asset's sha256 is in its URL key, which gives three
things at once: CloudFront caches immutable URLs forever, with no invalidations; the hash IS
the version pin, because assets are inputs to a run, so a run references {url, sha256} and is
reproducible; and the same blobs dedup. The runtime path is pure stdlib, urllib plus hashlib,
with no Amazon Web Services (AWS) deps, so it stays light for the standalone runtime and works for a
public-read Content Delivery Network (CDN), anonymous GET, and ``file://`` URLs alike. The one
exception is an ``s3://`` URL, which has no CDN in front of it: that shells out to the ``aws``
command-line tool so the standard credential chain applies, runner instance profile / access keys /
``aws sso login``, without adding an AWS SDK dependency.

Multi-file Universal Scene Description (USD) assets ship as a single self-contained ``.usdz``, one blob
that ``newton.add_usd`` / Isaac Lab ``UsdFileCfg`` open in place, so one sha256 pins the whole asset.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

# Hosted assets sit on a flat, content-addressed scheme under a registry's base:
# ``<base>/assets/<kind>/<name>-<sha256>.<ext>``, where ``<kind>`` is the whole folder under
# ``assets/``: ``usd/vehicles`` and ``usd/scenes`` for the self-contained ``.usdz`` files and their
# ``.glb`` previews, ``policies`` for an example's exported ``.pt`` policy. ``<name>`` is
# human-readable and the sha in the key makes objects immutable, cached forever. This is the single
# source of truth for the scheme; the base is the registry's own, so where the blobs live is data a
# catalog carries and not a name in this package. The docs preview hook and
# ``scripts/assets/prepare_asset_upload.py`` derive their URLs here.


def hosted_url(base: str, kind: str, name: str, sha256: str, ext: str = "usdz") -> str:
    """URL of a content-addressed hosted asset under *base*. ``kind`` is the folder under
    ``assets/``: ``usd/vehicles``, ``usd/scenes`` or ``policies``.

    The scheme of *base* decides how :func:`fetch` reads it: ``https://`` anonymously, ``file://``
    from disk, and ``s3://`` through the ``aws`` command-line tool's credential chain. An entry that
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


def _download_s3(url: str, dst: Path) -> None:
    """Download an ``s3://`` asset via the ``aws`` command-line tool's standard credential chain.

    A subprocess instead of an SDK keeps the resolver AWS-dependency-free for everyone whose assets
    are an ordinary web fetch. Only this path needs the tool.
    """
    hint = (
        f"fetching {url} needs the aws CLI with credentials for that bucket "
        "(runner instance profile, access keys, or `aws sso login`)"
    )
    try:
        proc = subprocess.run(
            ["aws", "s3", "cp", url, str(dst), "--no-progress"], check=False, capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"{hint}; the `aws` executable was not found") from e
    if proc.returncode != 0:
        raise RuntimeError(f"{hint}; `aws s3 cp` failed: {proc.stderr.strip() or proc.stdout.strip()}")


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
            ``s3://``, fetched via the aws command-line tool's credential chain.
        sha256: expected content hash; a mismatch raises, for integrity plus the version pin.
        cache_dir: cache root; defaults to :func:`default_cache`.
        filename: cached filename; defaults to the URL basename.
        timeout: per-connection socket timeout in seconds; a stalled host raises rather than
            hanging the caller, for example import-time resolution in a training app. Not applied to
            ``s3://`` fetches; the aws command-line tool brings its own retry/timeout behavior.

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
            _download_s3(url, tmp)
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
    """Resolve a registry asset ref ``{"url": ..., "sha256": ...}`` to a local path."""
    return fetch(ref["url"], ref["sha256"], filename=ref.get("filename"), **kw)
