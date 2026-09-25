"""Resolver cache + hash-verify: stdlib only, no network/Amazon Web Services (AWS). It uses file:// URLs,
and exercises the ``s3://`` path against a stub ``aws`` command-line tool on PATH.
"""

import hashlib
import os

import pytest

from nexus._src.assets import fetch, sha256_file
from nexus._src.assets.resolver import hosted_url


def _make(tmp_path, data=b"astro_max usd bytes"):
    src = tmp_path / "src.usdz"
    src.write_bytes(data)
    return src, hashlib.sha256(data).hexdigest()


def test_fetch_downloads_and_verifies(tmp_path):
    src, digest = _make(tmp_path)
    cache = tmp_path / "cache"
    out = fetch(src.as_uri(), digest, cache_dir=cache)
    assert out.exists() and sha256_file(out) == digest
    assert str(cache) in str(out) and digest in str(out)  # content-addressed path


def test_cache_hit_no_redownload(tmp_path):
    src, digest = _make(tmp_path)
    cache = tmp_path / "cache"
    out1 = fetch(src.as_uri(), digest, cache_dir=cache)
    src.unlink()  # remove origin; a cache hit must not need it
    out2 = fetch(src.as_uri(), digest, cache_dir=cache)
    assert out1 == out2 and out2.exists()


def test_hash_mismatch_raises(tmp_path):
    src, _ = _make(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch(src.as_uri(), "0" * 64, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.usdz"))  # nothing left in cache on failure


def test_hosted_url_schemes():
    # A literal pin on the content-addressed layout, so a drive-by change to it fails loudly. Where a
    # catalog keeps its blobs is its own base; the layout under that base is the framework's.
    assert (
        hosted_url("https://cdn.example/public", "usd/vehicles", "astro", "abc")
        == "https://cdn.example/public/assets/usd/vehicles/astro-abc.usdz"
    )
    assert (
        hosted_url("s3://a-bucket/catalog/", "usd/scenes", "site", "abc")
        == "s3://a-bucket/catalog/assets/usd/scenes/site-abc.usdz"
    )


def test_hosted_url_policies_kind():
    # A hosted policy is a sibling tree of the USD one: the kind is the whole folder under assets/.
    assert (
        hosted_url("https://cdn.example/public", "policies", "goto_policy", "abc", "pt")
        == "https://cdn.example/public/assets/policies/goto_policy-abc.pt"
    )


def _stub_aws(tmp_path, monkeypatch, script: str):
    """Put a stub ``aws`` executable on PATH; its argv is ``aws s3 cp <source> <dst> --no-progress``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "aws"
    stub.write_text(f"#!/bin/sh\n{script}\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_s3_fetch_via_aws_cli(tmp_path, monkeypatch):
    src, digest = _make(tmp_path)
    _stub_aws(tmp_path, monkeypatch, f'cp "{src}" "$4"')
    out = fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", digest, cache_dir=tmp_path / "cache")
    assert sha256_file(out) == digest


def test_s3_fetch_no_credentials_friendly_error(tmp_path, monkeypatch):
    _stub_aws(tmp_path, monkeypatch, 'echo "Unable to locate credentials" >&2; exit 1')
    with pytest.raises(RuntimeError, match="credentials for that bucket"):
        fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", "0" * 64, cache_dir=tmp_path / "cache")


def test_s3_fetch_no_cli_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no aws executable anywhere
    with pytest.raises(RuntimeError, match="credentials for that bucket"):
        fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", "0" * 64, cache_dir=tmp_path / "cache")
