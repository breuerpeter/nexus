"""Resolver cache + hash-verify: no network/Amazon Web Services (AWS). It uses file:// URLs, and
exercises the ``s3://`` path against a stand-in bucket served on a local endpoint.
"""

import hashlib
import sys

import pytest

from nexus_uav._src.assets import fetch, sha256_file
from nexus_uav._src.assets.resolver import hosted_url


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
    # A hosted policy is a sibling tree of the usdz one: the kind is the whole folder under assets/.
    assert (
        hosted_url("https://cdn.example/public", "policies", "goto_policy", "abc", "pt")
        == "https://cdn.example/public/assets/policies/goto_policy-abc.pt"
    )


def test_an_s3_asset_lands_in_the_cache_with_no_aws_tool(tmp_path, bucket):
    """With `nexus-uav[s3]` installed and no `aws` tool on PATH, an `s3://` asset lands in the
    verified cache.
    """
    bucket["bkt/catalog/assets/usd/vehicles/x.usdz"] = b"astro_max usd bytes"
    digest = "f64912b37b15da27a2aa364a89fb43ceb77550d414b81b51ce7179f2e0cf97f6"
    out = fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", digest, cache_dir=tmp_path / "cache")
    assert out.read_bytes() == b"astro_max usd bytes"


def test_an_s3_fetch_with_no_credentials_names_the_url_and_the_credential_chain(tmp_path, no_aws):
    """An `s3://` fetch with no credentials fails, naming the URL and the credential chain, not the
    `aws` tool.
    """
    url = "s3://bkt/catalog/assets/usd/vehicles/x.usdz"
    with pytest.raises(RuntimeError) as err:
        fetch(url, "0" * 64, cache_dir=tmp_path / "cache")
    message = str(err.value)
    assert all(s in message for s in (url, "AWS_PROFILE", "instance profile")) and "executable" not in message


def test_an_s3_fetch_without_the_s3_extra_names_it(tmp_path, monkeypatch, no_aws):
    """Without the `s3` extra, an `s3://` fetch fails, naming `nexus-uav[s3]`."""
    monkeypatch.setitem(sys.modules, "boto3", None)  # an import of boto3 raises ImportError
    with pytest.raises(RuntimeError, match=r"nexus-uav\[s3\]"):
        fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", "0" * 64, cache_dir=tmp_path / "cache")
