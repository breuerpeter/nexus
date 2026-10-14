"""Resolver cache + hash-verify: no network/Amazon Web Services (AWS). It uses file:// URLs, and
exercises the ``s3://`` path against a stand-in bucket served on a local endpoint.
"""

import hashlib
import http.server
import os
import sys
import threading

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


def _no_aws(tmp_path, monkeypatch):
    """Drop every AWS setting and the user's AWS files, the instance metadata lookup and every
    executable on PATH, so no credential and no `aws` tool reaches a fetch.
    """
    for key in [k for k in os.environ if k.startswith("AWS_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """A stand-in S3 endpoint on localhost that serves the objects put in the returned dict, keyed
    ``<bucket>/<key>``, with test credentials in the environment and no `aws` tool on PATH.
    """
    objects: dict[str, bytes] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self._reply(body=False)

        def do_GET(self):
            self._reply(body=True)

        def _reply(self, body):
            data = objects.get(self.path.split("?", 1)[0].lstrip("/"))
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", f'"{hashlib.md5(data).hexdigest()}"')
            self.send_header("Last-Modified", "Wed, 01 Jan 2025 00:00:00 GMT")
            self.end_headers()
            if body:
                self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _no_aws(tmp_path, monkeypatch)
    monkeypatch.setenv("AWS_ENDPOINT_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    yield objects
    server.shutdown()
    server.server_close()


def test_an_s3_asset_lands_in_the_cache_with_no_aws_tool(tmp_path, bucket):
    """With `nexus-uav[s3]` installed and no `aws` tool on PATH, an `s3://` asset lands in the
    verified cache.
    """
    bucket["bkt/catalog/assets/usd/vehicles/x.usdz"] = b"astro_max usd bytes"
    digest = "f64912b37b15da27a2aa364a89fb43ceb77550d414b81b51ce7179f2e0cf97f6"
    out = fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", digest, cache_dir=tmp_path / "cache")
    assert out.read_bytes() == b"astro_max usd bytes"


def test_an_s3_fetch_with_no_credentials_names_the_url_and_the_credential_chain(tmp_path, monkeypatch):
    """An `s3://` fetch with no credentials fails, naming the URL and the credential chain, not the
    `aws` tool.
    """
    _no_aws(tmp_path, monkeypatch)
    url = "s3://bkt/catalog/assets/usd/vehicles/x.usdz"
    with pytest.raises(RuntimeError) as err:
        fetch(url, "0" * 64, cache_dir=tmp_path / "cache")
    message = str(err.value)
    assert url in message and "credentials" in message and "executable" not in message


def test_an_s3_fetch_without_the_s3_extra_names_it(tmp_path, monkeypatch):
    """Without the `s3` extra, an `s3://` fetch fails, naming `nexus-uav[s3]`."""
    _no_aws(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "boto3", None)  # an import of boto3 raises ImportError
    with pytest.raises(RuntimeError, match=r"nexus-uav\[s3\]"):
        fetch("s3://bkt/catalog/assets/usd/vehicles/x.usdz", "0" * 64, cache_dir=tmp_path / "cache")
