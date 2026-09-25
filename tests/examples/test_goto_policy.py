"""The goto_policy example's policy resolution: the hosted policy from the catalog's base, or the
``--policy`` override. No GPU, no network: the base is a ``file://`` directory.
"""

import hashlib
import re

import pytest

pytest.importorskip("torch")

from nexus.examples.controllers.policy.goto import flight


def _catalog_with_base(tmp_path, monkeypatch):
    """A project catalog in the working directory whose ``assets.base`` is a ``file://`` directory,
    and an asset cache under ``tmp_path``. Returns the base directory.
    """
    base = tmp_path / "blobs"
    base.mkdir()
    (tmp_path / "nexus.registry.yaml").write_text(f"assets:\n  base: {base.as_uri()}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))
    return base


def test_hosted_policy_is_fetched_from_the_catalog_base(tmp_path, monkeypatch):
    """The hosted policy is fetched from the catalog's `assets.base` at
    `<base>/assets/policies/<name>-<sha>.pt`.
    """
    data = b"a torchscript policy"
    sha = hashlib.sha256(data).hexdigest()
    base = _catalog_with_base(tmp_path, monkeypatch)
    blob = base / "assets" / "policies" / f"goto_policy-{sha}.pt"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)

    out = flight._resolve_policy(None, asset={"name": "goto_policy", "sha256": sha})

    got = hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert (str(tmp_path / "cache") in out, got) == (True, sha)


def test_unreachable_hosted_policy_exits_with_the_policy_hint(tmp_path, monkeypatch):
    """A hosted policy the base cannot serve exits with the `--policy` hint, not a traceback."""
    _catalog_with_base(tmp_path, monkeypatch)  # an empty base: nothing to fetch

    with pytest.raises(SystemExit) as exc:
        flight._resolve_policy(None)

    message = str(exc.value)
    assert ("--policy" in message, "train.py" in message) == (True, True)


def test_policy_override_flies_that_file_and_skips_the_fetch(tmp_path, monkeypatch):
    """`--policy <file>` flies that file and skips the fetch."""
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"a fresh local export")
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))

    out = flight._resolve_policy(str(policy))

    assert (out, (tmp_path / "cache").exists()) == (str(policy), False)


def test_policy_override_that_is_not_a_file_exits_naming_it(tmp_path):
    """`--policy <not-a-file>` exits naming the path."""
    missing = tmp_path / "nope.pt"

    with pytest.raises(SystemExit, match=re.escape("nope.pt")):
        flight._resolve_policy(str(missing))
