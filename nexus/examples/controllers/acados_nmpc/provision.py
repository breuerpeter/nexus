"""How the acados example reaches acados on this machine.

acados can't come from pip: it's a C library the machine compiles, and its Python interface,
``acados_template``, ships only inside that source tree, tied to the library built from it. So the
example provisions one tree and finds everything in it. The example ships the pin, ``acados.ref``
beside this module: line one is ``repo@sha``. :func:`provision` fetches that commit into
:func:`acados_dir`, builds the library there with CMake, and downloads the Tera renderer acados
needs to generate a solver's C code. :func:`require` then makes that tree usable in the process: it
puts the tree's interface on ``sys.path`` and loads the tree's C libraries, or says what this machine
still lacks.

The Python packages the interface imports are on PyPI, and the ``acados`` extra names them.

This module imports the standard library only, so the flight script and the CI harness call it
before the example's own imports.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROVISION_COMMAND = "python -m nexus.examples acados_nmpc --provision"
TERA_VERSION = "0.2.1"
# The sha256 of the release binary, read from two separate downloads on 2026-10-05. Move it with the version.
TERA_SHA256 = "64a0a0f8d85be0b92234231fd1a9ff50c9d8f13d6208063707e56c9354a976d3"
_PIN = Path(__file__).with_name("acados.ref")
# What the example and the interface import at import time, all from the `acados` extra.
_EXTRA_MODULES = ("casadi", "ruckig", "matplotlib", "deprecated")
# The tree's C libraries, each after the ones it needs, and the handles this process holds on them.
_LIBRARIES = ("libblasfeo.so", "libhpipm.so", "libacados.so")
_loaded: list[ctypes.CDLL] = []


def pin() -> tuple[str, str]:
    """The acados tree the example flies: its GitHub repository, ``owner/name``, and its commit."""
    repo, sha = _PIN.read_text().splitlines()[0].strip().split("@", 1)
    return repo, sha


def acados_dir() -> Path:
    """The acados tree: ``$ACADOS_SOURCE_DIR``, acados' own convention, when set, else the folder
    the example owns for the pinned commit, ``~/.cache/nexus/acados/<sha>``.
    """
    if env := os.environ.get("ACADOS_SOURCE_DIR"):
        return Path(env)
    return Path.home() / ".cache" / "nexus" / "acados" / pin()[1]


def _interface(tree: Path) -> Path:
    return tree / "interfaces" / "acados_template"


def _missing() -> list[str]:
    """What this machine lacks to fly the example, each with the command that provides it."""
    missing = []
    if any(importlib.util.find_spec(module) is None for module in _EXTRA_MODULES):
        missing.append(
            "the acados extra isn't installed: pip install 'nexus-sim[acados]', or uv sync --extra acados in a checkout"
        )
    tree = acados_dir()
    built = (tree / "lib" / "libacados.so").exists() and os.access(tree / "bin" / "t_renderer", os.X_OK)
    if not (built and _interface(tree).is_dir()):
        missing.append(f"acados isn't built at {tree}: {PROVISION_COMMAND}")
    return missing


def require() -> None:
    """Make acados usable in this process from the provisioned tree: ``acados_template`` importable and
    the tree's C libraries loaded, so the caller sets no ``LD_LIBRARY_PATH``.

    Raises:
        RuntimeError: The ``acados`` extra or the acados build is missing. The message
            names each missing part and the command that provides it.
    """
    if missing := _missing():
        raise RuntimeError("acados_nmpc can't fly on this machine:\n" + "\n".join(f"  - {m}" for m in missing))
    tree = acados_dir()
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(tree))  # where the interface looks for the tree
    interface = str(_interface(tree))
    if interface not in sys.path:
        sys.path.insert(0, interface)
    if not _loaded:
        # libacados.so names libhpipm.so and libblasfeo.so with no path, and a generated solver names
        # libacados.so the same way. The loader resolves such a name among the libraries the process
        # already holds, so loading the three by path here does what LD_LIBRARY_PATH would, and the
        # loader reads that variable only when a process starts.
        _loaded.extend(ctypes.CDLL(str(tree / "lib" / name), mode=ctypes.RTLD_GLOBAL) for name in _LIBRARIES)


def _run(*cmd: str | Path) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def provision() -> Path:
    """Fetch the pinned acados commit into :func:`acados_dir` and build it there. Idempotent: a tree
    that already holds the library keeps it, and so does a renderer already downloaded.

    Returns:
        The acados tree.

    Raises:
        RuntimeError: This machine has no CMake, or the renderer download doesn't match the pinned
            digest.
        subprocess.CalledProcessError: A git or CMake command failed.
    """
    tree = acados_dir()
    if not (tree / "lib" / "libacados.so").exists():
        if shutil.which("cmake") is None:
            raise RuntimeError("cmake not found: install it, for example with apt install cmake")
        if not (tree / "CMakeLists.txt").exists():
            repo, sha = pin()
            print(f"fetching acados {sha[:12]} into {tree}", flush=True)
            tree.mkdir(parents=True, exist_ok=True)
            _run("git", "-C", tree, "init", "-q")
            _run("git", "-C", tree, "fetch", "-q", "--depth", "1", f"https://github.com/{repo}.git", sha)
            _run("git", "-C", tree, "checkout", "-q", "FETCH_HEAD")
            _run("git", "-C", tree, "submodule", "update", "--quiet", "--init", "--recursive")
        # qpOASES's vendored C99 doesn't compile under gcc 14 and later, so it stays off. The High
        # Performance Interior Point Method (HPIPM) solver the example uses is acados' default.
        _run(
            "cmake", "-S", tree, "-B", tree / "build",
            "-DACADOS_WITH_QPOASES=OFF", f"-DACADOS_INSTALL_DIR={tree}", "-DCMAKE_BUILD_TYPE=Release",
        )  # fmt: skip
        _run("cmake", "--build", tree / "build", "--target", "install", "-j")
    renderer = tree / "bin" / "t_renderer"
    if not os.access(renderer, os.X_OK):
        # The C build doesn't include the renderer: acados publishes it as a binary of its own. This
        # module pins its bytes, as it pins everything else it fetches, so a changed release never runs.
        url = f"https://github.com/acados/tera_renderer/releases/download/v{TERA_VERSION}/t_renderer-v{TERA_VERSION}-linux-amd64"
        renderer.parent.mkdir(parents=True, exist_ok=True)
        download = renderer.with_suffix(".part")
        urllib.request.urlretrieve(url, download)
        digest = hashlib.sha256(download.read_bytes()).hexdigest()
        if digest != TERA_SHA256:
            download.unlink()
            raise RuntimeError(f"{url} has sha256 {digest}, not the pinned {TERA_SHA256}: the renderer isn't installed")
        download.chmod(0o755)
        download.replace(renderer)
    print(f"acados ready at {tree}", flush=True)
    return tree
