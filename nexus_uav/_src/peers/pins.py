"""How every peer finds its pin: what it runs, a commit or an image.

Each peer ships its pin in its own folder, ``peers/<name>/pin.ref``, with the pin on line one. An
installed package that adds a catalog overrides it with a ``nexus.<name>.ref`` of the same form beside
that catalog, a project's file beside its own catalog overrides a package's, and a variable named for
the peer overrides them all, for a pin or a local source such as a checkout of your own. The name is the
peer's folder name, as in ``nexus.px4_sitl.ref`` and ``NEXUS_KIT_REF``. This module reads the value and
its source. The peer states the form the value takes and checks it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from nexus_uav._src.config.catalog import CatalogSource

_PEERS = Path(__file__).parent


class PeerPin(BaseModel):
    """What a peer runs, as the receipt records it: its pin as its source states it, and that source.

    The source is the variable that set it, ``NEXUS_KIT_REF`` for the Kit peer, or the path of the pin file: a
    ``nexus.<name>.ref`` beside a project's or an installed package's catalog, or the one the peer ships in
    its folder. ``commit`` is the checked-out commit of a pin that names a local checkout, else ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    ref: str
    source: str
    commit: str | None = None


def variable(name: str) -> str:
    """The variable that overrides peer ``name``'s pin: ``NEXUS_KIT_REF`` for ``kit``."""
    return f"NEXUS_{name.upper()}_REF"


def shipped() -> list[str]:
    """The name of each peer the framework ships: each folder here that holds a ``pin.ref``."""
    return sorted(folder.name for folder in _PEERS.iterdir() if (folder / "pin.ref").is_file())


def _pin_folders(sources: Sequence[CatalogSource]) -> list[CatalogSource]:
    """The sources that can hold pin files: every one but the bundled catalog, whose pins live in the peers' folders."""
    return [source for source in sources if source.kind != "bundled"]


def resolve(name: str, sources: Sequence[CatalogSource] = ()) -> PeerPin | None:
    """Peer ``name``'s pin and its source: the variable, else the pin file of the top source that holds one,
    else the pin file the peer ships, or ``None`` for a peer, a project's own, that no source pins. The variable
    wins whenever the environment holds it, empty or not, so the peer's form check fails on it rather than fly a
    pin the variable meant to replace.

    Args:
        name: The peer's folder name, such as ``px4_sitl``.
        sources: The catalog sources the run loaded, in load order; none for a run handed its catalog.

    Raises:
        ValueError: Two installed packages ship a pin file for the peer and neither a project file nor the
            variable picks one; the message names the peer and the packages.
    """
    if (value := os.environ.get(variable(name))) is not None:
        return PeerPin(ref=value.strip(), source=variable(name))
    filename = f"nexus.{name}.ref"
    holding = [source for source in _pin_folders(sources) if (source.path.parent / filename).is_file()]
    if len(holding) > 1 and holding[-1].kind == "package":
        packages = " and ".join(source.name for source in holding)
        raise ValueError(
            f"peer {name!r}: installed packages {packages} each ship {filename}, and no order between them means "
            f"anything; a project's {filename} or {variable(name)} picks the pin"
        )
    source = holding[-1].path.parent / filename if holding else _PEERS / name / "pin.ref"
    if not source.is_file():
        return None
    lines = source.read_text().splitlines()
    return PeerPin(ref=lines[0].strip() if lines else "", source=str(source))


def check(sources: Sequence[CatalogSource], names: list[str]) -> None:
    """Fail on a pin file beside a catalog of ``sources`` that names no peer of ``names``.

    Raises:
        ValueError: A ``nexus.<name>.ref`` beside a catalog names no peer; the message names the
            file and the peer names a pin file takes.
    """
    for source in _pin_folders(sources):
        for path in sorted(source.path.parent.glob("nexus.*.ref")):
            if path.name.removeprefix("nexus.").removesuffix(".ref") not in names:
                accepted = ", ".join(f"nexus.{n}.ref" for n in sorted(names))
                raise ValueError(f"{path} names no peer; a pin file beside a catalog is one of {accepted}")
