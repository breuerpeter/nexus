"""How every peer finds its pin: what it runs, a commit or an image.

Each peer ships its pin in its own folder, ``peers/<name>/pin.ref``, with the pin on line one. A
project overrides it with a ``nexus.<name>.ref`` of the same form beside its catalog, and a
variable named for the peer overrides both, for a pin or a local source such as a checkout of your
own. The name is the peer's folder name, as in ``nexus.px4_sitl.ref`` and ``NEXUS_KIT_REF``. This
module reads the value and its source. The peer states the form the value takes and checks it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_PEERS = Path(__file__).parent


class PeerPin(BaseModel):
    """What a peer runs, as the receipt records it: its pin as its source states it, and that source.

    The source is the variable that set it, ``NEXUS_KIT_REF`` for the Kit peer, or the path of the pin file: a
    project's ``nexus.<name>.ref`` beside its catalog, or the one the peer ships in its folder.
    ``commit`` is the checked-out commit of a pin that names a local checkout, else ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    ref: str
    source: str
    commit: str | None = None


def variable(name: str) -> str:
    """The variable that overrides peer ``name``'s pin: ``NEXUS_KIT_REF`` for ``kit``."""
    return f"NEXUS_{name.upper()}_REF"


def resolve(name: str, catalog: Path | None = None) -> PeerPin:
    """Peer ``name``'s pin and its source: the variable, else the project's pin file beside ``catalog``,
    else the pin file the peer ships.

    Args:
        name: The peer's folder name, such as ``px4_sitl``.
        catalog: The project catalog the run resolved against, or ``None`` for no project.
    """
    if value := os.environ.get(variable(name)):
        return PeerPin(ref=value.strip(), source=variable(name))
    source = _PEERS / name / "pin.ref"
    if catalog is not None and (candidate := Path(catalog).parent / f"nexus.{name}.ref").is_file():
        source = candidate
    lines = source.read_text().splitlines()
    return PeerPin(ref=lines[0].strip() if lines else "", source=str(source))


def check(catalog: Path | None, names: list[str]) -> None:
    """Fail on a pin file beside ``catalog`` that names no peer of ``names``.

    Raises:
        ValueError: A ``nexus.<name>.ref`` beside the catalog names no peer; the message names the
            file and the peer names a pin file takes.
    """
    if catalog is None:
        return
    for path in sorted(Path(catalog).parent.glob("nexus.*.ref")):
        if path.name.removeprefix("nexus.").removesuffix(".ref") not in names:
            accepted = ", ".join(f"nexus.{n}.ref" for n in sorted(names))
            raise ValueError(f"{path} names no peer; a pin file beside the catalog is one of {accepted}")
