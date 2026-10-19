"""The peer legs a change flies, scripts/ci/peer_legs.py: which peers a diff touches, read from the tree, and
what each touched peer's box runs, its tests and the example one table maps it to.
"""

import importlib.util
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Loaded as it runs, `python scripts/ci/peer_legs.py`, and registered under its name: a dataclass's decorator
# looks its module up in `sys.modules` to resolve the annotations.
_spec = importlib.util.spec_from_file_location("peer_legs", ROOT / "scripts" / "ci" / "peer_legs.py")
peer_legs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = peer_legs
_spec.loader.exec_module(peer_legs)


def _tree(root: pathlib.Path, *peers: str, markers: tuple[str, ...] = ()) -> pathlib.Path:
    """A repo tree at ``root`` with a peer folder per name, each with its ``pin.ref``, and a ``pyproject.toml``
    that registers ``markers`` with pytest, as the repo's does.
    """
    for name in peers:
        folder = root / "nexus_uav" / "_src" / "peers" / name
        folder.mkdir(parents=True)
        (folder / "pin.ref").write_text("example/repo@0123abc\n")
    registered = "".join(f'    "{m}: flies {m}",\n' for m in markers)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""\
            [tool.pytest.ini_options]
            markers = [
            {registered}]
            """
        )
    )
    return root


def test_a_new_peer_folder_in_the_tree_is_a_peer_with_no_edit_to_ci(tmp_path):
    """The peers are the folders under `nexus_uav/_src/peers/`, read from the tree, and no list in CI names them:
    given a tree with a new folder `nexus_uav/_src/peers/foo/`, when a path under it changes and the leg computes
    the touched peers, then `foo` is among them with no edit to CI.
    """
    root = _tree(tmp_path, "foo")

    assert peer_legs.touched(["nexus_uav/_src/peers/foo/runner.py"], root) == ["foo"]


def test_a_path_under_a_peers_folder_touches_that_peer_alone():
    """The peers a diff touches follow from one rule, a path under `peers/<name>/` names that peer: given the
    changed paths `nexus_uav/_src/peers/px4_sitl/pin.ref` and `docs/index.md`, when the leg computes the
    touched peers, then `px4_sitl` alone.
    """
    changed = ["nexus_uav/_src/peers/px4_sitl/pin.ref", "docs/index.md"]

    assert peer_legs.touched(changed, ROOT) == ["px4_sitl"]


def test_a_path_in_the_peers_folder_itself_touches_every_peer():
    """A path in `peers/` itself names every peer: given `nexus_uav/_src/peers/instance.py`, when the leg
    computes the touched peers, then `kit` and `px4_sitl`.
    """
    assert peer_legs.touched(["nexus_uav/_src/peers/instance.py"], ROOT) == ["kit", "px4_sitl"]


def test_a_path_outside_the_peers_folder_touches_no_peer():
    """Any other path names none: given `nexus_uav/examples/controllers/px4/flight.py` alone, when the leg
    computes the touched peers, then none.
    """
    assert peer_legs.touched(["nexus_uav/examples/controllers/px4/flight.py"], ROOT) == []


def test_the_table_maps_px4_sitl_to_its_example_and_names_no_other_peer():
    """One table maps a peer's name to the example its box flies: given the table, then it holds
    `px4_sitl` → `px4_sitl` and no `kit` until #125 adds it.
    """
    assert peer_legs.EXAMPLES == {"px4_sitl": "px4_sitl"}


def test_a_touched_peer_with_tests_and_no_table_entry_flies_its_tests_alone(tmp_path):
    """A touched peer the table doesn't name flies its tests alone: given a touched peer with tests marked
    with its name and no table entry, when the leg computes its box's work, then its tests and no example.
    """
    root = _tree(tmp_path, "foo", markers=("foo",))

    assert peer_legs.legs(["nexus_uav/_src/peers/foo/runner.py"], root) == [
        peer_legs.Leg(peer="foo", tests="foo", example=None)
    ]


def test_a_touched_peer_with_no_tests_and_no_example_has_nothing_to_fly():
    """A touched peer with no test marked with its name and no example has nothing to fly: given `kit`
    touched, with no test marked `kit` and no entry, when the leg computes its box's work, then nothing to fly.
    """
    assert peer_legs.legs(["nexus_uav/_src/peers/kit/runner.py"], ROOT) == [
        peer_legs.Leg(peer="kit", tests=None, example=None)
    ]


def test_the_report_names_a_touched_peer_with_nothing_to_fly():
    """The report names a touched peer with nothing to fly: given `kit` touched, with no test marked `kit`
    and no entry, when the leg writes the report, then its line says so.
    """
    legs = [peer_legs.Leg(peer="kit", tests=None, example=None)]

    assert peer_legs.report(legs) == "kit: touched, no test marked kit and no example, nothing to fly"
