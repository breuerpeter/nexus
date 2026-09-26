"""The Kit image's tag: it names the files the image build reads, and nothing else."""

import shutil

from nexus._src.rendering.peer import KIT_DIR, image_tag


def _copy(tmp_path):
    copy = tmp_path / "kit-peer"
    shutil.copytree(KIT_DIR, copy)
    return copy


def test_the_tag_names_the_build_folder_wherever_it_lies(tmp_path):
    """An installed package and a checkout hold the same folder at different paths: one tag."""
    assert image_tag(_copy(tmp_path)) == image_tag(KIT_DIR)


def test_an_edit_to_a_file_the_build_reads_changes_the_tag(tmp_path):
    copy = _copy(tmp_path)
    before = image_tag(copy)
    (copy / "Dockerfile").write_text((copy / "Dockerfile").read_text() + "\n# an edit\n")
    assert image_tag(copy) != before


def test_bytecode_beside_the_program_leaves_the_tag(tmp_path):
    """An installer byte-compiles every Python file it installs; the build ignores the result."""
    copy = _copy(tmp_path)
    before = image_tag(copy)
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "serve.cpython-312.pyc").write_bytes(b"\x00bytecode")
    assert image_tag(copy) == before
