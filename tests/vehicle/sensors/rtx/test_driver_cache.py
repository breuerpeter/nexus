"""The render stage keeps the driver's shader cache outside ovrtx's install folder, which a fresh environment drops."""

from __future__ import annotations

from nexus_uav._src.vehicle.sensors.rtx.stage import keep_driver_cache


def _cache(install):
    return install / "bin" / "cache" / "nv_shadercache"


def test_a_cache_in_the_install_folder_moves_to_the_kept_folder_and_the_install_links_to_it(tmp_path):
    install, keep = tmp_path / "ovrtx", tmp_path / "home" / "ovrtx" / "nv_shadercache"
    _cache(install).mkdir(parents=True)
    (_cache(install) / "shader.bin").write_text("compiled")

    keep_driver_cache(install, keep)

    assert (_cache(install).is_symlink(), _cache(install).resolve()) == (True, keep.resolve())
    assert (keep / "shader.bin").read_text() == "compiled"


def test_a_fresh_install_links_to_the_cache_a_former_environment_kept(tmp_path):
    install, keep = tmp_path / "ovrtx", tmp_path / "home" / "nv_shadercache"
    keep.mkdir(parents=True)
    (keep / "shader.bin").write_text("compiled")
    _cache(install).mkdir(parents=True)  # what a fresh install's first renderer would write to

    keep_driver_cache(install, keep)
    keep_driver_cache(install, keep)

    assert (_cache(install).resolve(), sorted(p.name for p in keep.iterdir())) == (keep.resolve(), ["shader.bin"])
    assert sorted(p.name for p in _cache(install).parent.iterdir()) == ["nv_shadercache"]


def test_a_link_to_a_kept_folder_since_deleted_gets_the_folder_back(tmp_path):
    install, keep = tmp_path / "ovrtx", tmp_path / "home" / "nv_shadercache"
    _cache(install).mkdir(parents=True)
    keep_driver_cache(install, keep)
    keep.rmdir()

    keep_driver_cache(install, keep)

    assert (keep.is_dir(), _cache(install).resolve()) == (True, keep.resolve())


def test_an_install_folder_the_run_cannot_write_keeps_its_cache_and_the_run_goes_on(tmp_path):
    install, keep = tmp_path / "ovrtx", tmp_path / "home" / "nv_shadercache"
    _cache(install).parent.mkdir(parents=True)
    _cache(install).parent.chmod(0o555)
    try:
        keep_driver_cache(install, keep)
        assert not _cache(install).exists()
    finally:
        _cache(install).parent.chmod(0o755)
