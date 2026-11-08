"""Storage routing tests to execute on the server, not locally."""
from pathlib import Path
import pytest

from ood_explore.storage import configure_storage


def test_storage_replaces_inherited_hf_cache(monkeypatch, tmp_path):
    for name in ("OODEXPLORE_DATA_ROOT", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
                 "HF_XET_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(name, "/old/cache")
    root = configure_storage(tmp_path)
    import os
    assert root == tmp_path.resolve()
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME"):
        assert Path(os.environ[name]).is_relative_to(root)
        assert Path(os.environ[name]).is_dir()
    assert "TRANSFORMERS_CACHE" not in os.environ


def test_storage_environment_override(monkeypatch, tmp_path):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(name, "old")
    monkeypatch.setenv("OODEXPLORE_DATA_ROOT", str(tmp_path))
    assert configure_storage() == tmp_path.resolve()


def test_show_config_storage_does_not_create_directories(monkeypatch, tmp_path):
    for name in ("OODEXPLORE_DATA_ROOT", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(name, "old")
    target = tmp_path / "absent" / "data"
    assert configure_storage(target, create=False) == target.resolve()
    assert not target.exists()


def test_storage_permission_error_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("OODEXPLORE_DATA_ROOT", str(tmp_path))
    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(RuntimeError, match="--data-root"):
        configure_storage(tmp_path)


def test_default_storage_is_under_user_home(monkeypatch, tmp_path):
    for name in ("OODEXPLORE_DATA_ROOT", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert configure_storage(create=False) == tmp_path / "wdl" / "data"
    assert not (tmp_path / "wdl").exists()
