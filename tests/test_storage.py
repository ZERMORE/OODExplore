"""Storage routing tests to execute on the server, not locally."""
from pathlib import Path

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
