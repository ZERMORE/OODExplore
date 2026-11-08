"""Set cache locations before importing Hugging Face libraries."""
import os
from pathlib import Path


def configure_storage(root=None):
    root = Path(root or os.environ.get("OODEXPLORE_DATA_ROOT", "/wdl/data")).expanduser().resolve()
    os.environ["OODEXPLORE_DATA_ROOT"] = str(root)
    paths = {
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface/hub",
        "HF_DATASETS_CACHE": root / "huggingface/datasets",
        "HF_XET_CACHE": root / "huggingface/xet",
        "HF_ASSETS_CACHE": root / "huggingface/assets",
        "TORCH_HOME": root / "cache/torch",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    # Older Transformers honors this variable ahead of HF_HUB_CACHE.
    # Remove a conflicting inherited value rather than set a deprecated variable.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return root
