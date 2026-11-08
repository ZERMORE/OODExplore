"""Set cache locations before importing Hugging Face libraries."""
import os
from pathlib import Path


def configure_storage(root=None, *, create=True):
    root = Path(root or os.environ.get("OODEXPLORE_DATA_ROOT") or Path.home() / "wdl" / "data").expanduser().resolve()
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
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot create storage directory {path}: {exc.strerror}. "
                    "Choose an existing writable data disk with --data-root PATH "
                    "(before the subcommand) or export OODEXPLORE_DATA_ROOT=PATH. "
                    "The default is ~/wdl/data under the current user's home directory."
                ) from exc
        os.environ[key] = str(path)
    # Older Transformers honors this variable ahead of HF_HUB_CACHE.
    # Remove a conflicting inherited value rather than set a deprecated variable.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return root
