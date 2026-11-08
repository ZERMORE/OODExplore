#!/usr/bin/env bash
# Source this file before Conda/pip installation or manual HF downloads.
export OODEXPLORE_DATA_ROOT="${OODEXPLORE_DATA_ROOT:-$HOME/wdl/data}"
export HF_HOME="$OODEXPLORE_DATA_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export TORCH_HOME="$OODEXPLORE_DATA_ROOT/cache/torch"
export PIP_CACHE_DIR="$OODEXPLORE_DATA_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$OODEXPLORE_DATA_ROOT/conda/pkgs"
export TMPDIR="$OODEXPLORE_DATA_ROOT/tmp"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" \
  "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"
