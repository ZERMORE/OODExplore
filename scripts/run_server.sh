#!/usr/bin/env bash
set -euo pipefail

# Run from OODExplore root after: conda activate oodexplore
# Usage: bash scripts/run_server.sh [config] [data-dir] [output-dir]
config="${1:-configs/qwen3b.json}"
data_dir="${2:-data/scienceqa}"
output_dir="${3:-outputs/main}"

if [[ ! -f "$data_dir/manifest.json" ]]; then
  python -m ood_explore prepare --output "$data_dir" --dev-count 100 --test-count 500 --seed 42
fi
python -m ood_explore run --config "$config" --data "$data_dir/dev.jsonl" --output "$output_dir/dev" --resume
python -m ood_explore run --config "$config" --data "$data_dir/test.jsonl" --output "$output_dir/test" --resume
if [[ ! -f "$output_dir/analysis/COMPLETE" ]]; then
  python -m ood_explore analyze --dev "$output_dir/dev/predictions.jsonl" --test "$output_dir/test/predictions.jsonl" --output "$output_dir/analysis"
fi
