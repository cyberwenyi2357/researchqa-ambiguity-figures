#!/usr/bin/env bash
# Draw the disambiguation-strategy drop chart (all benchmarks and models pooled) into output/ from the
# paired records already in plot_input/ambiguity_drop_records_paired.json. To rebuild that file from
# data/ (judge categories) and scores/ (per-item scores), run
#   python3 scripts/build_ambiguity_drop_records.py
set -euo pipefail
cd "$(dirname "$0")"

python3 scripts/plot_ambiguity_drop_by_ablation_type.py --pooled
