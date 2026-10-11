#!/usr/bin/env bash
# Draw all four query-vs-variant arrow charts into output/ from the paired statistics already in
# output/*.json. To recompute those statistics from data/, run
#   python3 scripts/paired_query_variant_diff.py --dataset {researchqa,scholarqa,drb}
set -euo pipefail
cd "$(dirname "$0")"

for dataset in all researchqa scholarqa drb; do
    python3 scripts/plot_query_vs_variant_diff.py --dataset "$dataset"
done
