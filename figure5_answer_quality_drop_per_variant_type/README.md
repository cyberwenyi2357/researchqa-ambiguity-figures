# Query vs. query-variant score drop (arrow charts)

Self-contained copy of the code and data behind `query_vs_variant_diff_combined.png`: for each model
and benchmark, the mean score on the original query vs. each ablated query variant (paired on the same
items), the share of items that scored lower, and a Wilcoxon signed-rank test (Bonferroni x12).

## Run

```bash
pip install matplotlib
bash run.sh
```

`run.sh` only draws the charts, from the paired statistics already in `output/`:

- `query_vs_variant_diff.json`, `query_vs_variant_diff_scholarqa.json`, `query_vs_variant_diff_drb.json`
  - paired statistics per model x variant (input to the charts).
- `query_vs_variant_diff_combined.png` - the three benchmarks side by side.
- `query_vs_variant_diff.png`, `query_vs_variant_diff_scholarqa.png`, `query_vs_variant_diff_drb.png`
  - one benchmark each.

To recompute the statistics from the per-item scores in `data/` (needs `scipy`):

```bash
for d in researchqa scholarqa drb; do python3 scripts/paired_query_variant_diff.py --dataset $d; done
```

## Layout

- `scripts/paired_query_variant_diff.py` - per-item scores -> paired statistics (JSON).
- `scripts/plot_query_vs_variant_diff.py` - statistics -> charts. Titles, axis ranges, colors, model
  names, arrow width etc. are constants at the top.
- `data/` - per-item scores, Gemini 3.1 Pro Preview as judge (query + 4 variants x 3 models):
  - `researchqa/{gpt 5.4|qwen|tongyi}/results_{variant}_*.json` - rubric `coverage_score`.
  - `deepresearch_bench/{model}_{variant}_gemini3.1/raw_results.jsonl` - RACE `overall_score`.
  - `scholarqa_bench/scholarQA_score_all.json` - rubric `score`; two schemas, joined by `row_id`
    or by the hash `case_id`.

Copied from `ResearchQA/scripts`, `ResearchQA/results_using_Gemini3.1ProPreview_as_judge`,
`ResearchQA/results` and `~/deep_research_bench/results/race`; the copies of the scripts only differ in
the paths at the top.
