# Answer-quality drop by disambiguation strategy

Self-contained copy of the code and data behind `ambiguity_drop_by_ablation_type*.png`: for each
ablation type, the score drop (original query -> variant) of responses that stated an assumption and
proceeded vs. responses with no recognition of the ambiguity.

## Run

```bash
pip install matplotlib numpy
bash run.sh
```

`run.sh` only draws the chart, from `plot_input/ambiguity_drop_records_paired.json`, into
`output/ambiguity_drop_by_ablation_type_pooled.png`: all benchmarks and models pooled (standardized
scores), one panel per ablation type, every arrow starting at 0.

To rebuild the records from `data/` and `scores/` (needs `scipy`):

```bash
python3 scripts/build_ambiguity_drop_records.py
```

## Layout

- `scripts/plot_ambiguity_drop_by_ablation_type.py` - records -> charts.
- `scripts/build_ambiguity_drop_records.py` - judge categories + scores -> records.
- `scripts/paired_query_variant_diff.py` - the per-item score loaders the build script uses.
- `plot_input/ambiguity_drop_records_paired.json` - one record per variant response:
  `{dataset, model, variant, id, category, query_score, variant_score, drop}`.
- `data/` - ambiguity-awareness judge category of each variant response (`stated_assumption_and_proceeded`,
  `asked_clarifying_question` or `no_recognition`):
  - `researchqa/{variant}_{5.4|qwen|tongyi}.jsonl`
  - `deepresearch_bench/answer_deepresearch_bench_{variant}_{5.4|qwen|tongyi}.jsonl`
  - `scholarqa_bench/ambiguity_judge_scholarqa_cs.jsonl` - two record schemas: hash `id` (matches
    `case_id` of the cs_multi scores) and `<condition>_<n>` / int `id` (n = `row_id`).
- `scores/` - per-item scores, Gemini 3.1 Pro Preview as judge (query + 4 variants x 3 models):
  - `researchqa/{gpt 5.4|qwen|tongyi}/results_{variant}_*.json` - rubric `coverage_score`.
  - `deepresearch_bench/{model}_{variant}_gemini3.1/raw_results.jsonl` - RACE `overall_score`.
  - `scholarqa_bench/scholarQA_score_all.json` - rubric `score`.

Copied from `ResearchQA/scripts`, `ResearchQA/results`, `ResearchQA/results_using_Gemini3.1ProPreview_as_judge`
and `~/deep_research_bench/results`; the copies of the scripts only differ in the paths at the top.
