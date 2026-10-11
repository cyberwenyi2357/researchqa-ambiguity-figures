# ResearchQA ambiguity figures

Code and data for two figures on how deep research agents (GPT-5.4 w/ Tools, DR Tulu, Tongyi
DeepResearch) handle underspecified queries, evaluated on ResearchQA, ScholarQA-Bench and
DeepResearch Bench with Gemini 3.1 Pro Preview as the judge.

| Folder | Figure |
|---|---|
| [`figure5_answer_quality_drop_per_variant_type/`](figure5_answer_quality_drop_per_variant_type/) | Score drop from the original query to each ablated query variant, per model and benchmark (paired, Wilcoxon signed-rank). |
| [`figure7_answer_quality_drop_of_different_disambiguation_strategy/`](figure7_answer_quality_drop_of_different_disambiguation_strategy/) | Score drop of responses that stated an assumption and proceeded vs. responses with no recognition of the ambiguity, per ablation type. |

Each folder is self-contained: `bash run.sh` redraws the figure into `output/` from the analysis
results already in the folder, and its README explains how to recompute those results from the raw
per-item data.

```bash
pip install matplotlib numpy scipy
bash figure5_answer_quality_drop_per_variant_type/run.sh
bash figure7_answer_quality_drop_of_different_disambiguation_strategy/run.sh
```
