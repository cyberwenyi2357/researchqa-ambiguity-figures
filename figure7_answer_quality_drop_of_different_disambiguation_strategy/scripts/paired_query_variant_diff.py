"""Mean score difference between the original query and each query variant, per model,
with paired significance tests (Gemini 3.1 Pro Preview as judge).

Datasets:
  researchqa - rubric coverage_score from scores/researchqa/
  drb        - DeepResearch Bench RACE overall_score from scores/deepresearch_bench/
  scholarqa  - ScholarQA-Bench score from scores/scholarqa_bench/scholarQA_score_all.json

For each model and variant, only ids scored under both the query and the variant are compared.
diff = query score - variant score (positive = the variant scored lower).
Tests: Wilcoxon signed-rank (primary) and paired t-test; p-values Bonferroni-corrected over the
12 model x variant comparisons within the dataset.

Usage:
    python scripts/paired_query_variant_diff.py [--dataset researchqa|drb|scholarqa]
Output: query_vs_variant_diff.json (researchqa) / query_vs_variant_diff_drb.json (drb) /
        query_vs_variant_diff_scholarqa.json (scholarqa) in output/
"""

import argparse
import json
import statistics
from pathlib import Path

from scipy import stats

# In this folder the script is used for its score loaders (by build_ambiguity_drop_records.py).
BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "scores" / "researchqa"
DRB_ROOT = BASE / "scores" / "deepresearch_bench"
SCHOLARQA_FILE = BASE / "scores" / "scholarqa_bench" / "scholarQA_score_all.json"

RQA_MODELS = {
    "GPT-5.4": ("gpt 5.4", "5.4_sample500_batch"),
    "Qwen": ("qwen", "qwen_sample500"),
    "Tongyi": ("tongyi", "tongyi_sample500"),
}
DRB_MODELS = {"GPT-5.4": "gpt5.4", "Qwen": "qwen", "Tongyi": "tongyi"}
VARIANTS = ["fully_underspecified", "without_scope", "without_term", "without_preference"]


def load_researchqa(model, variant):
    folder, suffix = RQA_MODELS[model]
    data = json.loads((ROOT / folder / f"results_{variant}_{suffix}.json").read_text(encoding="utf-8"))
    return {obj_id: obj["coverage_score"] for obj_id, obj in data.items() if obj.get("coverage_score") is not None}


def load_drb(model, variant):
    scores = {}
    with (DRB_ROOT / f"{DRB_MODELS[model]}_{variant}_gemini3.1" / "raw_results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("overall_score") is not None:
                    scores[r["id"]] = r["overall_score"]
    return scores


def load_scholarqa(model, variant):
    # Two record schemas: rows with row_id (stable across variants) and cs_multi rows keyed by a hash case_id.
    scores = {}
    for r in json.loads(SCHOLARQA_FILE.read_text(encoding="utf-8")):
        if r["model"] == DRB_MODELS[model] and r["variant"] == variant and r.get("score") is not None:
            key = ("row", r["row_id"]) if "row_id" in r else ("case", r["case_id"])
            scores[key] = r["score"]
    return scores


DATASETS = {
    "researchqa": (load_researchqa, BASE / "output" / "query_vs_variant_diff.json"),
    "drb": (load_drb, BASE / "output" / "query_vs_variant_diff_drb.json"),
    "scholarqa": (load_scholarqa, BASE / "output" / "query_vs_variant_diff_scholarqa.json"),
}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="researchqa")
    args = parser.parse_args()
    load_scores, output = DATASETS[args.dataset]

    rows = []
    for model in RQA_MODELS:
        query = load_scores(model, "query")
        for variant in VARIANTS:
            other = load_scores(model, variant)
            ids = sorted(set(query) & set(other))
            q = [query[i] for i in ids]
            v = [other[i] for i in ids]
            diffs = [a - b for a, b in zip(q, v)]
            rows.append({
                "model": model, "variant": variant, "n": len(ids),
                "query_mean": statistics.mean(q), "variant_mean": statistics.mean(v),
                "mean_diff": statistics.mean(diffs), "sd_diff": statistics.stdev(diffs),
                "pct_variant_lower": sum(d > 0 for d in diffs) / len(diffs),
                "wilcoxon_p": stats.wilcoxon(q, v).pvalue,
                "ttest_p": stats.ttest_rel(q, v).pvalue,
            })

    for r in rows:
        r["wilcoxon_p_bonf"] = min(1.0, r["wilcoxon_p"] * len(rows))
        r["ttest_p_bonf"] = min(1.0, r["ttest_p"] * len(rows))
        r["significance"] = stars(r["wilcoxon_p_bonf"])

    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{'model':8s} {'variant':22s} {'n':>4s} {'query':>7s} {'variant':>8s} {'diff':>7s} "
          f"{'%lower':>7s} {'wilcoxon p(bonf)':>17s} {'t-test p(bonf)':>15s}  sig")
    for r in rows:
        print(f"{r['model']:8s} {r['variant']:22s} {r['n']:4d} {r['query_mean']:7.4f} {r['variant_mean']:8.4f} "
              f"{r['mean_diff']:+7.4f} {r['pct_variant_lower']*100:6.1f}% {r['wilcoxon_p_bonf']:17.2e} "
              f"{r['ttest_p_bonf']:15.2e}  {r['significance']}")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
