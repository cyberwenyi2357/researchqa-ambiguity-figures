"""Per-item records joining the ambiguity-awareness judge category of each variant response with the
paired original-query score and variant score (Gemini 3.1 Pro Preview as judge), for all 3 benchmarks.

Score loaders are shared with scripts/paired_query_variant_diff.py, so the scores match the
query-vs-variant arrow charts. Only items scored under both the query and the variant are kept.

Usage:
    python scripts/build_ambiguity_drop_records.py
Output: plot_input/ambiguity_drop_records_paired.json
    [{dataset, model, variant, id, category, query_score, variant_score, drop}, ...]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from paired_query_variant_diff import VARIANTS, load_drb, load_researchqa, load_scholarqa  # noqa: E402

OUTPUT = ROOT / "plot_input" / "ambiguity_drop_records_paired.json"
MODELS = {"GPT-5.4": "5.4", "Qwen": "qwen", "Tongyi": "tongyi"}  # display name -> judge file suffix
SCHOLARQA_MODELS = {"GPT-5.4": "gpt5.4", "Qwen": "qwen", "Tongyi": "tongyi"}
JUDGE_DIR = ROOT / "data"  # ambiguity-awareness judge output per benchmark
DRB_JUDGE_DIR = JUDGE_DIR / "deepresearch_bench"
SCHOLARQA_JUDGE = JUDGE_DIR / "scholarqa_bench" / "ambiguity_judge_scholarqa_cs.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_researchqa(model, variant):
    rows = read_jsonl(JUDGE_DIR / "researchqa" / f"{variant}_{MODELS[model]}.jsonl")
    return {r["id"]: r["category"] for r in rows}


def judge_drb(model, variant):
    rows = read_jsonl(DRB_JUDGE_DIR / f"answer_deepresearch_bench_{variant}_{MODELS[model]}.jsonl")
    return {r["id"]: r["category"] for r in rows}


_scholarqa_rows = None


def judge_scholarqa(model, variant):
    # Two schemas in one file: hash ids (= case_id of the cs_multi scores) and
    # "<condition>_<n>" / int ids whose number is the row_id of the other scores.
    global _scholarqa_rows
    if _scholarqa_rows is None:
        _scholarqa_rows = read_jsonl(SCHOLARQA_JUDGE)
    out = {}
    for r in _scholarqa_rows:
        if "setting" in r:
            if r["setting"] == variant and r["model"] == SCHOLARQA_MODELS[model]:
                out[("case", r["id"])] = r["category"]
        elif r["condition"] == variant and r["agent"] == SCHOLARQA_MODELS[model]:
            rid = r["id"]
            row = rid if isinstance(rid, int) else int(re.search(r"_(\d+)$", rid).group(1))
            out[("row", row)] = r["category"]
    return out


DATASETS = {
    "ResearchQA": (load_researchqa, judge_researchqa),
    "ScholarQA-Bench": (load_scholarqa, judge_scholarqa),
    "DeepResearch Bench": (load_drb, judge_drb),
}


def main():
    records = []
    for dataset, (load_scores, load_judge) in DATASETS.items():
        for model in MODELS:
            query = load_scores(model, "query")
            for variant in VARIANTS:
                scores = load_scores(model, variant)
                for item_id, category in load_judge(model, variant).items():
                    if item_id in query and item_id in scores:
                        records.append({
                            "dataset": dataset, "model": model, "variant": variant,
                            "id": item_id if not isinstance(item_id, tuple) else f"{item_id[0]}:{item_id[1]}",
                            "category": category, "query_score": query[item_id],
                            "variant_score": scores[item_id], "drop": query[item_id] - scores[item_id],
                        })

    OUTPUT.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {OUTPUT} ({len(records)} records)")
    for key, n in sorted(Counter((r["dataset"], r["category"]) for r in records).items()):
        print(f"  {key[0]:20s} {key[1]:32s} {n}")


if __name__ == "__main__":
    main()
