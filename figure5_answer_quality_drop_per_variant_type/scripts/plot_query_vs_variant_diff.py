"""Arrow chart of the score drop from the original query to each query variant.

Each arrow runs from the original-query score (grey dot) to the variant score on the same items,
labeled with the mean change, the share of items where the variant scored lower, and significance.
Reads the output of scripts/paired_query_variant_diff.py.

Usage:
    python scripts/plot_query_vs_variant_diff.py [--dataset researchqa|scholarqa|drb|all]
    (all = ResearchQA, ScholarQA-Bench and DeepResearch Bench side by side in one figure)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Reads the JSON written by paired_query_variant_diff.py and saves the figures next to it.
ROOT = Path(__file__).resolve().parent.parent / "output"
# dataset -> (input json, output png, title, short title, y-axis label, (y min, y max), y tick step)
# Y ranges are cropped to the data (plus room for the labels) so the drops are easy to compare.
DATASETS = {
    "researchqa": ("query_vs_variant_diff.json", "query_vs_variant_diff.png",
                   "Coverage score drop vs. original query (ResearchQA)", "ResearchQA",
                   "Rubric-based Coverage Score", (0.1, 0.9), 0.1),
    "scholarqa": ("query_vs_variant_diff_scholarqa.json", "query_vs_variant_diff_scholarqa.png",
                  "Rubric score drop vs. original query (ScholarQA-Bench)", "ScholarQA-Bench",
                  "Rubric-based Coverage Score", (0.0, 0.8), 0.1),
    "drb": ("query_vs_variant_diff_drb.json", "query_vs_variant_diff_drb.png",
            "RACE score drop vs. original query (DeepResearch Bench)", "DeepResearch Bench",
            "Reference-based Adaptive\nCriteria-driven Evaluation Score", (0.1, 0.5), 0.1),
}
COMBINED_OUTPUT = "query_vs_variant_diff_combined.png"
# Panels whose y title is dropped in the combined figure (only ResearchQA keeps one).
COMBINED_HIDE_YLABEL = {"scholarqa", "drb"}

MODELS = {"GPT-5.4": "GPT-5.4 w/ Tools", "Qwen": "DR Tulu", "Tongyi": "Tongyi"}
# Order and colors follow Oct_ARR/fig/query_drift_paired_difference_forest.png
VARIANTS = ["fully_underspecified", "without_preference", "without_scope", "without_term"]
LABELS = {"fully_underspecified": "Fully Underspec.", "without_preference": "W/o Preference",
          "without_scope": "W/o Scope", "without_term": "W/o Term"}
COLORS = {"fully_underspecified": "#e34948", "without_preference": "#008300",
          "without_scope": "#2a78d6", "without_term": "#eda100"}
BACKGROUND = "#ffffff"
ARROW_LW = 5.5
# Drops smaller than this (shown as −0.01 or less) get no arrow, only the dot and labels.
MIN_ARROW_DROP = 0.015


def draw_panel(fig, ax, dataset):
    input_name, _, _, _, ylabel, (ymin, ymax), ystep = DATASETS[dataset]
    yrange = ymax - ymin
    by = {(r["model"], r["variant"]): r for r in json.loads((ROOT / input_name).read_text(encoding="utf-8"))}
    # Label offsets below the arrowhead, as a fraction of the y range so they look the same on any scale.
    off = [yrange * f for f in (0.025, 0.08, 0.13)]
    axes_height_pt = ax.get_position().height * fig.get_figheight() * 72

    ax.set_facecolor(BACKGROUND)
    group_width = 0.9
    slot = group_width / len(VARIANTS)
    for g, model in enumerate(MODELS):
        for j, variant in enumerate(VARIANTS):
            r = by[(model, variant)]
            x = g - group_width / 2 + slot * (j + 0.5)
            q, v = r["query_mean"], r["variant_mean"]
            # Filled arrow (no outline) so the head stays sharp at this width. Sizes are in points:
            # shaft ARROW_LW wide, head 2.5x as wide; the head shortens on very short arrows so it
            # never overshoots the arrow's length.
            length_pt = (q - v) / yrange * axes_height_pt
            head_length_pt = min(ARROW_LW * 2.2, max(4.0, length_pt * 0.9))
            if r["mean_diff"] >= MIN_ARROW_DROP:
                ax.annotate("", xy=(x, v), xytext=(x, q),
                            arrowprops=dict(arrowstyle=f"simple,tail_width={ARROW_LW / 10},"
                                                       f"head_width={ARROW_LW * 2.5 / 10},"
                                                       f"head_length={head_length_pt / 10}",
                                            mutation_scale=10, color=COLORS[variant], lw=0,
                                            shrinkA=0, shrinkB=0), zorder=4)
            ax.plot(x, q, "o", ms=8, color="#52514e", mec=BACKGROUND, mew=1.2, zorder=3)  # under the arrow
            sig = r["significance"]
            ax.text(x, v - off[0], f"−{r['mean_diff']:.2f}", ha="center", va="top", fontsize=10.5,
                    color="#0b0b0b", fontweight="bold")
            ax.text(x, v - off[1], f"{r['pct_variant_lower'] * 100:.0f}%", ha="center", va="top",
                    fontsize=9.5, color="#52514e")
            ax.text(x, v - off[2], sig, ha="center", va="top", fontsize=11 if sig != "ns" else 9.5,
                    color="#0b0b0b" if sig != "ns" else "#898781")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS.values(), fontsize=14, color="#52514e")
    ax.set_xlim(-0.5 + 0.02, len(MODELS) - 0.5 - 0.02)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([round(ymin + i * ystep, 2) for i in range(round(yrange / ystep) + 1)])
    ax.tick_params(axis="y", labelsize=12, labelcolor="#898781")
    ax.tick_params(length=0)
    ax.set_ylabel(ylabel, fontsize=14, color="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    for boundary in (0.5, 1.5):
        ax.axvline(boundary, color="#e1e0d9", linewidth=0.8)


def add_legend(fig, y, fontsize=11.5):
    scale = fontsize / 11.5
    handles = [Line2D([], [], marker="o", ls="", color="#52514e", ms=8 * scale, label="Original Query")] + \
              [Line2D([], [], color=COLORS[v], lw=ARROW_LW * scale, label=LABELS[v]) for v in VARIANTS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y), ncol=5, frameon=False,
               fontsize=fontsize, columnspacing=1.4, handlelength=1.5, handletextpad=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="researchqa")
    args = parser.parse_args()

    if args.dataset == "all":
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.4 * len(DATASETS), 6.3), dpi=200)
        fig.subplots_adjust(left=0.045, right=0.995, top=0.83, bottom=0.1, wspace=0.08)
        for ax, dataset in zip(axes, DATASETS):
            draw_panel(fig, ax, dataset)
            if dataset in COMBINED_HIDE_YLABEL:
                ax.set_ylabel("")
            ax.set_title(DATASETS[dataset][3], fontsize=16, color="#0b0b0b", pad=12)
        add_legend(fig, 0.95, fontsize=16)
        output = ROOT / COMBINED_OUTPUT
    else:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.1)
        draw_panel(fig, ax, args.dataset)
        fig.suptitle(DATASETS[args.dataset][2], fontsize=16, color="#0b0b0b", y=0.97)
        add_legend(fig, 0.88)
        output = ROOT / DATASETS[args.dataset][1]

    fig.patch.set_facecolor(BACKGROUND)
    fig.savefig(output, facecolor=fig.get_facecolor())
    print(f"saved {output}")


if __name__ == "__main__":
    main()
