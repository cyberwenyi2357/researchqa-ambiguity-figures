"""Arrow chart: paired original-query score -> variant score, for responses that stated an assumption
vs. those with no recognition of the ambiguity, by ablation type, one panel per benchmark
(pooled across the 3 models). Same drawing style as scripts/plot_query_vs_variant_diff.py.

Each arrow runs from the mean original-query score (dot) to the mean variant score of the same items.
Each arrow is labeled with the mean change.
Reads plot_input/ambiguity_drop_records_paired.json (scripts/build_ambiguity_drop_records.py).

With --pooled, the 3 benchmarks are merged into one panel. Their scores are on different scales, so
each score is standardized first: (score - benchmark mean original-query score) / benchmark score SD.
A drop of 1.0 then means one SD of that benchmark's score, the same scaling as the earlier SD-scaled drops.

With --by-model, the standardized scores of the 3 benchmarks are pooled and split into one panel per
model, all on the same y axis.

Usage:
    python scripts/plot_ambiguity_drop_by_ablation_type.py [--pooled | --by-model]
Output: results/ambiguity_drop_by_ablation_type.png (per benchmark)
        results/ambiguity_drop_by_ablation_type_pooled.png (--pooled)
        results/ambiguity_drop_by_ablation_type_by_model.png (--by-model)
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "plot_input" / "ambiguity_drop_records_paired.json"
ROOT = BASE / "output"
OUTPUT = ROOT / "ambiguity_drop_by_ablation_type.png"
POOLED_OUTPUT = ROOT / "ambiguity_drop_by_ablation_type_pooled.png"
BY_MODEL_OUTPUT = ROOT / "ambiguity_drop_by_ablation_type_by_model.png"

# dataset -> (y-axis label, (y min, y max), y tick step)
DATASETS = {
    "ResearchQA": ("Rubric-based Coverage Score", (0.1, 0.85), 0.1),
    "ScholarQA-Bench": ("", (0.1, 0.75), 0.1),
    "DeepResearch Bench": ("", (0.15, 0.5), 0.05),
}
POOLED = ("All Benchmarks", "Standardized Score (SD Units)", (-1.75, 0.25), 0.25)
BY_MODEL_YLIM = (-1.85, 0.5)
MODEL_TITLES = {"GPT-5.4": "GPT-5.4 w/ Tools", "Qwen": "DR Tulu", "Tongyi": "Tongyi"}
# Score SD per benchmark, the same values used for the earlier SD-scaled drops
# (results/ambiguity_drop_records_combined.json: raw_drop / scaled_drop).
SCORE_SD = {"ResearchQA": 0.328888, "ScholarQA-Bench": 0.300894, "DeepResearch Bench": 0.125845}
GROUPS = ["fully_underspecified", "without_preference", "without_scope", "without_term"]
GROUP_LABELS = {"fully_underspecified": "Fully\nUnderspec.", "without_preference": "W/o\nPreference",
                "without_scope": "W/o\nScope", "without_term": "W/o\nTerm"}
CATEGORIES = ["stated_assumption_and_proceeded", "no_recognition"]
CATEGORY_LABELS = {"stated_assumption_and_proceeded": "Stated Assumption & Proceeded",
                   "no_recognition": "No Recognition of Ambiguity"}
# Kept clear of the red/green/blue/yellow used for query types in plot_query_vs_variant_diff.py.
COLORS = {"stated_assumption_and_proceeded": "#6f42c1", "no_recognition": "#9a9893"}
BACKGROUND = "#ffffff"
ARROW_LW = 5.5


def draw_arrow(ax, x, start, end, color, yrange, axes_height_pt, lw=ARROW_LW):
    length_pt = abs(start - end) / yrange * axes_height_pt
    head_length_pt = min(lw * 2.2, max(4.0, length_pt * 0.9))
    ann = ax.annotate("", xy=(x, end), xytext=(x, start), annotation_clip=False,
                      arrowprops=dict(arrowstyle=f"simple,tail_width={lw / 10},"
                                                 f"head_width={lw * 2.5 / 10},"
                                                 f"head_length={head_length_pt / 10}",
                                      mutation_scale=10, color=color, lw=0, shrinkA=0, shrinkB=0), zorder=4)
    ann.arrow_patch.set_clip_path(ax.patch)  # an arrow starting above a cropped axis is cut at the frame


def standardize(records):
    query_mean = {ds: statistics.mean(r["query_score"] for r in records if r["dataset"] == ds) for ds in SCORE_SD}
    out = []
    for r in records:
        mu, sd = query_mean[r["dataset"]], SCORE_SD[r["dataset"]]
        out.append({**r, "query_score": (r["query_score"] - mu) / sd,
                    "variant_score": (r["variant_score"] - mu) / sd, "drop": r["drop"] / sd})
    return out


def group_means(records, group):
    """(query mean, variant mean) per category for one ablation type."""
    out = {}
    for category in CATEGORIES:
        rs = [r for r in records if r["variant"] == group and r["category"] == category]
        out[category] = (statistics.mean(r["query_score"] for r in rs), statistics.mean(r["variant_score"] for r in rs))
    return out


def tip_gap(records, group):
    tips = [variant for _, variant in group_means(records, group).values()]
    return max(tips) - min(tips)


def fitted_ylim(records, group, span):
    """A y range of fixed height `span` (the same for every panel, so equal gaps look equal), placed
    around this ablation type's arrowheads: the deeper tip sits just above the bottom, leaving room
    for its label. The axis does not reach 0; arrows enter from above the frame."""
    deeper = min(variant for _, variant in group_means(records, group).values())
    ymin = deeper - span * 0.17
    if ymin + span > span * 0.05:  # don't show empty space above the arrows' start at 0
        ymin = span * 0.05 - span
    step = next(s for s in (0.05, 0.1, 0.2, 0.25, 0.5) if span / s <= 5)
    return (ymin, ymin + span), step


def fade_out_tops(fig, ax, xs, lw):
    """Cover the top of each cut-off arrow with a background-colored gradient (opaque at the frame,
    transparent lower down), so the shaft fades in and does not look like it starts at the frame."""
    if not xs:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x_per_pt = (x1 - x0) / (ax.get_position().width * fig.get_figwidth() * 72)
    half_width = lw * 0.9 * x_per_pt
    fade = (y1 - y0) * 0.22
    rgb = matplotlib.colors.to_rgb(BACKGROUND)
    alpha = np.linspace(1, 0, 256) ** 1.5  # top row opaque
    gradient = np.dstack([np.full((256, 1), c) for c in rgb] + [alpha[:, None]])
    for x in xs:
        ax.imshow(gradient, extent=(x - half_width, x + half_width, y1 - fade, y1), aspect="auto",
                  interpolation="bicubic", zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def draw_panel(fig, ax, title, records, ylabel, ylim, ystep, groups_to_draw=GROUPS,
               offset=0.24, lw=ARROW_LW):
    (ymin, ymax) = ylim
    yrange = ymax - ymin
    axes_height_pt = ax.get_position().height * fig.get_figheight() * 72
    groups = {(g, c): [] for g in groups_to_draw for c in CATEGORIES}
    for r in records:
        if r["category"] in CATEGORIES and r["variant"] in groups_to_draw:
            groups[(r["variant"], r["category"])].append(r)

    ax.set_facecolor(BACKGROUND)
    faded = []  # x of arrows whose start (0) is above the visible range
    # offset = half the distance between the two arrows in a group
    for g, group in enumerate(groups_to_draw):
        for k, category in enumerate(CATEGORIES):
            rs = groups[(group, category)]
            q = statistics.mean(r["query_score"] for r in rs)
            v = statistics.mean(r["variant_score"] for r in rs)
            x = g + (-offset if k == 0 else offset)
            draw_arrow(ax, x, q, v, COLORS[category], yrange, axes_height_pt, lw)
            if q > ymax:
                faded.append(x)
            ax.plot(x, q, "o", ms=8, color="#52514e", mec=BACKGROUND, mew=1.2, zorder=3)
            change = v - q
            # The change label sits just past the arrowhead (below for a drop, above for an increase).
            y0, va = (v - yrange * 0.025, "top") if change <= 0 else (v + yrange * 0.025, "bottom")
            ax.text(x, y0, f"{'+' if change > 0 else '−'}{abs(change):.2f}", ha="center", va=va,
                    fontsize=10.5, color="#0b0b0b", fontweight="bold")

    if len(groups_to_draw) > 1:
        ax.set_xticks(range(len(groups_to_draw)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups_to_draw], fontsize=12, color="#52514e")
    else:
        ax.set_xticks([])
    margin = 0.5 if len(groups_to_draw) > 1 else offset * 2.2
    ax.set_xlim(-margin, len(groups_to_draw) - 1 + margin)
    ax.set_ylim(ymin, ymax)
    fade_out_tops(fig, ax, faded, lw)
    ticks = [round(ymin + i * ystep, 2) for i in range(round(yrange / ystep) + 1)]
    if ymin < 0:  # standardized axis: ticks on round multiples of the step, with a real minus sign
        ticks = [round(t * ystep, 2) for t in range(math.ceil(ymin / ystep), math.floor(ymax / ystep) + 1)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}".replace("-", "−") for t in ticks])
    else:
        ax.set_yticks(ticks)
    ax.tick_params(axis="y", labelsize=12, labelcolor="#898781")
    ax.tick_params(length=0)
    ax.set_ylabel(ylabel, fontsize=14, color="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title(title, fontsize=16, color="#0b0b0b", pad=12)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pooled", action="store_true", help="merge the 3 benchmarks into one panel")
    mode.add_argument("--by-model", action="store_true", help="merge the benchmarks, one panel per model")
    args = parser.parse_args()
    records = json.loads(INPUT.read_text(encoding="utf-8"))

    if args.pooled:
        # One small panel per ablation type, each with its own y range fitted to its two arrows, so the
        # difference in arrow length is as visible as possible (ranges differ across panels).
        # Every arrow starts at 0 (the original query) so the two arrow lengths are directly comparable.
        standardized = [{**r, "query_score": 0.0, "variant_score": -r["drop"]} for r in standardize(records)]
        fig, axes = plt.subplots(1, len(GROUPS), figsize=(2.6 * len(GROUPS), 6.8), dpi=200)
        fig.subplots_adjust(left=0.1, right=0.995, top=0.83, bottom=0.04, wspace=0.45)
        # Same y span in every panel: room for the largest gap between the two arrowheads, plus labels.
        span = math.ceil(max(tip_gap(standardized, g) for g in GROUPS) * 3.4 * 10) / 10
        for i, (ax, group) in enumerate(zip(axes, GROUPS)):
            ylim, ystep = fitted_ylim(standardized, group, span)
            draw_panel(fig, ax, GROUP_LABELS[group].replace("\n", " "), standardized,
                       "Change vs. Original Query (SD Units)" if i == 0 else "", ylim, ystep,
                       groups_to_draw=[group], offset=0.2, lw=9)
        output = POOLED_OUTPUT
        legend_size = 14
    elif args.by_model:
        standardized = standardize(records)
        fig, axes = plt.subplots(1, len(MODEL_TITLES), figsize=(7.4 * len(MODEL_TITLES), 6.3), dpi=200)
        fig.subplots_adjust(left=0.045, right=0.995, top=0.83, bottom=0.13, wspace=0.08)
        for i, (ax, (model, title)) in enumerate(zip(axes, MODEL_TITLES.items())):
            draw_panel(fig, ax, title, [r for r in standardized if r["model"] == model],
                       POOLED[1] if i == 0 else "", BY_MODEL_YLIM, 0.25)
        output = BY_MODEL_OUTPUT
        legend_size = 16
    else:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.4 * len(DATASETS), 6.3), dpi=200)
        fig.subplots_adjust(left=0.04, right=0.995, top=0.83, bottom=0.13, wspace=0.08)
        for ax, dataset in zip(axes, DATASETS):
            ylabel, ylim, ystep = DATASETS[dataset]
            draw_panel(fig, ax, dataset, [r for r in records if r["dataset"] == dataset], ylabel, ylim, ystep)
        output = OUTPUT
        legend_size = 16
    fig.patch.set_facecolor(BACKGROUND)

    scale = legend_size / 16
    handles = ([] if args.pooled else
               [Line2D([], [], marker="o", ls="", color="#52514e", ms=11 * scale, label="Original Query")]) + \
              [Line2D([], [], color=COLORS[c], lw=ARROW_LW * 1.4 * scale, label=CATEGORY_LABELS[c])
               for c in CATEGORIES]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False,
               fontsize=legend_size, columnspacing=1.6, handlelength=1.5, handletextpad=0.5)
    fig.savefig(output, facecolor=fig.get_facecolor())
    print(f"saved {output}")


if __name__ == "__main__":
    main()
