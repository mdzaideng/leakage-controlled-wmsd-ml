from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures" / "manuscript"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
})


def box(ax, xy, wh, title, body, color, title_size=10.5, body_size=8.8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor="#2F3B45", linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", weight="bold", fontsize=title_size, linespacing=1.05)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=body_size, linespacing=1.12)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.15, color="#44546A",
        linestyle="--" if dashed else "-",
        shrinkA=2, shrinkB=2,
    ))


def figure1():
    fig, ax = plt.subplots(figsize=(11.0, 5.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    colors = ["#D9EAF7", "#E2F0D9", "#FFF2CC", "#FCE4D6"]
    xs = [0.035, 0.285, 0.535, 0.785]
    titles = ["1. Data acquisition", "2. Feature construction", "3. Leakage-controlled\nevaluation", "4. Evidence output"]
    bodies = [
        "Bilateral 10-min video\nWorker and workplace data\nmNMQ/SEP responses",
        "Frame-level RULA\nTime-weighted posture exposure\nPrespecified non-postural features",
        "Repeated nested stratified\n5-fold cross-validation\nFold-contained preprocessing/tuning\nNo outcome-derived predictors",
        "Held-out metrics and uncertainty\nFeature-set and sensitivity analyses\nInternal decision-support evidence",
    ]
    for x, title, body, color in zip(xs, titles, bodies, colors):
        box(ax, (x, 0.35), (0.18, 0.32), title, body, color, title_size=9.2, body_size=7.7)
    for i in range(3):
        arrow(ax, (xs[i] + 0.18, 0.51), (xs[i + 1], 0.51))
    ax.text(0.5, 0.83, "Corrected WMSD risk-classification study workflow", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.16, "External validation and deployment testing remain future steps", ha="center", va="center", fontsize=9.5, style="italic", color="#595959")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "Figure_01_corrected_workflow.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure2():
    fig, ax = plt.subplots(figsize=(9.0, 7.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.955, "Data preparation and worker-level record construction", ha="center", va="center", fontsize=14, weight="bold")

    box(ax, (0.05, 0.75), (0.38, 0.14), "Postural stream", "Left/right 10-min videos\nKinovea tracking with manual correction", "#D9EAF7")
    box(ax, (0.57, 0.75), (0.38, 0.14), "Worker/workplace stream", "Demographic, physiological, lifestyle\nand work-organisation variables", "#E2F0D9")
    box(ax, (0.05, 0.53), (0.38, 0.14), "Frame-level posture processing", "Upper/lower arm, neck and trunk angles\nplus recorded RULA component inputs", "#D9EAF7")
    box(ax, (0.57, 0.53), (0.38, 0.14), "Outcome construction", "mNMQ symptoms and SEP assessment\nMaximum regional MSD Index and risk class", "#E2F0D9")
    box(ax, (0.05, 0.31), (0.38, 0.14), "Dynamic exposure summary", "Frame-level RULA lookup\nLeft/right time-weighted RULA scores", "#FFF2CC")
    box(ax, (0.31, 0.08), (0.38, 0.14), "Deidentified analytical record", "Prespecified posture + worker/workplace predictors\nThree-class outcome: Low, Medium or High", "#FCE4D6")

    arrow(ax, (0.24, 0.75), (0.24, 0.67))
    arrow(ax, (0.76, 0.75), (0.76, 0.67))
    arrow(ax, (0.24, 0.53), (0.24, 0.45))
    arrow(ax, (0.24, 0.31), (0.42, 0.22))
    arrow(ax, (0.76, 0.53), (0.58, 0.22))
    arrow(ax, (0.76, 0.75), (0.76, 0.67))
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "Figure_02_data_pipeline.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rula_distribution(filename, side, counts):
    labels = ["≤3.00", ">3.00–4.00", ">4.00–5.00", ">5.00–6.00", ">6.00"]
    colors = ["#70AD47", "#4DB6AC", "#5B9BD5", "#F4A261", "#E76F51"]
    fig, ax = plt.subplots(figsize=(8.1, 4.85))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, width=0.58, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Analytical records")
    ax.set_xlabel("Weighted RULA interval")
    ax.set_title(f"{side}-side weighted RULA distribution", pad=10, weight="bold")
    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0, 21, 2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45, str(value), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    rula_distribution("Figure_7_RULA_right_corrected.png", "Right", [4, 12, 18, 18, 8])
    rula_distribution("Figure_8_RULA_left_corrected.png", "Left", [2, 12, 18, 15, 13])
