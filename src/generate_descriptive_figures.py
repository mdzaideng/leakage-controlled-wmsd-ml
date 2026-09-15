from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INTERVALS = [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf]
INTERVAL_LABELS = ["≤3.00", ">3.00–4.00", ">4.00–5.00", ">5.00–6.00", ">6.00"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 600,
    })


def save(fig, output: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output / filename, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pain_figure(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["figure"] == "pain_prevalence"].copy()
    data["percent"] = 100 * data["count"] / data["denominator"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(np.arange(len(data)), data["percent"], color="#347793")
    ax.set_xticks(np.arange(len(data)), data["category"], rotation=25, ha="right")
    ax.set_ylabel("Records reporting pain (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, data["percent"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9, weight="bold")
    save(fig, output, "Figure_06_pain_prevalence.png")


def rula_figure(data: pd.DataFrame, column: str, side: str, output: Path, filename: str) -> None:
    groups = pd.cut(data[column], bins=INTERVALS, labels=INTERVAL_LABELS, include_lowest=True, right=True)
    counts = groups.value_counts(sort=False).to_numpy()
    colors = ["#70AD47", "#4DB6AC", "#5B9BD5", "#F4A261", "#E76F51"]
    fig, ax = plt.subplots(figsize=(8.1, 4.85))
    x = np.arange(len(INTERVAL_LABELS))
    bars = ax.bar(x, counts, width=0.58, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(x, INTERVAL_LABELS)
    ax.set_ylabel("Analytical records"); ax.set_xlabel("Weighted RULA interval")
    ax.set_title(f"{side}-side weighted RULA distribution", weight="bold")
    ax.set_ylim(0, 20); ax.set_yticks(np.arange(0, 21, 2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.45, str(value), ha="center", fontsize=10)
    save(fig, output, filename)


def class_figure(data: pd.DataFrame, output: Path) -> None:
    order = ["Low", "Medium", "High"]
    counts = data["Risk Class"].value_counts().reindex(order).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bars = ax.bar(order, counts, color="#347793", width=0.5)
    ax.set_ylabel("Analytical records"); ax.set_xlabel("Risk class (SEP-based MSD Index)")
    ax.set_ylim(0, 45); ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.8, f"{value} ({100*value/len(data):.1f}%)", ha="center", fontsize=9, weight="bold")
    save(fig, output, "Figure_09_risk_class_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary-counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    style()
    data = pd.read_excel(args.data)
    summary = pd.read_csv(args.summary_counts)
    pain_figure(summary, args.output)
    rula_figure(data, "RULA_Right", "Right", args.output, "Figure_07_RULA_right.png")
    rula_figure(data, "RULA_Left", "Left", args.output, "Figure_08_RULA_left.png")
    class_figure(data, args.output)


if __name__ == "__main__":
    main()
