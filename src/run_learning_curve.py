from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from run_reanalysis import FEATURE_SETS, SEED, load_data, preprocessing


FRACTIONS = (0.40, 0.55, 0.70, 0.85, 1.00)
MODEL_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.03,
    "max_depth": 1,
    "min_samples_leaf": 2,
}


def stratified_subset(indices: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return a deterministic stratified subset of an outer-training partition."""
    if fraction >= 1.0:
        return indices.copy()
    selected, _ = train_test_split(
        indices,
        train_size=fraction,
        stratify=y[indices],
        random_state=seed,
    )
    return np.asarray(selected)


def run_curve(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    features = FEATURE_SETS["combined"]
    x = df[features]
    y = df["Risk Class"].astype(str).to_numpy()
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    base = Pipeline([
        ("prep", preprocessing(df, features)),
        ("model", GradientBoostingClassifier(random_state=SEED, **MODEL_PARAMETERS)),
    ])
    rows: list[dict[str, float | int]] = []
    for split_index, (train_index, test_index) in enumerate(outer.split(x, y)):
        repeat = split_index // 5
        fold = split_index % 5
        for fraction_index, fraction in enumerate(FRACTIONS):
            subset = stratified_subset(
                train_index,
                y,
                fraction,
                SEED + 1000 * split_index + fraction_index,
            )
            estimator = clone(base)
            estimator.set_params(model__random_state=SEED + split_index)
            estimator.fit(x.iloc[subset], y[subset])
            train_prediction = estimator.predict(x.iloc[subset])
            validation_prediction = estimator.predict(x.iloc[test_index])
            rows.append({
                "repeat": repeat,
                "fold": fold,
                "training_fraction": fraction,
                "training_records": len(subset),
                "low_training_records": int(np.sum(y[subset] == "Low")),
                "medium_training_records": int(np.sum(y[subset] == "Medium")),
                "high_training_records": int(np.sum(y[subset] == "High")),
                "train_balanced_accuracy": balanced_accuracy_score(y[subset], train_prediction),
                "validation_balanced_accuracy": balanced_accuracy_score(y[test_index], validation_prediction),
                "train_macro_f1": f1_score(y[subset], train_prediction, average="macro", zero_division=0),
                "validation_macro_f1": f1_score(y[test_index], validation_prediction, average="macro", zero_division=0),
            })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fraction, group in raw.groupby("training_fraction", sort=True):
        row: dict[str, float | int] = {
            "training_fraction": fraction,
            "training_records_median": int(group["training_records"].median()),
            "n_outer_splits": len(group),
        }
        for metric in (
            "train_balanced_accuracy",
            "validation_balanced_accuracy",
            "train_macro_f1",
            "validation_macro_f1",
        ):
            values = group[metric].to_numpy(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            half_width = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = max(0.0, mean - half_width)
            row[f"{metric}_ci_high"] = min(1.0, mean + half_width)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curve(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    x = summary["training_records_median"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), sharey=True)
    panels = [
        ("balanced_accuracy", "Balanced accuracy", "(a) Balanced accuracy"),
        ("macro_f1", "Macro-F1", "(b) Macro-F1"),
    ]
    colors = {"Training": "#1F4E79", "Outer validation": "#C55A11"}
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for prefix, label in (("train", "Training"), ("validation", "Outer validation")):
            mean = summary[f"{prefix}_{metric}_mean"].to_numpy()
            low = summary[f"{prefix}_{metric}_ci_low"].to_numpy()
            high = summary[f"{prefix}_{metric}_ci_high"].to_numpy()
            ax.plot(x, mean, marker="o", linewidth=1.8, markersize=4.5, color=colors[label], label=label)
            ax.fill_between(x, low, high, color=colors[label], alpha=0.14, linewidth=0)
        ax.axhline(1 / 3, color="#666666", linestyle="--", linewidth=0.9, label="Chance" if metric == "balanced_accuracy" else None)
        ax.set_title(title, pad=8)
        ax.set_xlabel("Training records (median)")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_xticks(x)
        ax.set_ylim(0.25, 1.02)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic GBT learning-curve analysis")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    raw = run_curve(df, repeats=args.repeats)
    summary = summarize(raw)
    raw.to_csv(args.output / "learning_curve_split_metrics.csv", index=False)
    summary.to_csv(args.output / "learning_curve_summary.csv", index=False)
    plot_curve(summary, args.output / "learning_curve_gbt.png")
    (args.output / "learning_curve_metadata.json").write_text(json.dumps({
        "seed": SEED,
        "outer_folds": 5,
        "outer_repeats": args.repeats,
        "training_fractions": FRACTIONS,
        "model": "GradientBoostingClassifier",
        "model_parameters": MODEL_PARAMETERS,
        "status": "diagnostic; not used for model selection or primary performance estimation",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
