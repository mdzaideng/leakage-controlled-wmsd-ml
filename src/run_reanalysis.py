from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from scipy.stats import binomtest, t
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 20260904
LABELS = ["Low", "Medium", "High"]


@dataclass(frozen=True)
class ModelSpec:
    estimator: object
    grid: dict[str, list]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if len(df) != 60:
        raise ValueError(f"Expected 60 final participants; found {len(df)}")
    expected_counts = {"Low": 6, "Medium": 37, "High": 17}
    counts = df["Risk Class"].value_counts().to_dict()
    if counts != expected_counts:
        raise ValueError(f"Unexpected class counts: {counts}")
    if df["Participant ID"].duplicated().any():
        raise ValueError("Participant IDs are not unique")

    for col in ["Medical History", "Alcohol Consumption", "Physical Exercise"]:
        df[col] = df[col].fillna("None reported")
    df["RULA_Mean"] = (df["RULA_Left"] + df["RULA_Right"]) / 2.0
    df["RULA_Asymmetry"] = (df["RULA_Left"] - df["RULA_Right"]).abs()
    return df


FEATURE_SETS = {
    "postural": ["RULA_Mean", "RULA_Asymmetry"],
    "nonpostural": [
        "Age", "Gender", "BMI", "Medical History", "Smoking Habits",
        "Alcohol Consumption", "Physical Exercise", "Job Tenure",
        "Working Days", "Work Duration", "Work Breaks", "Table Height",
        "Sitting Height",
    ],
}
FEATURE_SETS["combined"] = FEATURE_SETS["postural"] + FEATURE_SETS["nonpostural"]


def preprocessing(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    # Support both legacy object-backed strings and pandas' newer dedicated
    # string extension dtype. This keeps preprocessing stable across pandas
    # versions without changing the predictor specification.
    cats = [
        c
        for c in features
        if is_object_dtype(df[c].dtype) or is_string_dtype(df[c].dtype)
    ]
    nums = [c for c in features if c not in cats]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, nums),
        ("categorical", cat_pipe, cats),
    ], sparse_threshold=0)


def model_specs() -> dict[str, ModelSpec]:
    return {
        "LR": ModelSpec(
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "NB": ModelSpec(GaussianNB(), {"model__var_smoothing": [1e-11, 1e-9, 1e-7]}),
        "kNN": ModelSpec(
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]},
        ),
        "DT": ModelSpec(
            DecisionTreeClassifier(random_state=SEED),
            {
                "model__max_depth": [2, 3, 5, None],
                "model__min_samples_leaf": [2, 4, 6],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "RF": ModelSpec(
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
            {
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "SVM": ModelSpec(
            SVC(probability=True, random_state=SEED),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["linear", "rbf"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "AB": ModelSpec(
            AdaBoostClassifier(random_state=SEED),
            {"model__n_estimators": [25, 50, 100], "model__learning_rate": [0.05, 0.2, 0.5]},
        ),
        "GBT": ModelSpec(
            GradientBoostingClassifier(random_state=SEED),
            {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
                "model__min_samples_leaf": [2, 4],
            },
        ),
    }


def metric_row(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def align_probabilities(estimator, prob: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(prob), len(LABELS)), dtype=float)
    classes = list(estimator.classes_)
    for j, label in enumerate(LABELS):
        if label in classes:
            aligned[:, j] = prob[:, classes.index(label)]
    return aligned


def nested_evaluate(df: pd.DataFrame, feature_set: str, specs: dict[str, ModelSpec], repeats: int):
    features = FEATURE_SETS[feature_set]
    X = df[features]
    y = df["Risk Class"].astype(str).to_numpy()
    ids = df["Participant ID"].astype(str).to_numpy()
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)

    records = []
    predictions = []
    tuning = []
    start = time.perf_counter()
    for split_i, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat = split_i // 5
        fold = split_i % 5
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + split_i + 1)
        for model_name, spec in specs.items():
            pipe = Pipeline([
                ("prep", preprocessing(df, features)),
                ("model", clone(spec.estimator)),
            ])
            search = GridSearchCV(
                pipe,
                spec.grid,
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X.iloc[train_idx], y[train_idx])
            pred = search.predict(X.iloc[test_idx])
            prob = align_probabilities(search.best_estimator_.named_steps["model"], search.predict_proba(X.iloc[test_idx]))
            row = {"model": model_name, "feature_set": feature_set, "repeat": repeat, "fold": fold}
            row.update(metric_row(y[test_idx], pred))
            records.append(row)
            tuning.append({
                "model": model_name,
                "feature_set": feature_set,
                "repeat": repeat,
                "fold": fold,
                "inner_best_balanced_accuracy": search.best_score_,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })
            for local_i, global_i in enumerate(test_idx):
                predictions.append({
                    "participant_id": ids[global_i],
                    "true_class": y[global_i],
                    "model": model_name,
                    "feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "predicted_class": pred[local_i],
                    **{f"p_{label.lower()}": prob[local_i, j] for j, label in enumerate(LABELS)},
                })
    elapsed = time.perf_counter() - start
    return pd.DataFrame(records), pd.DataFrame(predictions), pd.DataFrame(tuning), elapsed


def repeat_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature_set, repeat), g in pred.groupby(["model", "feature_set", "repeat"]):
        row = {"model": model, "feature_set": feature_set, "repeat": repeat}
        row.update(metric_row(g["true_class"], g["predicted_class"]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_repeats(rep: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc"]
    rows = []
    for (model, feature_set), g in rep.groupby(["model", "feature_set"]):
        row = {"model": model, "feature_set": feature_set, "n_repeats": len(g)}
        for m in metrics:
            vals = g[m].to_numpy(float)
            mean = vals.mean()
            se = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else np.nan
            half = t.ppf(0.975, len(vals) - 1) * se if len(vals) > 1 else np.nan
            row[f"{m}_mean"] = mean
            row[f"{m}_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[f"{m}_ci_low"] = mean - half
            row[f"{m}_ci_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False)


def consensus_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    probs = ["p_low", "p_medium", "p_high"]
    base = pred.groupby(["model", "feature_set", "participant_id", "true_class"], as_index=False)[probs].mean()
    base["predicted_class"] = np.array(LABELS)[base[probs].to_numpy().argmax(axis=1)]
    return base


def class_metrics(consensus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature_set), g in consensus.groupby(["model", "feature_set"]):
        cm = confusion_matrix(g["true_class"], g["predicted_class"], labels=LABELS)
        precision, recall, f1, support = precision_recall_fscore_support(
            g["true_class"], g["predicted_class"], labels=LABELS, zero_division=0
        )
        for i, label in enumerate(LABELS):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            rows.append({
                "model": model, "feature_set": feature_set, "class": label,
                "support": int(support[i]), "precision": precision[i],
                "sensitivity": recall[i], "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "f1": f1[i],
            })
    return pd.DataFrame(rows)


def holm_adjust(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals), dtype=float)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvals[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def mcnemar_table(consensus: pd.DataFrame, best_model: str) -> pd.DataFrame:
    wide = consensus.pivot(index=["participant_id", "true_class"], columns="model", values="predicted_class")
    true = wide.index.get_level_values("true_class").to_numpy()
    best_correct = wide[best_model].to_numpy() == true
    rows = []
    for other in sorted(c for c in wide.columns if c != best_model):
        other_correct = wide[other].to_numpy() == true
        b = int(np.sum(best_correct & ~other_correct))
        c = int(np.sum(~best_correct & other_correct))
        p = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue if b + c else 1.0
        rows.append({"best_model": best_model, "other_model": other, "best_only_correct": b, "other_only_correct": c, "p_raw": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"].tolist())
    return out


def plot_model_summary(summary: pd.DataFrame, output: Path):
    s = summary.sort_values("balanced_accuracy_mean")
    err_low = s["balanced_accuracy_mean"] - s["balanced_accuracy_ci_low"]
    err_high = s["balanced_accuracy_ci_high"] - s["balanced_accuracy_mean"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(s["model"], s["balanced_accuracy_mean"], color="#356E9A")
    ax.errorbar(s["balanced_accuracy_mean"], s["model"], xerr=np.vstack([err_low, err_high]), fmt="none", ecolor="black", capsize=3)
    ax.axvline(1 / 3, color="#A43C3C", linestyle="--", linewidth=1, label="Chance balanced accuracy")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Repeated nested-CV balanced accuracy")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_data(args.data)
    specs = model_specs()
    fold, pred, tuning, elapsed = nested_evaluate(df, "combined", specs, args.repeats)
    rep = repeat_metrics(pred)
    summary = summarize_repeats(rep)
    consensus = consensus_predictions(pred)
    per_class = class_metrics(consensus)
    best = summary.iloc[0]["model"]
    paired = mcnemar_table(consensus, best)

    # Prespecified feature-set ablations for the selected algorithm.
    ab_fold = []
    ab_pred = []
    ab_tuning = []
    ab_elapsed = 0.0
    for feature_set in ["postural", "nonpostural"]:
        f, p, tune, sec = nested_evaluate(df, feature_set, {best: specs[best]}, args.repeats)
        ab_fold.append(f); ab_pred.append(p); ab_tuning.append(tune); ab_elapsed += sec
    all_fold = pd.concat([fold] + ab_fold, ignore_index=True)
    all_pred = pd.concat([pred] + ab_pred, ignore_index=True)
    all_tuning = pd.concat([tuning] + ab_tuning, ignore_index=True)
    ab_rep = repeat_metrics(all_pred[all_pred["model"] == best])
    ab_summary = summarize_repeats(ab_rep)

    all_fold.to_csv(args.output / "fold_metrics.csv", index=False)
    rep.to_csv(args.output / "repeat_metrics_primary.csv", index=False)
    summary.to_csv(args.output / "model_summary_primary.csv", index=False)
    all_pred.to_csv(args.output / "out_of_fold_predictions.csv", index=False)
    all_tuning.to_csv(args.output / "inner_tuning_choices.csv", index=False)
    consensus.to_csv(args.output / "consensus_predictions_primary.csv", index=False)
    per_class.to_csv(args.output / "class_metrics_primary.csv", index=False)
    paired.to_csv(args.output / "paired_mcnemar_holm.csv", index=False)
    ab_summary.to_csv(args.output / "feature_ablation_summary.csv", index=False)
    plot_model_summary(summary, args.output / "model_balanced_accuracy.png")
    joblib.dump({"seed": SEED, "labels": LABELS, "features": FEATURE_SETS, "models": list(specs)}, args.output / "analysis_manifest.joblib")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "seed": SEED,
        "participants": len(df),
        "class_counts": df["Risk Class"].value_counts().to_dict(),
        "outer_folds": 5,
        "outer_repeats": args.repeats,
        "inner_folds": 3,
        "selected_best_model": best,
        "primary_runtime_seconds": elapsed,
        "ablation_runtime_seconds": ab_elapsed,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nBest model: {best}")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
