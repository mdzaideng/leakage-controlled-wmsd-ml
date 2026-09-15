from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from run_reanalysis import (
    FEATURE_SETS,
    LABELS,
    SEED,
    align_probabilities,
    consensus_predictions,
    load_data,
    metric_row,
    preprocessing,
)


class BalancedGradientBoosting(BaseEstimator, ClassifierMixin):
    """Gradient boosting with class-balanced weights computed inside each fit."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 1,
        min_samples_leaf: int = 2,
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        weights = compute_sample_weight(class_weight="balanced", y=y)
        self.model_.fit(X, y, sample_weight=weights)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


GRID = {
    "model__n_estimators": [50, 100],
    "model__learning_rate": [0.03, 0.1],
    "model__max_depth": [1, 2],
    "model__min_samples_leaf": [2, 4],
}


def nested_gbt(df: pd.DataFrame, features: list[str], weighted: bool, repeats: int):
    X = df[features]
    y = df["Risk Class"].astype(str).to_numpy()
    ids = df["Participant ID"].astype(str).to_numpy()
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    records = []
    predictions = []
    tuning = []
    for split_i, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat = split_i // 5
        fold = split_i % 5
        estimator = (
            BalancedGradientBoosting(random_state=SEED + split_i)
            if weighted
            else GradientBoostingClassifier(random_state=SEED + split_i)
        )
        pipe = Pipeline([
            ("prep", preprocessing(df, features)),
            ("model", estimator),
        ])
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + split_i + 1)
        search = GridSearchCV(
            pipe,
            GRID,
            scoring="balanced_accuracy",
            cv=inner,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X.iloc[train_idx], y[train_idx])
        pred = search.predict(X.iloc[test_idx])
        model = search.best_estimator_.named_steps["model"]
        prob = align_probabilities(model, search.predict_proba(X.iloc[test_idx]))
        row = {"repeat": repeat, "fold": fold, "weighted": weighted}
        row.update(metric_row(y[test_idx], pred))
        records.append(row)
        tuning.append({
            "repeat": repeat,
            "fold": fold,
            "weighted": weighted,
            "inner_best_balanced_accuracy": search.best_score_,
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })
        for local_i, global_i in enumerate(test_idx):
            predictions.append({
                "participant_id": ids[global_i],
                "true_class": y[global_i],
                "repeat": repeat,
                "fold": fold,
                "weighted": weighted,
                "predicted_class": pred[local_i],
                **{f"p_{label.lower()}": prob[local_i, j] for j, label in enumerate(LABELS)},
            })
    return pd.DataFrame(records), pd.DataFrame(predictions), pd.DataFrame(tuning)


def repeat_summary(pred: pd.DataFrame, group_fields: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(group_fields + ["repeat"]):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_fields + ["repeat"], keys))
        row.update(metric_row(g["true_class"], g["predicted_class"]))
        rows.append(row)
    result = pd.DataFrame(rows)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mcc"]
    summary = result.groupby(group_fields)[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return result, summary.reset_index()


def stratified_bootstrap_consensus(consensus: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    metrics = {"accuracy": [], "balanced_accuracy": [], "macro_f1": [], "mcc": []}
    groups = [g.reset_index(drop=True) for _, g in consensus.groupby("true_class", sort=False)]
    for _ in range(n_boot):
        sample = pd.concat(
            [g.iloc[rng.integers(0, len(g), len(g))] for g in groups],
            ignore_index=True,
        )
        y = sample["true_class"]
        p = sample["predicted_class"]
        metrics["accuracy"].append(accuracy_score(y, p))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y, p))
        metrics["macro_f1"].append(f1_score(y, p, average="macro", zero_division=0))
        metrics["mcc"].append(matthews_corrcoef(y, p))
    rows = []
    for name, values in metrics.items():
        arr = np.asarray(values)
        rows.append({
            "metric": name,
            "estimate": {
                "accuracy": accuracy_score(consensus.true_class, consensus.predicted_class),
                "balanced_accuracy": balanced_accuracy_score(consensus.true_class, consensus.predicted_class),
                "macro_f1": f1_score(consensus.true_class, consensus.predicted_class, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(consensus.true_class, consensus.predicted_class),
            }[name],
            "bootstrap_ci_low": np.quantile(arr, 0.025),
            "bootstrap_ci_high": np.quantile(arr, 0.975),
        })
    return pd.DataFrame(rows)


def clustered_bootstrap_repeated_oof(predictions: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    """Bootstrap workers within outcome strata, retaining all repeated OOF predictions."""
    rng = np.random.default_rng(SEED + 1)
    participant_table = predictions[["participant_id", "true_class"]].drop_duplicates()
    strata = {
        label: participant_table.loc[participant_table["true_class"] == label, "participant_id"].to_numpy()
        for label in LABELS
    }
    observed = metric_row(predictions["true_class"], predictions["predicted_class"])
    label_index = {label: i for i, label in enumerate(LABELS)}
    per_participant = {}
    for pid, g in predictions.groupby("participant_id"):
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        for true, predicted in zip(g["true_class"], g["predicted_class"]):
            cm[label_index[true], label_index[predicted]] += 1
        per_participant[pid] = cm
    boot = np.empty((n_boot, len(LABELS), len(LABELS)), dtype=np.int64)
    for i in range(n_boot):
        boot[i] = sum(
            per_participant[pid]
            for label in LABELS
            for pid in rng.choice(strata[label], size=len(strata[label]), replace=True)
        )
    tp = np.diagonal(boot, axis1=1, axis2=2)
    true_totals = boot.sum(axis=2)
    pred_totals = boot.sum(axis=1)
    total = boot.sum(axis=(1, 2))
    accuracy = tp.sum(axis=1) / total
    balanced_accuracy = np.mean(tp / true_totals, axis=1)
    macro_f1 = np.mean(2 * tp / (true_totals + pred_totals), axis=1)
    numerator = tp.sum(axis=1) * total - np.sum(true_totals * pred_totals, axis=1)
    denominator = np.sqrt(
        (total**2 - np.sum(pred_totals**2, axis=1))
        * (total**2 - np.sum(true_totals**2, axis=1))
    )
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mcc": numerator / denominator,
    }
    return pd.DataFrame([
        {
            "metric": name,
            "estimate": observed[name],
            "bootstrap_ci_low": np.quantile(values, 0.025),
            "bootstrap_ci_high": np.quantile(values, 0.975),
        }
        for name, values in metrics.items()
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    primary = pd.read_csv(args.primary_predictions, dtype={"participant_id": str})
    primary = primary[(primary["model"] == "GBT") & (primary["feature_set"] == "combined")].copy()
    sex = df[["Participant ID", "Gender"]].copy()
    sex["participant_id"] = sex["Participant ID"].astype(str)
    primary = primary.merge(sex[["participant_id", "Gender"]], on="participant_id", validate="many_to_one")
    sex_repeat, sex_summary = repeat_summary(primary, ["Gender"])
    sex_repeat.to_csv(args.output / "sex_stratified_repeat_metrics.csv", index=False)
    sex_summary.to_csv(args.output / "sex_stratified_summary.csv", index=False)

    primary_consensus = consensus_predictions(primary.assign(model="GBT", feature_set="combined"))
    stratified_bootstrap_consensus(primary_consensus, n_boot=10000).to_csv(
        args.output / "consensus_bootstrap_intervals.csv", index=False
    )
    clustered_bootstrap_repeated_oof(primary, n_boot=10000).to_csv(
        args.output / "primary_clustered_bootstrap_intervals.csv", index=False
    )

    sensitivity_sets = {
        "combined_without_exercise": [c for c in FEATURE_SETS["combined"] if c != "Physical Exercise"],
        "combined_without_lifestyle": [
            c for c in FEATURE_SETS["combined"]
            if c not in {"Medical History", "Smoking Habits", "Alcohol Consumption", "Physical Exercise"}
        ],
    }
    sensitivity_predictions = []
    sensitivity_folds = []
    sensitivity_tuning = []
    for name, features in sensitivity_sets.items():
        fold, pred, tuning = nested_gbt(df, features, weighted=False, repeats=args.repeats)
        fold["sensitivity_set"] = name
        pred["sensitivity_set"] = name
        tuning["sensitivity_set"] = name
        sensitivity_folds.append(fold)
        sensitivity_predictions.append(pred)
        sensitivity_tuning.append(tuning)

    for weighted in [False, True]:
        fold, pred, tuning = nested_gbt(df, FEATURE_SETS["combined"], weighted=weighted, repeats=args.repeats)
        fold["sensitivity_set"] = "imbalance_weighting"
        pred["sensitivity_set"] = "imbalance_weighting"
        tuning["sensitivity_set"] = "imbalance_weighting"
        sensitivity_folds.append(fold)
        sensitivity_predictions.append(pred)
        sensitivity_tuning.append(tuning)

    all_pred = pd.concat(sensitivity_predictions, ignore_index=True)
    all_fold = pd.concat(sensitivity_folds, ignore_index=True)
    all_tuning = pd.concat(sensitivity_tuning, ignore_index=True)
    all_fold.to_csv(args.output / "sensitivity_fold_metrics.csv", index=False)
    all_pred.to_csv(args.output / "sensitivity_oof_predictions.csv", index=False)
    all_tuning.to_csv(args.output / "sensitivity_tuning.csv", index=False)

    all_pred["analysis"] = np.where(
        all_pred["sensitivity_set"] == "imbalance_weighting",
        np.where(all_pred["weighted"], "combined_class_weighted", "combined_unweighted_rerun"),
        all_pred["sensitivity_set"],
    )
    _, sens_summary = repeat_summary(all_pred, ["analysis"])
    sens_summary.to_csv(args.output / "sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    main()
