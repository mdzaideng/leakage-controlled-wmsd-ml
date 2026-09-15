from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from run_reanalysis import FEATURE_SETS, SEED, load_data, preprocessing


TABLE_A = np.array([
    [[[1, 2], [2, 2], [2, 3], [3, 3]], [[2, 2], [2, 2], [3, 3], [3, 3]], [[2, 3], [3, 3], [3, 3], [4, 4]]],
    [[[2, 3], [3, 3], [3, 4], [4, 4]], [[3, 3], [3, 3], [3, 4], [4, 4]], [[3, 4], [4, 4], [4, 4], [5, 5]]],
    [[[3, 3], [4, 4], [4, 4], [5, 5]], [[3, 4], [4, 4], [4, 4], [5, 5]], [[4, 4], [4, 5], [5, 5], [5, 5]]],
    [[[4, 4], [4, 4], [4, 5], [5, 5]], [[4, 4], [4, 4], [4, 5], [5, 6]], [[3, 4], [5, 5], [5, 6], [6, 6]]],
    [[[1, 5], [5, 6], [6, 6], [6, 6]], [[2, 5], [6, 6], [6, 7], [7, 7]], [[3, 6], [6, 7], [7, 7], [7, 8]]],
    [[[1, 7], [7, 8], [8, 8], [8, 9]], [[2, 8], [8, 9], [9, 9], [9, 9]], [[3, 9], [9, 9], [9, 9], [9, 9]]],
])
TABLE_B = np.array([
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
])
TABLE_C = np.array([
    [1,2,3,4,5,6,7,7,7,7,7], [2,2,3,4,4,5,5,7,7,7,7],
    [3,3,3,4,4,5,6,7,7,7,7], [3,3,3,4,5,6,6,7,7,7,7],
    [4,4,4,5,6,7,7,7,7,7,7], [4,4,5,6,6,7,7,7,7,7,7],
    [5,5,6,6,7,7,7,7,7,7,7], [5,5,6,7,7,7,7,7,7,7,7],
    [7,7,7,7,7,7,7,7,7,7,7], [7,7,7,7,7,7,7,7,7,7,7],
    [7,7,7,7,7,7,7,7,7,7,7],
])

REQUIRED = [
    "upper_arm(1-6)", "lower_arm(1-3)", "wrist(1-4)", "wrist_twist(1-2)",
    "muscle_use_a(0-1)", "load_a(0-3)", "neck(1-6)", "trunk(1-6)",
    "legs(1-2)", "muscle_use_b(0-1)", "load_b(0-3)",
]


def score_frames(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing pre-scored RULA columns: {missing}")
    a = TABLE_A[
        df["upper_arm(1-6)"].to_numpy(int)-1,
        df["lower_arm(1-3)"].to_numpy(int)-1,
        df["wrist(1-4)"].to_numpy(int)-1,
        df["wrist_twist(1-2)"].to_numpy(int)-1,
    ]
    a = np.clip(a + df["muscle_use_a(0-1)"].to_numpy(int) + df["load_a(0-3)"].to_numpy(int), 1, 11)
    b = TABLE_B[
        df["neck(1-6)"].to_numpy(int)-1,
        df["trunk(1-6)"].to_numpy(int)-1,
        df["legs(1-2)"].to_numpy(int)-1,
    ]
    b = np.clip(b + df["muscle_use_b(0-1)"].to_numpy(int) + df["load_b(0-3)"].to_numpy(int), 1, 11)
    scores = TABLE_C[a-1, b-1]
    return scores, float(scores.mean())


def timed(callable_, repeats: int) -> tuple[object, list[float]]:
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        values.append(time.perf_counter() - start)
    return result, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--video-loading-seconds", type=float)
    parser.add_argument("--kinovea-tracking-seconds", type=float)
    parser.add_argument("--manual-correction-seconds", type=float)
    parser.add_argument("--angle-export-seconds", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    frames, load_times = timed(lambda: pd.read_csv(args.frames), args.repeats)
    (scores, weighted), score_times = timed(lambda: score_frames(frames), args.repeats)

    study = load_data(args.data)
    features = FEATURE_SETS["combined"]
    model = Pipeline([
        ("prep", preprocessing(study, features)),
        ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, min_samples_leaf=2, random_state=SEED)),
    ])
    fit_start = time.perf_counter()
    model.fit(study[features], study["Risk Class"])
    fit_seconds = time.perf_counter() - fit_start
    one = study[features].iloc[[0]]
    _, prediction_times = timed(lambda: model.predict_proba(one), max(args.repeats, 100))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    computational = statistics.median(load_times) + statistics.median(score_times) + statistics.median(prediction_times)
    manual_values = [args.video_loading_seconds, args.kinovea_tracking_seconds, args.manual_correction_seconds, args.angle_export_seconds]
    complete = computational + sum(x for x in manual_values if x is not None) if all(x is not None for x in manual_values) else None
    duration_seconds = 600.0
    report = {
        "classification": "offline/batch processing",
        "benchmark_file": str(args.frames),
        "frames": int(len(frames)),
        "nominal_video_seconds": duration_seconds,
        "weighted_rula": weighted,
        "csv_load_seconds_median": statistics.median(load_times),
        "rula_scoring_and_aggregation_seconds_median": statistics.median(score_times),
        "rula_processing_frames_per_second": len(frames) / statistics.median(score_times),
        "final_model_fit_seconds_not_deployment_inference": fit_seconds,
        "single_worker_prediction_seconds_median": statistics.median(prediction_times),
        "measured_python_stage_seconds": computational,
        "video_loading_seconds_author_entered": args.video_loading_seconds,
        "kinovea_tracking_seconds_author_entered": args.kinovea_tracking_seconds,
        "manual_correction_seconds_author_entered": args.manual_correction_seconds,
        "angle_export_seconds_author_entered": args.angle_export_seconds,
        "complete_workflow_seconds": complete,
        "peak_python_tracemalloc_bytes": int(peak_bytes),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "note": "Run on the study computer. tracemalloc reports Python allocations, not total system/GPU memory. Kinovea/manual stages require stopwatch measurements.",
    }
    (args.output / "computational_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([report]).to_csv(args.output / "computational_benchmark.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
