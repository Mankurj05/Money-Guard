from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import DEFAULT_DATA_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN
from .data import load_dataset
from .features import build_features
from .pipeline import build_pipeline


@dataclass
class TrainingMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    train_rows: int
    valid_rows: int


def select_threshold(y_true: pd.Series, y_prob: pd.Series) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_f1 = -1.0
    best_threshold = 0.5

    for index, threshold in enumerate(thresholds):
        precision_value = precision[index]
        recall_value = recall[index]
        denominator = precision_value + recall_value
        if denominator == 0:
            continue
        f1 = 2 * (precision_value * recall_value) / denominator
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def train_model(data_path: Path, sample_frac: float | None, sample_rows: int | None) -> TrainingMetrics:
    df = load_dataset(data_path)

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=RANDOM_STATE)

    X = build_features(df)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_valid)[:, 1]
    threshold = select_threshold(y_valid, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = TrainingMetrics(
        roc_auc=float(roc_auc_score(y_valid, y_prob)),
        pr_auc=float(average_precision_score(y_valid, y_prob)),
        precision=float(precision_score(y_valid, y_pred, zero_division=0)),
        recall=float(recall_score(y_valid, y_pred, zero_division=0)),
        f1=float(f1_score(y_valid, y_pred, zero_division=0)),
        threshold=float(threshold),
        train_rows=int(len(X_train)),
        valid_rows=int(len(X_valid)),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": pipeline,
        "threshold": metrics.threshold,
        "features_version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    return metrics


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction of rows to sample, e.g. 0.2",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=500000,
        help="Optional max number of rows to sample",
    )

    args = parser.parse_args()
    metrics = train_model(args.data_path, args.sample_frac, args.sample_rows)
    print("Training complete")
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    cli_main()
