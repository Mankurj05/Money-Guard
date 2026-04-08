from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODEL_PATH
from .features import build_features


class ArtifactNotFoundError(FileNotFoundError):
    pass


def load_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise ArtifactNotFoundError(
            "Model artifact not found. Train first with: python scripts/train_model.py"
        )
    return joblib.load(model_path)


def predict_from_record(record: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    if artifact is None:
        artifact = load_artifact()

    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))

    input_df = pd.DataFrame([record])
    features = build_features(input_df)

    proba = float(model.predict_proba(features)[0][1])
    prediction = int(proba >= threshold)

    return {
        "is_fraud": prediction,
        "fraud_probability": round(proba, 6),
        "threshold": threshold,
    }
