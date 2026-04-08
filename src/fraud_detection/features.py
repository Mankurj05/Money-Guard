from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "orig_zero_after",
    "dest_zero_before",
]

CATEGORICAL_FEATURES = ["type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["type"] = data["type"].astype(str).str.upper()
    data["orig_balance_error"] = data["oldbalanceOrg"] - data["newbalanceOrig"] - data["amount"]
    data["dest_balance_error"] = data["newbalanceDest"] - data["oldbalanceDest"] - data["amount"]
    data["orig_zero_after"] = (data["newbalanceOrig"] == 0).astype(int)
    data["dest_zero_before"] = (data["oldbalanceDest"] == 0).astype(int)

    return data[FEATURE_COLUMNS]
