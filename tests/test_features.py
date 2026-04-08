import pandas as pd

from fraud_detection.features import build_features


def test_build_features_returns_expected_columns() -> None:
    input_df = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 1000.0,
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 4000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1000.0,
            }
        ]
    )

    result = build_features(input_df)

    expected = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "orig_balance_error",
        "dest_balance_error",
        "orig_zero_after",
        "dest_zero_before",
    }

    assert set(result.columns) == expected
    assert result.loc[0, "orig_balance_error"] == 0.0
    assert result.loc[0, "dest_balance_error"] == 0.0
