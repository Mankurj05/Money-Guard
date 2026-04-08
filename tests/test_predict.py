from fraud_detection.predict import predict_from_record


class DummyModel:
    def predict_proba(self, _features):
        return [[0.2, 0.8]]


def test_predict_from_record_contract() -> None:
    artifact = {"model": DummyModel(), "threshold": 0.5}
    record = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 100.0,
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
    }

    result = predict_from_record(record, artifact=artifact)

    assert result["is_fraud"] == 1
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["threshold"] == 0.5
