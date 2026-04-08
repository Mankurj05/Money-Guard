from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import ArtifactNotFoundError, load_artifact, predict_from_record

app = FastAPI(title="Money Guard API", version="1.0.0")


class PredictionRequest(BaseModel):
    step: int = Field(ge=1)
    type: str
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    threshold: float


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _ = load_artifact()
        return {"status": "ok", "model_loaded": True}
    except ArtifactNotFoundError:
        return {"status": "ok", "model_loaded": False}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = predict_from_record(payload.model_dump())
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**result)
