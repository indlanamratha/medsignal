"""MedSignal prediction API: score a new adverse event report for seriousness.

Run locally:  uv run uvicorn medsignal.api.main:app --reload
Docs:         http://localhost:8000/docs
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from medsignal.ml.features import features_from_report

STUDY_DRUGS = {
    "semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "exenatide", "metformin",
    "empagliflozin", "dapagliflozin", "sitagliptin", "insulin glargine", "phentermine",
    "orlistat", "naltrexone/bupropion",
}

logger = logging.getLogger("medsignal.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "models/serious_model.joblib"))


def prediction_log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the trained model, its feature list, and its decision threshold once."""
    return joblib.load(model_path())


class Report(BaseModel):
    age_years: float | None = Field(None, ge=0, le=120, description="Patient age in years")
    weight_kg: float | None = Field(None, gt=0, le=400)
    sex: Literal["M", "F", "U"] = "U"
    reporter_type: Literal["Consumer", "Physician", "Pharmacist",
                           "Other health professional", "Lawyer", "Unknown"] = "Unknown"
    country: str | None = Field(None, min_length=2, max_length=2, description="Two-letter code, e.g. US")
    suspect_drugs: list[str] = Field(..., min_length=1, description="Study drug groups suspected")
    n_concomitant_drugs: int = Field(0, ge=0, le=100)
    reactions: list[str] = Field(..., min_length=1, description="MedDRA preferred terms")

    @field_validator("suspect_drugs")
    @classmethod
    def known_drugs(cls, drugs: list[str]) -> list[str]:
        drugs = [d.strip().lower() for d in drugs]
        unknown = sorted(set(drugs) - STUDY_DRUGS)
        if unknown:
            raise ValueError(f"unknown drug group(s): {unknown}; expected one of {sorted(STUDY_DRUGS)}")
        return drugs

    @field_validator("country")
    @classmethod
    def upper_country(cls, country: str | None) -> str | None:
        return country.upper() if country else None


class Prediction(BaseModel):
    serious_probability: float
    flag_for_review: bool
    threshold: float
    model: str


app = FastAPI(title="MedSignal API", version="1.0.0",
              description="Predicts whether an FDA adverse event report is serious, "
                          "so safety teams can review likely-serious reports first.")


@app.get("/health")
def health() -> dict:
    bundle = load_bundle()
    return {"status": "ok", "model": bundle["model_name"], "threshold": bundle["threshold"],
            "n_features": len(bundle["features"])}


@app.post("/predict", response_model=Prediction)
def predict(report: Report) -> Prediction:
    start = time.perf_counter()
    bundle = load_bundle()
    X = features_from_report(report.model_dump(), bundle["features"])
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    result = Prediction(serious_probability=round(probability, 4),
                        flag_for_review=probability >= bundle["threshold"],
                        threshold=bundle["threshold"], model=bundle["model_name"])
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    log_path = prediction_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "latency_ms": latency_ms,
                            "request": report.model_dump(), **result.model_dump()}) + "\n")
    logger.info("prediction probability=%.3f flag=%s latency_ms=%.1f",
                probability, result.flag_for_review, latency_ms)
    return result
