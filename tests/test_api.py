"""API tests. They use a small stand-in model, so they run anywhere, including CI."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

FEATURES = ["age_years", "n_reactions", "drug_semaglutide", "rx_pancreatitis", "reporter_consumer"]
VALID_REPORT = {
    "age_years": 58, "sex": "F", "reporter_type": "Physician", "country": "US",
    "suspect_drugs": ["semaglutide"], "n_concomitant_drugs": 1,
    "reactions": ["PANCREATITIS", "VOMITING"],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    X = pd.DataFrame(np.random.default_rng(0).random((200, len(FEATURES))), columns=FEATURES)
    y = (X["rx_pancreatitis"] > 0.5).astype(int)
    model = LogisticRegression().fit(X, y)
    joblib.dump({"model": model, "features": FEATURES, "threshold": 0.5, "model_name": "test_model"},
                tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))

    from medsignal.api import main
    main.load_bundle.cache_clear()
    yield TestClient(main.app)
    main.load_bundle.cache_clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability(client, tmp_path):
    response = client.post("/predict", json=VALID_REPORT)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["serious_probability"] <= 1
    assert isinstance(body["flag_for_review"], bool)
    assert (tmp_path / "predictions.jsonl").exists()


@pytest.mark.parametrize("bad_field", [
    {"age_years": 500},
    {"sex": "X"},
    {"suspect_drugs": ["aspirin"]},
    {"reactions": []},
])
def test_invalid_input_is_rejected(client, bad_field):
    response = client.post("/predict", json={**VALID_REPORT, **bad_field})
    assert response.status_code == 422
