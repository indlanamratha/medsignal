"""Create a small stand-in model so CI can build and test the Docker image.

The real model is trained on local FAERS data, which is not stored in GitHub.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["age_years", "n_reactions", "drug_semaglutide", "rx_pancreatitis", "reporter_consumer"]

X = pd.DataFrame(np.random.default_rng(0).random((200, len(FEATURES))), columns=FEATURES)
model = LogisticRegression().fit(X, (X["rx_pancreatitis"] > 0.5).astype(int))

path = Path("models/serious_model.joblib")
path.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES, "threshold": 0.5, "model_name": "ci_dummy"}, path)
print(f"Saved dummy model to {path}")
