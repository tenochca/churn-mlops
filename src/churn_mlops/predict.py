from functools import lru_cache

import mlflow
import mlflow.sklearn
import pandas as pd

from churn_mlops.config import MLFLOW_TRACKING_URI, MODEL_URI

THRESHOLD = 0.5

@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(
        MODEL_URI
    )

def predict(records: list[dict], threshold: float = THRESHOLD) -> list[dict]:
    model = load_model()
    X = pd.DataFrame(records)
    probs = model.predict_proba(X)[:, 1]
    return [
        {"churn_probability": float(p), "churn": bool(p >= threshold)}
        for p in probs
    ]
