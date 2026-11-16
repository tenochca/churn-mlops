import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT} / 'mlflow.db"
REGISTERED_MODEL_NAME = "churn_model"
MODEL_URI = os.getenv("CHURN_MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}@champion")
