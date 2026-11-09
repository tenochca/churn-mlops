import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
