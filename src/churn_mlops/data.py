from pathlib import Path
import pandas as pd
from churn_mlops.config import RAW_DATA_PATH


def load_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
