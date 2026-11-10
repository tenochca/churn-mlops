from pathlib import Path
import pandas as pd
from churn_mlops.config import RAW_DATA_PATH


def load_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    "Load and clean up the dataset"

    df = pd.read_csv(path)

    #Replacing the empty strings with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    #Encoding 1/0 for the target variable
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}) #TARGET

    print(df.shape)

    return df
