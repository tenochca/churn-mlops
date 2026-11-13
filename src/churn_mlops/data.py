from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_mlops.config import RAW_DATA_PATH


def load_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    "Load and clean up the dataset"

    df = pd.read_csv(path)

    #Replacing the empty strings with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    #Encoding 1/0 for the target variable
    mapping = {"No": 0, "Yes": 1}
    df["Churn"] = df["Churn"].replace(mapping) #TARGET

    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=117):
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
