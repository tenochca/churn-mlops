from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ['tenure', "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
"gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
"InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
"TechSupport", "StreamingTV", "StreamingMovies", "Contract",
"PaperlessBilling", "PaymentMethod", "SeniorCitizen",
]

SERVICE_FEATURES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

def build_preprocessor() -> ColumnTransformer:
    "Preps the transformers to be used in the pipeline"

    processor = ColumnTransformer([
        ("scaler", StandardScaler(), NUMERICAL_FEATURES),
        ("onehot", OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

    return processor

def build_pipeline(processor: ColumnTransformer, classifier: Any) -> Pipeline:
    "Assembles the pipeline with preprocessing and model steps"

    pipeline = Pipeline(steps=[
        ('preprocessor', processor),
        ('classifier', classifier)
    ])

    return pipeline
