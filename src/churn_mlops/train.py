import mlflow
import mlflow.sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_mlops.config import RAW_DATA_PATH
from churn_mlops.data import load_data, split_data
from churn_mlops.features import build_pipeline, build_preprocessor

RANDOM_STATE = 117

def evaluate(pipeline, X_test, y_test) -> dict[str, float]:
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc" : average_precision_score(y_test, prob),
        "recall" : recall_score(y_test, pred),
        "precision" : precision_score(y_test, pred)
    }

def train_and_log(clf: ClassifierMixin, run_name: str, params: dict | None = None, experiment: str = "churn"):


    df = load_data(RAW_DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    processor = build_preprocessor()
    pipeline = build_pipeline(processor, clf)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": type(clf).__name__,
                "split_random_state": RANDOM_STATE,
                "n_train_rows": len(X_train),
                **(params or {}),
            }
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"],)

    return pipeline, metrics

def main():
    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    clf = RandomForestClassifier(**params)
    _, metrics = train_and_log(clf, run_name="random_forest", params=params)
    print(metrics)

if __name__ == "__main__":
    main()
