import mlflow
from mlflow import MlflowClient

from churn_mlops.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


def promote_best(metric: str = "pr_auc", experiment: str = "churn") -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    runs = mlflow.search_runs(
        experiment_names=[experiment],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    print(runs.head)

    best_run_id = runs.iloc[0]["run_id"]

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    print("versions", versions)
    best = next(v for v in versions if v.run_id == best_run_id)

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "best", best.version)
    return best.version

if __name__ == "__main__":
    print("Best model is now version: ", promote_best())
