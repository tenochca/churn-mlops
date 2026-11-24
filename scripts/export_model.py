import shutil

import mlflow
import mlflow.artifacts

from churn_mlops.config import (
    MLFLOW_TRACKING_URI,
    MODEL_URI,
    PROJECT_ROOT,
)

DEST = PROJECT_ROOT / "models" / "churn_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
shutil.rmtree(DEST, ignore_errors=True)
DEST.parent.mkdir(exist_ok=True)

mlflow.artifacts.download_artifacts(
    artifact_uri=MODEL_URI,
    dst_path=(DEST),
)

print("Exported best model to ", DEST)
