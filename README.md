# Churn MLOps

An end-to-end MLOps project that takes a customer churn model from training to a validated prediction API. The model is intentionally simple; the focus is the workflow around it: reproducible environments, experiment tracking, a model registry, and serving.

## What it does

- Trains and compares a random forest and a logistic regression on the Telco Customer Churn dataset
- Uses a single scikit-learn `Pipeline` (preprocessing + model), so training and serving apply identical transformations
- Tracks parameters, metrics, and model artifacts with MLflow
- Registers every trained model and promotes the best one (by PR-AUC) to a `best` alias
- Serves predictions with FastAPI, with Pydantic validation that rejects values the model never saw in training
- Manages dependencies and the Python version with uv and a committed lockfile

## Architecture

```
Raw CSV -> clean/split -> sklearn Pipeline -> train.py -> MLflow tracking + registry
                                                                  |
                                                    registry.py promotes "best"
                                                                  |
                                          predict.py loads champion -> FastAPI /predict
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd churn-mlops
uv sync
```

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at `data/raw/Telco-Customer-Churn.csv`.

```bash
# Train models and register them in MLflow
uv run python -m churn_mlops.train

# Promote the best model (by PR-AUC) to the "champion" alias
uv run python -m churn_mlops.registry

# Start the API
uv run uvicorn churn_mlops.api:app --reload

# Browse experiments and models
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

API docs are at http://localhost:8000/docs.

### Example request

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 151.65
}'
```

## Results

Evaluated on a stratified 20% hold-out set. Churn is imbalanced (~26%), so ROC-AUC and PR-AUC are the primary metrics rather than accuracy.

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|-------|---------|--------|--------|-----------|
| Random forest | 0.841 | 0.631 | 0.794 | 0.525 |
| Logistic regression | 0.833 | 0.620 | 0.791 | 0.496 |

## Project structure

```
src/churn_mlops/
├── config.py     # paths and settings, resolved from the project root
├── data.py       # loading, cleaning, train/test split
├── features.py   # preprocessing and pipeline construction
├── train.py      # training, evaluation, MLflow logging and registration
├── registry.py   # promotes the best model to the "champion" alias
├── predict.py    # loads the champion model and runs inference
└── api.py        # FastAPI app and Pydantic request schema
notebooks/        # exploratory data analysis
```

## Design decisions

- **Whole pipeline is the artifact.** The registered model includes preprocessing, so the API only needs raw input columns.
- **Strict input validation.** Categorical fields use `Literal` types, because the one-hot encoder would otherwise silently turn a typo into an all-zeros row.
- **Alias-based promotion.** The API loads `models:/churn-model@champion`, so deploying a new model means moving an alias, not changing code.
- **Notebook for EDA only.** All training logic lives in `src/`, which keeps it testable and automatable.

## Roadmap

- [ ] Tests for data cleaning and the API
- [ ] Dockerfile
- [ ] GitHub Actions CI with a model-quality gate
- [ ] Data drift monitoring with Evidently
- [ ] Automated retraining and promotion

## Tech stack

Python, uv, pandas, scikit-learn, MLflow, FastAPI, Pydantic
