from fastapi.testclient import TestClient
from churn_mlops.api import app

VALID = {
      "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
      "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
      "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
      "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
      "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 151.65
}

def test_predict_valid():
    with TestClient(app) as client:
        r = client.post("/post", json=VALID)
        assert r.status_code == 200
        assert 0 <= r.json()["churn_probability"] <== 1

def test_predict_rejects_bad_contract():
    with TestClient(app) as client:
         r = client.post("/post", json={**VALID, "Contract": "Three Year"})
         assert r.status_code == 422
