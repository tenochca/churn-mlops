from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from churn_mlops.config import RAW_DATA_PATH
from churn_mlops.data import load_data, split_data
from churn_mlops.features import build_pipeline, build_preprocessor

df = load_data(RAW_DATA_PATH)

X_train, X_test, y_train, y_test = split_data(df)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("\nX_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

clf = RandomForestClassifier(n_estimators=300, random_state=117, max_depth=8, class_weight="balanced", n_jobs=-1)

processor = build_preprocessor()
pipeline = build_pipeline(processor, clf)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:,1]

report = classification_report(y_test, pred, target_names=["No Churn", "Churn"])
auc = roc_auc_score(y_test, prob)

print(report)
print("AUC :", auc)
