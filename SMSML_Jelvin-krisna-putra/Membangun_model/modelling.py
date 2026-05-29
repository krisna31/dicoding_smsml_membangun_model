import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Employee_Churn_Modelling")

folder_path = "employee_churn_data_preprocessing"

X_train = pd.read_csv(os.path.join(folder_path, "X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(folder_path, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(folder_path, "y_train.csv")).squeeze("columns")
y_test = pd.read_csv(os.path.join(folder_path, "y_test.csv")).squeeze("columns")

input_example = X_train[0:5]

with mlflow.start_run():
    mlflow.autolog()
    model_rf = RandomForestClassifier(class_weight='balanced', random_state=44, n_estimators=100, max_depth=8)
    model_rf.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="model",
        input_example=input_example
    )

    y_pred = model_rf.predict(X_test)
    acc = model_rf.score(X_test, y_test)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
