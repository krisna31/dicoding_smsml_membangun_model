import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Employee Churn Modelling Tuning")

folder_path = "employee_churn_data_preprocessing"

X_train = pd.read_csv(os.path.join(folder_path, "X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(folder_path, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(folder_path, "y_train.csv")).squeeze("columns")
y_test = pd.read_csv(os.path.join(folder_path, "y_test.csv")).squeeze("columns")

input_example = X_train[0:5]

n_estimators_range = [150, 200, 250, 300]
max_depth_range = [4, 6, 8, 10]

best_recall = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"grid_search_{n_estimators}_{max_depth}"):
            # mlflow.autolog()
            model_rf = RandomForestClassifier(class_weight='balanced', random_state=44, n_estimators=n_estimators, max_depth=max_depth)
            model_rf.fit(X_train, y_train)

            # manual autolog()
            y_train_pred = model_rf.predict(X_train)
            y_train_prob = model_rf.predict_proba(X_train)[:, 1]

            training_score = model_rf.score(X_train, y_train) 

            mlflow.log_metric("training_accuracy_score", training_score)
            mlflow.log_metric("training_precision_score", precision_score(y_train, y_train_pred, zero_division=0))
            mlflow.log_metric("training_recall_score", recall_score(y_train, y_train_pred, zero_division=0))
            mlflow.log_metric("training_f1_score", f1_score(y_train, y_train_pred, zero_division=0))
            mlflow.log_metric("training_log_loss", log_loss(y_train, y_train_prob))
            mlflow.log_metric("training_roc_auc", roc_auc_score(y_train, y_train_prob))
            mlflow.log_metric("training_score", training_score)

            y_pred = model_rf.predict(X_test)
            acc = model_rf.score(X_test, y_test)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_metric("RandomForestClassifier_score_X_test", acc)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            if rec > best_recall:
                best_recall = rec
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(
                    sk_model=model_rf,
                    artifact_path="model",
                    input_example=input_example
                )
