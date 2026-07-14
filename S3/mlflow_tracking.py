# mlflow_tracking.py
# Day 2, Session 2 - Introduction to MLflow (experiment tracking)

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters, metrics, and the model
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Logged run with accuracy: {acc:.2f}")

# To view results, run in your terminal:
#   mlflow ui
# Then open http://localhost:5000 in your browser
