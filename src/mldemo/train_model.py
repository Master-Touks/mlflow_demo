# src/mldemo/train_model.py

import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, choices=["rf", "lr"], default="rf")
args = parser.parse_args()

mlflow.sklearn.autolog()
mlflow.set_experiment("tracking")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

if args.model_type == "rf":
    model = RandomForestClassifier(n_estimators=100)
else:
    model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

mlflow.log_metric("accuracy", score)
print(f"[{args.model_type.upper()}] Accuracy: {score:.3f}")
