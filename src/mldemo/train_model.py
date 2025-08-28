# src/mldemo/train_model.py
import argparse, os, json, sys
import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["rf", "lr"], default="rf")
    parser.add_argument("--experiment_name", default="tracking")
    args = parser.parse_args()

    # 0) Kill any env overrides that could redirect logging
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"):
        os.environ.pop(var, None)

    # 1) Force a single local store (absolute path)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")

    # 2) Create/select the exact experiment
    exp = mlflow.set_experiment(args.experiment_name)

    # 3) Start run first, then autolog so it attaches to THIS run
    with mlflow.start_run(run_name=args.model_type.upper()) as run:
        mlflow.sklearn.autolog()

        X, y = load_iris(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)

        if args.model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=200)

        model.fit(Xtr, ytr)
        score = model.score(Xte, yte)
        mlflow.log_metric("accuracy", score)
        print(f"[{args.model_type.upper()}] Accuracy: {score:.3f}")

        # 4) Show where it went
        print("\n=== MLflow run info ===")
        print("Tracking URI :", mlflow.get_tracking_uri())
        print("Experiment   :", exp.name, f"(id={exp.experiment_id})")
        print("Run ID       :", run.info.run_id)
        print("Artifacts    :", run.info.artifact_uri)
        print(f"Open UI here -> mlflow ui --backend-store-uri \"file:{MLRUNS_DIR}\" --port 5001")
        print("Then open:    http://127.0.0.1:5001/#/experiments/"
              f"{exp.experiment_id}/runs/{run.info.run_id}")

    # 5) Sanity check via client: list latest runs
    client = MlflowClient(tracking_uri=f"file:{MLRUNS_DIR}")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=3)
    print("\n=== Latest runs in this experiment ===")
    for r in runs:
        print(f"- {r.info.run_id}  name={r.data.tags.get('mlflow.runName','')}  "
              f"accuracy={r.data.metrics.get('accuracy')}")

if __name__ == "__main__":
    sys.exit(main())
