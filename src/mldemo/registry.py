"""Demo of MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html."""

# %% IMPORTS

from pprint import pprint

import mlflow
from mlflow.models.signature import infer_signature
from sklearn import datasets, ensemble, metrics, model_selection

# %% CONFIGS

# - MLflow
TRACKING_URI = "http://127.0.0.1:5001"
REGISTRY_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "registry"
MODEL_NAME = "mldemo"
STAGE = "Production"

# - Model
MAX_DEPTH = 3
N_ESTIMATORS = 5

# - Others
TEST_SIZE = 0.2
RANDOM_STATE = 42

# %% MLFLOW

# configure mlflow tracking/registry
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(REGISTRY_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog()

client = mlflow.tracking.MlflowClient()

# %% DATASETS

X, y = datasets.load_wine(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# %% TRAINING + REGISTRY

with mlflow.start_run(run_name="Training") as run:
    print(f"[START] Run ID: {run.info.run_id}")

    model = ensemble.RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model = model.fit(X_train, y_train)

    signature = infer_signature(X_train, y_test)

    mlflow.sklearn.log_model(
        model,
        MODEL_NAME,
        signature=signature,
        registered_model_name=MODEL_NAME
    )

    latest_version = client.get_latest_versions(MODEL_NAME, stages=[])[-1].version
    print(f"[STOP] Run ID: {run.info.run_id} - Registered as version {latest_version}")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage=STAGE,
        archive_existing_versions=True,
    )
    print(f"✅ Model '{MODEL_NAME}' version {latest_version} transitioned to stage '{STAGE}'")

# %% SEARCHING

for model in client.search_registered_models():
    print("Model:", model.name)
    pprint(dict(model), indent=4)

    for version in client.search_model_versions(f"name='{model.name}'"):
        print("- Version:", version.version)
        pprint(dict(version), indent=4)

# %% INFERENCE

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{STAGE}")
score = metrics.accuracy_score(y_test, model.predict(X_test))
print(f"📈 Accuracy: {score:.2f}")
