# MLflow Demo

MLflow demo for the MLOps Community Meetup at Luxembourg.

# Requirements

- [Python](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- [PyInvoke](https://www.pyinvoke.org/)
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

# Installation

```bash
# install the project environment
inv install
# start the MLflow server from Docker
inv serve
```

Remember to activate the Virtual Environment before accessing the project:

```bash
# on Linux and Mac
./.venv/bin/activate
```

Then 

```bash
 mlflow ui --backend-store-uri "file:/Users/tukanebari/PycharmProjects/mlops_gcp/mlflow_demo/mlruns" --port 5001
```
# Reproduction

## MLflow Tracking

```bash
poetry run python src/mldemo/tracking.py
```

## MLflow Registry

```bash
poetry run python src/mldemo/registry.py
```

## RUN 2 MODELS 

```bash
poetry run python src/mldemo/train_model.py --model_type=rf --experiment_name tracking
poetry run python src/mldemo/train_model.py --model_type=lr --experiment_name tracking
 
```
