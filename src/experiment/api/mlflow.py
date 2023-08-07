"""MLflow utils for ml experiments"""

import pathlib

# import logging
import json
import os
import mlflow
from urllib import parse
import requests
from typing import Any


class MLFlow:
    MLFLOW_HOST = "http://127.0.0.1"
    MLFLOW_TRACKING_PORT = 9999
    MLFLOW_MODEL_PORT = 1234
    BACKEND_URI_STORE = "sqlite:///experiments.sqlite"
    DEFAULT_ARTIFACT_ROOT = pathlib.Path("mlartifacts")

    def __init__(self, local_storage: bool = False):
        """
        If the local_storage flag is set to False, then the MLFlow.BACKEND_URI_STORE is set to the
        postgresql database URI, and the MLFlow.DEFAULT_ARTIFACT_ROOT is set to the AWS S3 bucket

        Args:
          local_storage (bool): If you want to store the artifacts locally, set this to True. Defaults
        to False
        """

        self.local_storage = local_storage

        if not self.local_storage:
            db_username = os.getenv("DB_USERNAME")
            db_password = parse.quote(os.getenv("DB_PASSWORD"))
            db_host = os.getenv("DB_HOST")
            db_name = os.getenv("DB_NAME")
            aws_bucket = os.getenv("AWS_BUCKET")

            MLFlow.BACKEND_URI_STORE = (
                f"postgresql://{db_username}:{db_password}@{db_host}:5432/{db_name}"
            )
            MLFlow.DEFAULT_ARTIFACT_ROOT = f"{aws_bucket}/mlartifacts"

        self.set_tracking_uri(f"{MLFlow.MLFLOW_HOST}:{MLFlow.MLFLOW_TRACKING_PORT}")

    def set_tracking_uri(self, uri: str) -> None:
        """
        > Sets the tracking server URI

        Args:
          uri (str): The location of the tracking server.
        """
        mlflow.set_tracking_uri(uri)

    def get_or_create_experiment(
        self, experiment_name: str, tags: dict[str, str] = {}
    ) -> str:
        """
        If an experiment with the given name exists, return its ID. Otherwise, create a new experiment
        with the given name and return its ID

        Args:
          experiment_name (str): The name of the experiment.
          tags (dict): A dictionary of key-value pairs that will be added to the experiment as tags.

        Returns:
          The experiment_id
        """

        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=str(MLFlow.DEFAULT_ARTIFACT_ROOT),
                tags=tags,
            )
        except mlflow.exceptions.RestException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

        return experiment_id

    def log_mlflow(
        self,
        model: Any,
        experiment_name: str,
        run_name: str,
        log_dict: dict[str, Any],
        registered_model_name: str,
        user_id: str = "anon",
        tags: dict[str, str] = {},
        artifact_path: str = "",
    ) -> str:
        """
        > This function takes a model, experiment name, run name, a dictionary of parameters and metrics,
        a registered model name, and an artifact path. It then creates an experiment if it doesn't exist,
        starts a run, logs the parameters and metrics, and logs the model. It returns the run ID

        Args:
          model (Any): Any
          experiment_name (str): The name of the experiment to log to.
          run_name (str): The name of the run.
          log_dict (dict): a dictionary of parameters and metrics to log
          registered_model_name (str): The name of the model in the MLflow Model Registry.
          artifact_path (str): The directory in which to save the model. If you don't specify anything,
        MLflow will save the model in a directory with the same name as the run ID.

        Returns:
          The run_id
        """
        experiment_id = self.get_or_create_experiment(experiment_name)
        tags = {**tags, **{"mlflow.user": user_id}}

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
            tags=tags,
        ) as run:
            run_id = run.info.run_id

            for param, value in log_dict["params"].items():
                mlflow.log_param(param, value)

            for metric, value in log_dict["metrics"].items():
                mlflow.log_metric(metric, value)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )

        return run_id

    def run_server(self, background: bool = True) -> None:
        """
        `run_server` starts an MLflow server on the port specified in the `MLFlow` class
        """
        os.system(
            f"""
                mlflow server --port {MLFlow.MLFLOW_TRACKING_PORT} \
                --backend-store-uri {MLFlow.BACKEND_URI_STORE} \
                --default-artifact-root {MLFlow.DEFAULT_ARTIFACT_ROOT} \
                {'&' if background else ''}
            """
        )

    def serve_model(self, run_id: str, background: bool = True) -> None:
        """
        `serve_model` takes a run_id and serves the model associated with that run_id on port
        `MLFlow.MLFLOW_MODEL_PORT`

        Args:
          run_id (str): The run ID of the model you want to serve.
        """

        if self.local_storage:
            MODAL_URI = (
                pathlib.Path(MLFlow.DEFAULT_ARTIFACT_ROOT) / run_id / "artifacts"
            )
        else:
            MODAL_URI = f"{MLFlow.DEFAULT_ARTIFACT_ROOT}/{run_id}/artifacts"

        os.system(
            f"""
                mlflow models serve --model-uri '{MODAL_URI}' \
                --port {MLFlow.MLFLOW_MODEL_PORT} \
                --no-conda {'&' if background else ''}
            """
        )

    def get_predictions(self, data: dict[Any, Any] = {}) -> Any:
        """
        It takes in a dictionary of data, and returns the response from the MLflow model server

        Args:
          data (dict): The data to be passed to the model for predictions.

        Returns:
          The response object.
        """
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            f"{MLFlow.MLFLOW_HOST}:{MLFlow.MLFLOW_MODEL_PORT}/invocations",
            data=json.dumps(data),
            headers=headers,
        )

        return response

    def clean(self):
        """
        > It kills the local ports used by MLFlow and removes the artifacts, experiments, and runs
        """
        os.system(
            f"kill $(lsof -t -i:{MLFlow.MLFLOW_TRACKING_PORT}) && kill $(lsof -t -i:{MLFlow.MLFLOW_MODEL_PORT})"
        )
        os.system("rm -rf mlartifacts experiments.sqlite mlruns predictions.csv")
