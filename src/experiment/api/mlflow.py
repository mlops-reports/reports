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
            db_username = os.getenv("MLFLOW_DB_USERNAME")
            db_password = parse.quote(os.getenv("MLFLOW_DB_PASSWORD"))
            db_host = os.getenv("MLFLOW_DB_HOST")
            db_name = os.getenv("MLFLOW_DB_NAME")
            aws_bucket = os.getenv("MLFLOW_AWS_BUCKET")

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
        ml_library: str = "tensorflow"
    ) -> str:
        """
        The `log_mlflow` function logs model parameters, metrics, and the model itself to MLflow for
        tracking and experimentation purposes.
        
        Args:
          model (Any): The `model` parameter is the machine learning model that you want to log.
          experiment_name (str): The `experiment_name` parameter is a string that represents the name of
        the MLflow experiment where the run will be logged.
          run_name (str): The `run_name` parameter is a string that represents the name of the MLflow
        run.
          log_dict (dict[str, Any]): The `log_dict` parameter is a dictionary that contains two keys:
        "params" and "metrics".
          registered_model_name (str): The parameter "registered_model_name" is the name of the model
        that will be registered in the MLflow model registry.
          user_id (str): The `user_id` parameter is an optional parameter that represents the user
        identifier.
          tags (dict[str, str]): The `tags` parameter is a dictionary that allows you to add custom
        key-value pairs as tags to the MLflow run.
          artifact_path (str): The `artifact_path` parameter is the directory within the MLflow run
        where the model artifacts will be saved.
          ml_library (str): The `ml_library` parameter is used to specify the machine learning library
        that was used to train the model.
        
        Returns:
          the `run_id` of the MLflow run.
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

            if ml_library == "tensorflow":
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    keras_model_kwargs={"save_format": "h5"}
                )
            elif ml_library == "sklearn":
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

    def clean(self, gc: bool = False):
        """
        The `clean` function kills processes running on specific ports, removes certain files and
        directories, and optionally performs garbage collection on a specified backend store URI.
        
        Args:
          gc (bool): The `gc` parameter is a boolean flag that determines whether to perform garbage
        collection on the MLFlow backend store. If `gc` is set to `True`, the code will execute the
        MLFlow garbage collection command to clean up any unused artifacts and metadata in the backend
        store.
        """
        os.system(
            f"kill $(lsof -t -i:{MLFlow.MLFLOW_TRACKING_PORT}) && kill $(lsof -t -i:{MLFlow.MLFLOW_MODEL_PORT})"
        )
        os.system("rm -rf mlartifacts experiments.sqlite mlruns predictions.csv")

        if gc:
            os.system(
                f"""
                    mlflow gc --backend-store-uri {MLFlow.BACKEND_URI_STORE}
                """
            )
