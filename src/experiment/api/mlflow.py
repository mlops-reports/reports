"""MLflow utils for ml experiments"""

import json
import logging
import os
import pathlib
from typing import Any, Union
from urllib import parse

import mlflow

# import logging
import numpy as np
import requests

from experiment.utils import bash as bash_utils
from experiment.utils.logging import logger


class MLFlow:
    MLFLOW_HOST = "http://127.0.0.1"
    MLFLOW_TRACKING_PORT = 9999
    MLFLOW_MODEL_PORT = 1234
    BACKEND_URI_STORE = "sqlite:///experiments.sqlite"
    DEFAULT_ARTIFACT_ROOT = pathlib.Path("mlartifacts")

    def __init__(self, local_storage: bool = False) -> None:
        """
        If the local_storage flag is set to False, then the MLFlow.BACKEND_URI_STORE is set to the
        postgresql database URI, and the MLFlow.DEFAULT_ARTIFACT_ROOT is set to the AWS S3 bucket

        Args:
          local_storage (bool): If you want to store the artifacts locally, set this to True. Defaults
        to False
        """

        self.local_storage = local_storage
        self.logger = logger

        if not self.local_storage:
            db_username = os.getenv("MLFLOW_DB_USERNAME")
            db_password = parse.quote(os.getenv("MLFLOW_DB_PASSWORD"))
            db_host = os.getenv("MLFLOW_DB_HOST")
            db_name = os.getenv("MLFLOW_DB_NAME")
            self.aws_bucket = os.getenv("MLFLOW_AWS_BUCKET")

            MLFlow.BACKEND_URI_STORE = (
                f"postgresql://{db_username}:{db_password}@{db_host}:5432/{db_name}"
            )
            MLFlow.DEFAULT_ARTIFACT_ROOT = f"s3://{self.aws_bucket}/mlartifacts"

        self.set_tracking_uri(f"{MLFlow.MLFLOW_HOST}:{MLFlow.MLFLOW_TRACKING_PORT}")

    def set_tracking_uri(self, uri: str) -> None:
        """
        > Sets the tracking server URI

        Args:
          uri (str): The location of the tracking server.
        """
        mlflow.set_tracking_uri(uri)

    def get_model_config(self, path: Union[str, pathlib.Path]) -> dict:
        """
        The function `get_model_config` reads a JSON file from the given path and returns its contents
        as a dictionary.

        Args:
          path (Union[str, pathlib.Path]): The `path` parameter is the path to the file containing the
        model configuration. It can be either a string representing the file path or a `pathlib.Path`
        object.

        Returns:
          a dictionary object, which is the model configuration loaded from the specified file.
        """
        model_config = {}
        with open(path, "r") as file:
            model_config = json.load(file)

        return model_config

    def log_artifact(self, path: str) -> None:
        """
        The function logs an artifact file to the MLflow tracking server.

        Args:
          path (str): The `path` parameter is a string that represents the file or directory path of the
        artifact that you want to log.
        """
        mlflow.log_artifact(path)

    def set_model_config(
        self, path: Union[str, pathlib.Path], update_fields: dict
    ) -> None:
        """
        The function `set_model_config` updates the model configuration with the provided fields and
        saves the updated configuration.

        Args:
          path (Union[str, pathlib.Path]): The `path` parameter is the path to the model configuration
        file.
          update_fields (dict): A dictionary containing the fields and their updated values that need to
        be added or modified in the model configuration.

        Returns:
          a dictionary.
        """
        model_config = self.get_model_config(path)
        model_config = {**model_config, **update_fields}

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as json_file:
            json_file.write(json.dumps(model_config, indent=4, ensure_ascii=False))

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

    def download_artifacts(
        self,
        run_id: str,
        artifact_path=Union[str, pathlib.Path],
        dst_path=Union[str, pathlib.Path],
    ) -> None:
        """
        The function `download_artifacts` is used to download artifacts from a specific MLflow run to a
        specified destination path.

        Args:
          run_id (str): The `run_id` parameter is a unique identifier for a specific MLflow run.
          artifact_path: The `artifact_path` parameter specifies the path of the artifact within the
        run.
          dst_path: The `dst_path` parameter is the destination path where the artifacts will be
        downloaded to.
        """
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dst_path,
        )

    def log_experiment_run(
        self,
        model: Any,
        experiment_name: str,
        run_name: str,
        log_dict: dict[str, Any],
        registered_model_name: str,
        user_id: str = os.getenv("MLFLOW_TRACKING_USERNAME", "anon"),
        tags: dict[str, str] = {},
        artifact_path: str = "",
        extra_artifacts: dict[str:str] = None,
        ml_library: str = "pytorch",
    ) -> str:
        """
        The `log_experiment_run` function logs model parameters, metrics, and the model itself to MLflow for
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
          extra_artifacts (dict): If exists, upload extra artifacts.

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

            if extra_artifacts:
                for paths in extra_artifacts.values():
                    mlflow.log_artifact(paths["local_path"], paths["save_path"])

            if ml_library == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            elif ml_library == "pytorch":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )

        return run_id

    def run_tracking_server(self, background: bool = True) -> None:
        """
        `run_tracking_server` starts an MLflow server on the port specified in the `MLFlow` class
        """

        mlflow_host_suffix = MLFlow.MLFLOW_HOST.split("//")[1]

        is_tracking_server_running = bash_utils.is_server_running(
            mlflow_host_suffix, MLFlow.MLFLOW_TRACKING_PORT
        )

        if not is_tracking_server_running:
            os.system(
                f"""
                  mlflow server --port {MLFlow.MLFLOW_TRACKING_PORT} \
                  --host {mlflow_host_suffix} \
                  --backend-store-uri {MLFlow.BACKEND_URI_STORE} \
                  --default-artifact-root {MLFlow.DEFAULT_ARTIFACT_ROOT} \
                  {'&' if background else ''} 
              """
            )

        else:
            logging.info("MLFlow tracking server is already running.")

        os.system(
            f"""
                open {MLFlow.MLFLOW_HOST}:{MLFlow.MLFLOW_TRACKING_PORT}
            """
        )

        # 192.168.0.230 | 195.88.86.196 | 10.121.120.4

    def get_best_run_by_metric(
        self, experiment_name: str, metric_name: str = None
    ) -> dict[str:str]:
        """
        The function `get_best_run_by_metric` retrieves the best run from an MLflow experiment
        based on a specified metric, and returns the run ID and corresponding metric value.

        Args:
          experiment_name (str): The name of the MLflow experiment you want to search for runs in.
          metric_name (str): The `metric_name` parameter is a string that specifies the name of the
        metric you want to use to determine the best run.

        Returns:
          a dictionary with the best run
        """
        runs = mlflow.search_runs(
            experiment_ids=[
                mlflow.get_experiment_by_name(experiment_name).experiment_id
            ]
        )

        # Filter the runs to exclude those without the specified metric
        runs_with_metric = runs[runs[f"metrics.{metric_name}"].notnull()]

        # Find the run with the highest value of the specified metric
        best_run = runs_with_metric.loc[
            runs_with_metric[f"metrics.{metric_name}"].idxmax()
        ]

        return best_run

    def run_inference_server(self, run_id: str, background: bool = True) -> None:
        """The `run_inference_server` function serves a machine learning model using MLflow's `mlflow models serve`
        command.

        Parameters
        ----------
        run_id : str
          The `run_id` parameter is a string that represents the unique identifier of a MLflow run. It is
        used to locate the model artifacts associated with the specified run.
        background : bool, optional
          The `background` parameter is a boolean flag that determines whether the model serving command
        should run in the background or not.
        """
        if self.local_storage:
            MODAL_URI = (
                pathlib.Path(MLFlow.DEFAULT_ARTIFACT_ROOT) / run_id / "artifacts"
            )
        else:
            MODAL_URI = f"{MLFlow.DEFAULT_ARTIFACT_ROOT}/{run_id}/artifacts"

        os.system(
            f"""
                mlflow models serve  \
                --model-uri '{MODAL_URI}' \
                --port {MLFlow.MLFLOW_MODEL_PORT} \
                --no-conda {'&' if background else ''}
            """
        )

    def get_predictions(self, data: list) -> Any:
        """The function `get_predictions` sends a POST request to a MLFlow server to get predictions for a
        given list of data.

        Parameters
        ----------
        data : list
          The `data` parameter is a list of input instances for which you want to get predictions. Each
        instance in the list should be in a format that is compatible with the ML model you are using.

        Returns
        -------
          a list of predictions.

        """
        headers = {"Content-Type": "application/json"}
        input_data = {"instances": data}

        response = requests.post(
            f"{MLFlow.MLFLOW_HOST}:{MLFlow.MLFLOW_MODEL_PORT}/invocations",
            json=input_data,
            headers=headers,
        )

        predictions = [
            np.argmax(prediction) for prediction in response.json()["predictions"]
        ]

        return predictions

    def clean(self, gc: bool = False) -> None:
        """The `clean` function kills processes running on specified ports and removes mlartifacts,
        experiments.sqlite, and mlruns directories.

        Parameters
        ----------
        gc : bool, optional
          The `gc` parameter is a boolean flag that determines whether garbage collection should be
        performed. If `gc` is set to `True`, the code will execute the `mlflow gc` command to perform
        garbage collection on the MLFlow backend store.

        """
        os.system(
            f"kill $(lsof -t -i:{MLFlow.MLFLOW_TRACKING_PORT}) && kill $(lsof -t -i:{MLFlow.MLFLOW_MODEL_PORT})"
        )

        if gc:
            os.system(
                f"""
                    mlflow gc --backend-store-uri {MLFlow.BACKEND_URI_STORE}
                """
            )

        os.system("rm -rf mlartifacts experiments.sqlite mlruns")
