from experiment.api import mlflow as mlflow_api

mlflow = mlflow_api.MLFlow()
mlflow.run_tracking_server()