from experiment.api import mlflow as mlflow_api
import argparse


def main():
    parser = argparse.ArgumentParser(description="Accept a boolean parameter 'gc'")
    parser.add_argument("experiment_name", type=str, help="Experiment name")
    parser.add_argument(
        "experiment_metric",
        type=str,
        help="Selects the best experiment by the given metric",
    )

    args = parser.parse_args()

    mlflow = mlflow_api.MLFlow()

    mlflow.run_tracking_server()

    # get the best run
    best_run = mlflow.get_best_run_by_metric(
        args.experiment_name, args.experiment_metric
    )
    best_run_id = best_run["run_id"]

    # serve the best model
    mlflow.run_inference_server(best_run_id)


if __name__ == "__main__":
    main()
