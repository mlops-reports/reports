from experiment.api import mlflow as mlflow_api
import argparse


def main():
    parser = argparse.ArgumentParser(description="Accept a boolean parameter 'gc'")
    parser.add_argument(
        "--gc", action="store_true", help="Enable GC (Garbage Collection)"
    )

    args = parser.parse_args()

    mlflow = mlflow_api.MLFlow()

    # clean with garbage collection
    mlflow.clean(gc=args.gc)


if __name__ == "__main__":
    main() 
