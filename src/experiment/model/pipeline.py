import os
from typing import Any, List, Tuple

import pandas as pd
from torch.nn import Module

from experiment.api.mlflow import MLFlow
from experiment.model.inference import BaseInferer
from experiment.model.train import BaseTrainer
from experiment.utils.logging import logger

FILE_PATH = os.path.dirname(__file__)


class ExperimentPipeline(MLFlow):
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs, all parameters for the Trainer."""
        self.kwargs = kwargs
        self.results_save_path = os.path.join(
            FILE_PATH, self.kwargs["model_name"], "results"
        )
        self.model_path = os.path.join(FILE_PATH, self.kwargs["model_name"], "models")
        self.model_per_fold: List[Tuple[Module, Module]] = []
        self.val_result_per_fold: List[float] = []
        self.test_result_per_fold: List[float] = []
        self.trainer = BaseTrainer(**self.kwargs)
        if "n_folds" in self.kwargs.keys():
            self.n_folds = self.kwargs["n_folds"]
        else:
            self.n_folds = 5

        if "metric_name" in self.kwargs.keys():
            self.metric_name = self.kwargs["metric_name"]
        else:
            self.metric_name = "accuracy"

    def train_model(self) -> None:
        """Run training loop for each cross validation fold."""
        for fold in range(self.n_folds):
            logger.info(f"--------------------- FOLD {fold} ---------------------")
            model = self.trainer.train(current_fold=fold)
            self.model_per_fold.append(model)
            # Last epochs validation score:
            self.last_val_result = self.trainer.val_loss_per_epoch[-1]
            self.val_result_per_fold.append(self.last_val_result)

    def run_inference(self) -> None:
        """Run inference with all models, each trained per fold."""
        for fold_idx in range(self.n_folds):
            if self.model_per_fold:
                self.inferer = BaseInferer(
                    dataset=self.kwargs["dataset"],
                    model=self.model_per_fold[fold_idx][0],
                    tokenizer=self.model_per_fold[fold_idx][1],
                )
            else:
                self.inferer = BaseInferer(
                    dataset=self.kwargs["dataset"],
                    model_path=self.model_path,
                )
            self.test_score = self.inferer.run(fold_id=fold_idx)
            logger.info(f"Fold: {fold_idx} - Test score: {self.test_score}")
            if self.test_score is not None:
                self.test_result_per_fold.append(self.test_score)

    def get_results_table(self) -> None:
        """Save experiment results in a table, this function should be called at the end."""
        results = {
            f"Test {self.metric_name} Scores": self.test_result_per_fold,
            # "Last Val. Scores": self.val_result_per_fold,
        }
        results_df = pd.DataFrame(results)
        # Index of the dataframe will indicate the fold id.
        if os.path.exists(self.results_save_path + ".csv"):
            prev_results_df = pd.read_csv(self.results_save_path + ".csv")
            if f"Test {self.metric_name} Scores" in prev_results_df.columns:
                logger.info("There are some previous results. Exiting...")
                return
            appended_results = pd.concat([prev_results_df, results_df], axis=1)
            appended_results.to_csv(self.results_save_path + ".csv", index=False)
        else:
            results["Folds"] = list(range(self.n_folds))
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_save_path + ".csv", index=False)
        logger.info("Experiments results are successfully saved.")
