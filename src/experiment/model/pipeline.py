import os
import pandas as pd
from typing import List, Any, Tuple
from torch.nn import Module

from experiment.model.train import BaseTrainer
from experiment.model.inference import BaseInferer
from experiment.api.mlflow import MLFlow

FILE_PATH = os.path.dirname(__file__)


class ExperimentPipeline(MLFlow):
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs, all parameters for the Trainer."""
        self.kwargs = kwargs
        self.results_save_path = os.path.join(
            FILE_PATH, "results", self.kwargs["model_name"]
        )
        self.model_path = os.path.join(
            FILE_PATH, "parameters", self.kwargs["model_name"]
        )
        self.model_per_fold: List[Tuple[Module, Module]] = []
        self.val_result_per_fold: List[float] = []
        self.test_result_per_fold: List[float] = []
        self.trainer = BaseTrainer(**self.kwargs)
        if "n_folds" in self.kwargs.keys():
            self.n_folds = self.kwargs["n_folds"]
        else:
            self.n_folds = 5

    def train_model(self) -> None:
        """Run training loop for each cross validation fold."""
        for fold in range(self.n_folds):
            print(f"--------------------- FOLD {fold} ---------------------")
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
            print(f"Fold: {fold_idx} - Test score: {self.test_score}")
            if self.test_score is not None:
                self.test_result_per_fold.append(self.test_score)

    def get_results_table(self) -> None:
        """Save experiment results in a table, this function should be called at the end."""
        results_df = pd.DataFrame(
            {
                "Avg. Test Scores": self.test_result_per_fold,
                # "Last Val. Scores": self.val_result_per_fold,
            }
        )
        # Index of the dataframe will indicate the fold id.
        if not os.path.exists(self.results_save_path):
            os.makedirs(self.results_save_path)
        results_df.to_csv(self.results_save_path + "_results.csv", index=True)
        print("Experiments results are successfully saved.")
