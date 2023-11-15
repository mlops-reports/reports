import os
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiment.model.dataset import ReportDataset, batch_collate_fn
from experiment.utils.logging import logger
from experiment.model.plotting import plot_beautify

DATASETS = {
    "report_dataset": ReportDataset,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = os.path.dirname(__file__)


class BaseInferer:
    """Inference loop for a trained model. Run the testing scheme."""

    def __init__(
        self,
        dataset: str,
        model: Optional[Module] = None,
        tokenizer: Optional[Module] = None,
        model_path: Optional[str] = None,
        out_path: Optional[str] = None,
        metric_name: str = "accuracy",
    ) -> None:
        """
        Initialize the inference (or testing) setup.

        Parameters
        ----------
        dataset: string
            Which dataset should be used for inference.
        model: torch Module, optional
            The model needs to be tested or inferred. If None, model_path
            and model_params should be specified to load a model.
        model_path: string, optional
            Path to the expected model.
        model_params: dictionary, optional
            Parameters that was specified before the training of the model.
        out_path: string, optional
            If you want to save the predictions, specify a path.
        metric_name: string
            Metric to evaluate the test performance of the model.
        """
        self.dataset = dataset
        self.model_path = model_path
        self.out_path = out_path

        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.metric = metric_name

        if metric_name not in ["accuracy", "f1_score", "roc_auc", "confusion_matrix"]:
            raise NotImplementedError()

    @torch.no_grad()
    def run(self, test_split_only: bool = True, fold_id: int = 0) -> Optional[float]:
        """
        Run inference loop whether for testing purposes or in-production.

        Parameters
        ----------
        test_split_only: bool
            Whether to use all dataset samples or just the testing split. This can be handy
            when testing a pretrained model on your private dataset. Set false if you want to
            use your model in production.

        Returns
        -------
        test_losses: list of floats
            Test loss for each sample. Or any metric you will define. Calculates only if test_split_only is True.
        """
        if self.model is None or self.tokenizer is None:
            assert (
                self.model_path is not None
            ), "Specify the model or specify model_path"
            self.model, self.tokenizer = self.load_model_from_file(
                self.model_path, fold_id
            )
        self.model.eval()

        if test_split_only:
            mode = "test"
        else:
            mode = "inference"

        test_dataset = DATASETS[self.dataset](mode=mode, n_folds=1)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=batch_collate_fn,
        )
        predictions_list = []
        labels_list = []
        for idx, (input_data, target_label) in enumerate(test_dataloader):
            encoded_inputs = self.tokenizer(
                input_data, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            encoded_inputs["labels"] = target_label
            output = self.model(**encoded_inputs)
            if self.out_path is not None:
                torch.save(
                    output.logits, os.path.join(self.out_path, f"sample_{idx}.pt")
                )
            predictions_list.append(self.postprocessing(output.logits))
            if test_split_only:
                labels_list.append(target_label.squeeze().detach().cpu().numpy())

        score = None
        if test_split_only:
            assert self.metric is not None
            logger.info("Running testing loop and evaluation.")
            all_predictions = np.array(predictions_list)
            all_labels = np.array(labels_list)
            score = self.evaluate(all_predictions, all_labels, fold_id)

        self.model.train()
        return score

    def evaluate(
        self, predictions: np.ndarray[int], labels: np.ndarray[int], fold_id: int
    ) -> float:
        logger.info(f"Predictions: {predictions}")
        if self.metric == "accuracy":
            return accuracy_score(labels, predictions)
        elif self.metric == "f1_score":
            return f1_score(labels, predictions, average="weighted")
        elif self.metric == "roc_auc":
            return roc_auc_score(labels, predictions, multi_class="ovr")
        elif self.metric == "confusion_matrix":
            cm = confusion_matrix(labels, predictions, labels=[1, 2, 3, 4])
            conf_mat_png_path = None
            if self.model_path is not None:
                conf_mat_path = os.path.join(
                    self.model_path, "..", "confusion_matrices"
                )
                conf_mat_png_path = os.path.join(conf_mat_path, f"fold{fold_id}.png")
                if not os.path.exists(conf_mat_path):
                    os.makedirs(conf_mat_path)
            plot_beautify(cm, ["1", "2", "3", "4"], conf_mat_png_path)
            return cm
        else:
            raise NotImplementedError()

    def postprocessing(self, logit: Tensor) -> int:
        return int(np.argmax(logit.detach().cpu().numpy()))

    def load_model_from_file(self, model_path: str, fold: int) -> Tuple[Module, Module]:
        """
        Load a pretrained model from file.

        Parameters
        ----------
        model_path: string
            Path to the file which is model saved.
        model_params: dictionary
            Parameters of the model is needed to initialize.

        Returns
        -------
        model: pytorch Module
            Pretrained model ready for inference, or continue training.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(model_path, f"model_fold{fold}"),
            num_labels=5,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, f"tokenizer_fold{fold}")
        )
        logger.info("A previous model and tokenizer are loaded from file.")
        return model, tokenizer
