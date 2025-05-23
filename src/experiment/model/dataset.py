import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import Dataset

from experiment.utils.dbutils import DatabaseUtils
from experiment.utils.logging import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_PATH = os.path.dirname(__file__)

# You can use such a random seed 35813 (part of the Fibonacci Sequence).
np.random.seed(35813)


class DataError(Exception):
    pass


def batch_collate_fn(batch_data: List[str]) -> Tuple[Tuple[Tensor, ...], Tensor]:
    """Definition of how the batched data will be treated."""
    inputs, labels = zip(*batch_data)
    batched_labels = torch.stack(labels)
    return inputs, batched_labels


class BaseDataset(Dataset):
    """Base class for common functionalities of all datasets."""

    def __init__(
        self,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        batch_size: int = 0,
        in_memory: bool = True,
    ):
        """
        Dataset class to initialize data operations, cross validation and preprocessing.

        Parameters
        ----------
        sql_query: string
            SQL query to data
        mode: string
            Which split to return, can be 'train', 'validation', 'test', or 'inference'.
            Use 'inference' to load all data if you have a pretrained model and want to use
            it in-production.
        n_folds: integer
            Number of cross validation folds.
        current_fold: integer
            Defines which cross validation fold will be selected for training.
        in_memory: bool
            Whether to store all data in memory or not.
        """
        super().__init__()
        assert (
            current_fold < n_folds
        ), "selected fold index cannot be more than number of folds."
        self.mode = mode
        self.n_folds = n_folds
        self.in_memory = in_memory
        self.current_fold = current_fold
        self.batch_size = batch_size

        self.dbutils = DatabaseUtils()

        if self.in_memory:
            self.loaded_samples = self.get_all_samples()
            self.n_samples_total = len(self.loaded_samples)
        else:
            self.n_samples_total = self.get_number_of_samples()

        if not mode == "inference" and in_memory:
            self.samples_labels = self.get_labels()
            assert len(self.samples_labels) == self.n_samples_total

        logger.info(f"Loading {self.n_samples_total} from dataset...")
        # Keep half of the data as 'unseen' to be used in inference.
        self.seen_data_indices, self.unseen_data_indices = self.get_fold_indices(
            self.n_samples_total, 2
        )

        if mode == "train" or mode == "validation":
            # Here split the 'seen' data to train and validation.
            if self.in_memory:
                self.seen_samples_labels = self.samples_labels[self.seen_data_indices]
                self.seen_samples_data = self.loaded_samples[self.seen_data_indices]

            self.n_samples_seen = len(self.seen_data_indices)
            self.tr_indices, self.val_indices = self.get_fold_indices(
                self.n_samples_seen,
                self.n_folds,
                self.current_fold,
            )
            logger.info(
                f"Train/Val/Test split is: {len(self.tr_indices)}/{len(self.val_indices)}/{len(self.unseen_data_indices)}"
            )

        if mode == "train":
            self.selected_indices = self.tr_indices
            if self.in_memory:
                self.samples_labels = self.seen_samples_labels[self.tr_indices]
                self.loaded_samples = self.seen_samples_data[self.tr_indices]
        elif mode == "validation":
            self.selected_indices = self.val_indices
            if self.in_memory:
                self.samples_labels = self.seen_samples_labels[self.val_indices]
                self.loaded_samples = self.seen_samples_data[self.val_indices]
        elif mode == "test":
            self.selected_indices = self.unseen_data_indices
            if self.in_memory:
                self.samples_labels = self.samples_labels[self.unseen_data_indices]
                self.loaded_samples = self.loaded_samples[self.unseen_data_indices]
        elif not mode == "inference":
            raise ValueError(
                "mode should be 'train', 'validation', 'test', or 'inference'"
            )
        if mode == "inference":
            self.n_samples_in_split = self.n_samples_total
        else:
            self.n_samples_in_split = len(self.selected_indices)

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]:
        if self.in_memory:
            if self.mode == "inference":
                label = None
            else:
                label = torch.from_numpy(self.samples_labels[index]).to(device)
            sample_data = self.loaded_samples[index][0]
        else:
            raise NotImplementedError(
                "Lazy loading not implemented, please use In-Memory execution."
            )
            # sample_data, label = self.get_sample_data(index)
        return sample_data, label

    def __len__(self) -> int:
        return self.n_samples_in_split

    def get_number_of_samples(self) -> int:
        """
        Method to find how many samples are expected in ALL dataset.
        E.g., number of images in the target folder, number of rows in dataframe.
        """
        return self.dbutils.get_table_size_by_table_name(
            "report_classifications", schema="annotation"
        )

    def get_labels(self, label_flag: str = "annotation_value_flag") -> np.ndarray:
        """
        Method to read and store labels in a numpy array

        Returns
        -------
        labels: numpy ndarray
            An array stores the labels for each sample.
        """
        df = self.dbutils.select_table_by_columns(
            columns=[label_flag], table="report_classifications", schema="annotation"
        )
        if df is None:
            raise DataError("Data couldnt be retrieved from database.")
        df = df.dropna()
        return df[[label_flag]].values.astype(np.int64)

    def get_all_samples(self) -> np.ndarray:
        """
        Convert data from all samples to the Torch Tensor objects to store in a list later.
        This function can be memory-consuming but time-saving, recommended to be used on small datasets.

        Returns
        -------
        all_data: np.ndarray
            A numpy array represents all data.
        """
        df = self.dbutils.select_table_by_columns(
            columns=["translated_text", "annotation_value_flag"],
            table="report_classifications",
            schema="annotation",
        )
        if df is None:
            raise DataError("Data couldnt be retrieved from database.")
        df = df.dropna(subset=["annotation_value_flag"])
        return df[["translated_text"]].values

    def get_fold_indices(
        self, all_data_size: int, n_folds: int, fold_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        --------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.

        Returns
        --------
        train_indices: numpy ndarray
            Indices to get the training dataset.
        val_indices: numpy ndarray
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True)
        split_indices = kf.split(range(all_data_size))
        train_indices, val_indices = [
            (np.array(train), np.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} dataset ({self.mode}) with"
            f" n_samples={self.n_samples_in_split}, "
            f"current fold:{self.current_fold+1}/{self.n_folds}"
        )


class ReportDataset(BaseDataset):
    """
    Report data
    """

    def __init__(
        self,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        batch_size: int = 0,
        in_memory: bool = True,
    ):
        super().__init__(
            mode,
            n_folds,
            current_fold,
            batch_size,
            in_memory,
        )
