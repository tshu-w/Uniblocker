#!/usr/bin/env python

from operator import itemgetter
from pathlib import Path
from typing import Optional

from datasets.load import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader

from .utils import EvaluationLoop


class Matching(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/deepmatcher/",
        dataset: str = "Structured/Walmart-Amazon",
        table_files: list[str] = ["tableA.csv", "tableB.csv"],
        label_files: list[str] = ["train.csv", "valid.csv", "test.csv"],
        label_columns: list[str] = ["ltable_id", "rtable_id"],
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        dataset_dir = Path(data_dir) / dataset
        self.table_files = [dataset_dir / f for f in table_files]
        self.label_files = [dataset_dir / f for f in label_files]
        self.label_columns = label_columns

    def prepare_data(self) -> None:
        for file in self.table_files + self.label_files:
            load_dataset(file.suffix[1:], data_files=str(file))

        self.convert_to_features = getattr(
            self.trainer.model, "convert_to_features", None
        )
        self.feature_columns = getattr(self.trainer.model, "feature_columns", None)
        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

        if self.convert_to_features is not None:
            preprocess_fn = self.preprocess
            self.preprocess = lambda x: self.convert_to_features(preprocess_fn(x))

        self.override_evaluation_loop()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datasets = [
            load_dataset(f.suffix[1:], data_files=str(f), split="train")
            for f in self.table_files
        ]

        for i, dataset in enumerate(self.datasets):
            self.datasets[i] = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=None,
            )
            if self.convert_to_features is not None:
                self.datasets[i].set_format(type="torch", columns=self.feature_columns)

        label_files_suffix = self.label_files[0].suffix[1:]
        assert all(label_files_suffix == f.suffix[1:] for f in self.label_files)
        self.golden_pairs = load_dataset(
            label_files_suffix, data_files=map(str, self.label_files), split="train"
        )
        self.golden_pairs = self.golden_pairs.filter(lambda x: x["label"] == 1)
        self.golden_pairs = set(
            zip(*itemgetter(*self.label_columns)(self.golden_pairs))
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=ConcatDataset(self.datasets),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def preprocess(self, batch):
        columns = [c for c in batch.keys() if "id" not in c]
        text = []
        for tuple in zip(*(batch[c] for c in columns)):
            text.append(" ".join(map(lambda x: str(x or ""), tuple)))

        return {"text": text}

    def override_evaluation_loop(self):
        self.trainer.test_loop = EvaluationLoop()

        empty_fn = lambda *args, **kwargs: None
        self.val_dataloader = self.test_dataloader = empty_fn
        self.trainer.model.validation_step = self.trainer.model.test_step = empty_fn
