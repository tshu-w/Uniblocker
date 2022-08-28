import warnings
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from datasets.load import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.utils.sequential_loader import SequentialLoader

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)


def get_dataset(f: Path) -> Dataset:
    try:
        return load_dataset("csv", data_files=str(f), split="train")
    except:
        return Dataset.from_pandas(pd.read_csv(f, low_memory=False))


class Blocking(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/blocking/walmart-amazon_heter",
        index_col: str = "id",
        n_neighbors: int = 100,
        direction: Literal["forward", "reversed", "both"] = "forward",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.table_paths = sorted(Path(data_dir).glob("[1-2]*.csv"))
        self.matches_path = Path(data_dir) / "matches.csv"

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            self.datasets = [get_dataset(t) for t in self.table_paths]

            convert_to_features = self.trainer.model.convert_to_features
            feature_columns = getattr(self.trainer.model, "feature_columns", None)
            preprocess_fn = partial(self._preprocess, index_col=self.hparams.index_col)
            preprocess = lambda x: convert_to_features(preprocess_fn(x))

            for i, dataset in enumerate(self.datasets):
                self.datasets[i] = dataset.map(
                    preprocess,
                    batched=True,
                    batch_size=None,
                )
                self.datasets[i].set_format(type="torch", columns=feature_columns)

        if not hasattr(self, "matches"):
            self.matches = set(
                pd.read_csv(self.matches_path).itertuples(index=False, name=None)
            )

        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

    def prepare_data(self) -> None:
        self.setup()  # setup first to ignore cache conflict in multi processes

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
                persistent_workers=self.hparams.num_workers > 0,
                shuffle=True,
            )
            for dataset in self.datasets
        )
        return SequentialLoader(*dataloaders)

    @staticmethod
    def _preprocess(batch: dict[list], index_col: str):
        columns = [c for c in batch.keys() if index_col not in c]
        batch_size = len(next(iter(batch.values())))

        record = []
        for i in range(batch_size):
            record.append([(c, batch[c][i]) for c in columns])

        return {"record": record}
