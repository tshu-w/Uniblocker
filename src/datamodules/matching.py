import warnings
from operator import itemgetter
from pathlib import Path
from typing import Optional

from datasets.load import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.utils.sequential_loader import SequentialLoader

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)


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
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        dataset_dir = Path(data_dir) / dataset
        self.table_files = [dataset_dir / f for f in table_files]
        self.label_files = [dataset_dir / f for f in label_files]
        self.label_columns = label_columns

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            self.datasets = [
                load_dataset(f.suffix[1:], data_files=str(f), split="train")
                for f in self.table_files
            ]

            convert_to_features = self.trainer.model.convert_to_features
            feature_columns = getattr(self.trainer.model, "feature_columns", None)
            preprocess_fn = self._preprocess
            preprocess = lambda x: convert_to_features(preprocess_fn(x))

            for i, dataset in enumerate(self.datasets):
                self.datasets[i] = dataset.map(
                    preprocess,
                    batched=True,
                    batch_size=None,
                )
                self.datasets[i].set_format(type="torch", columns=feature_columns)

        if not hasattr(self, "golden_pairs"):
            label_files_suffix = self.label_files[0].suffix[1:]
            assert all(label_files_suffix == f.suffix[1:] for f in self.label_files)
            self.golden_pairs = load_dataset(
                label_files_suffix, data_files=map(str, self.label_files), split="train"
            )
            self.golden_pairs = self.golden_pairs.filter(lambda x: x["label"] == 1)
            self.golden_pairs = set(
                zip(*itemgetter(*self.label_columns)(self.golden_pairs))
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
                shuffle=False,
            )
            for dataset in self.datasets
        )
        return SequentialLoader(*dataloaders)

    @staticmethod
    def _preprocess(batch: dict[list]):
        columns = [c for c in batch.keys() if "id" not in c]
        batch_size = len(next(iter(batch.values())))

        record = []
        for i in range(batch_size):
            record.append([(c, batch[c][i]) for c in columns])

        return {"record": record}
