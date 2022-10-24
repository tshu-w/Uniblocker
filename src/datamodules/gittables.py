from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from .datasets.gittables_dataset import GitTablesDataset


class GitTables(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/gittables/raw_4943312/",
        data_files: Optional[list[str]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.data_files = data_files

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "tables"):
            self.tables = GitTablesDataset(self.data_dir, self.data_files)

        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.tables,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )
