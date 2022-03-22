import json
from pathlib import Path
from typing import Optional

from datasets.load import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers.data.data_collator import DefaultDataCollator


class GitTables(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/gittables/",
        data_files: Optional[list[str]] = None,  # ["dwarf_tables.jsonl"],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.data_files = (
            [str(Path(data_dir) / f) for f in data_files] if data_files else None
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            dataset = load_dataset(
                "json",
                data_dir=self.data_dir,
                data_files=self.data_files,
                split="train",
                streaming=True,
            )

            convert_to_features = self.trainer.model.convert_to_features
            feature_columns = getattr(self.trainer.model, "feature_columns", None)
            preprocess_fn = self._preprocess
            preprocess = lambda x: convert_to_features(preprocess_fn(x))

            dataset = dataset.map(
                preprocess,
                batched=True,
                batch_size=4,
                remove_columns=["_file", "_idx", "tuple"],
            )
            self.dataset = dataset.with_format("torch")

        self.collate_fn = (
            getattr(self.trainer.model, "collate_fn", None) or DefaultDataCollator()
        )

    def prepare_data(self) -> None:
        self.setup()  # setup first to ignore cache conflict in multi processes

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    @staticmethod
    def _preprocess(batch: dict[list]):
        batch_size = len(next(iter(batch.values())))

        text = []
        for tpl in map(json.loads, batch["tuple"]):
            text.append(" ".join(f"{k} {v or ''}" for k, v in tpl.items()))

        return {"text": text}
