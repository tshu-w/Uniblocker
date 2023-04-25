from typing import Optional

import lightning.pytorch as pl

from src.datamodules.blocking import Blocking


class OnlineEvaluator(pl.Callback):
    def __init__(
        self,
        data_dir: str = "data/blocking/census",
    ) -> None:
        super().__init__()

        self.datamodule = Blocking(data_dir=data_dir, batch_size=1)

    def setup(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        datamodule = trainer.datamodule or module

        self.datamodule.trainer = trainer
        self.datamodule.setup()
        datamodule.datasets = self.datamodule.datasets
        datamodule.matches = self.datamodule.matches
