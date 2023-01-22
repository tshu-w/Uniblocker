from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback

from src.datamodules.blocking import Blocking


class OnlineEvaluator(Callback):
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
