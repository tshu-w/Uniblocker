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

        self.datamodule = Blocking(data_dir=data_dir)

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        datamodule = trainer.datamodule or pl_module
        datamodule.hparams.index_col = self.datamodule.hparams.index_col
        datamodule.hparams.n_neighbors = self.datamodule.hparams.n_neighbors
        datamodule.hparams.direction = self.datamodule.hparams.direction

        self.datamodule.trainer = trainer
        self.datamodule.setup()
        datamodule.datasets = self.datamodule.datasets
        datamodule.matches = self.datamodule.matches
