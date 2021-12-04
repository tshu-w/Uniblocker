#!/usr/bin/env python

from functools import partial
from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import F1, MetricCollection, Precision, Recall


class SampleModel(LightningModule):
    def __init__(self):
        super.__init__()
        self.save_hyperparameters()

        self.collate_fn = partial(self._collate_fn)

        metrics_kwargs = {}
        metrics = MetricCollection(
            {
                "f1": F1(**metrics_kwargs),
                "prc": Precision(**metrics_kwargs),
                "rec": Recall(**metrics_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, batch):
        ...

    def common_step(self, batch, step: str) -> Optional[STEP_OUTPUT]:
        loss = self.forward(batch)

        metrics = getattr(self, f"{step}_metrics")
        metrics()

        self.log_dict(metrics, prog_bar=True)
        self.log(f"{step}_loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "valid")

    def test_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        return super().configure_optimizers()

    @staticmethod
    def _collate_fn():
        ...
