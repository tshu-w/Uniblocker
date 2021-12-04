#!/usr/bin/env python

from functools import partial
from typing import Any, Optional, Union

import datasets
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_scheduler,
)


class GLUETransformer(LightningModule):
    def __init__(
        self,
        task_name: str,
        model_name_or_path: str,
        num_labels: int,
        max_length: Optional[int] = None,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.collate_fn = partial(
        #     self._collate_fn, tokenizer=tokenizer, max_length=max_length
        # )
        self.convert_to_features = partial(
            self._convert_to_features, tokenizer=tokenizer, max_length=max_length
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.metric = datasets.load_metric("glue", task_name)

    def forward(self, batch):
        return self.model.forward(**batch)

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> STEP_OUTPUT:
        output = self.forward(batch)
        return output.loss

    def metric_step(self, batch) -> Optional[STEP_OUTPUT]:
        output = self.forward(batch)
        loss, logits = output.loss, output.logits
        labels = batch["labels"]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Optional[STEP_OUTPUT]:
        return self.metric_step(batch)

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Optional[STEP_OUTPUT]:
        return self.metric_step(batch)

    def metric_epoch_end(self, outputs: EPOCH_OUTPUT, step: str) -> None:
        splits = getattr(self.trainer.datamodule, f"{step}_splits")
        if len(splits) > 1:
            for i, output in enumerate(outputs):
                split = splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()

                split_metrics = {
                    f"{step}_{k}_{split}": v
                    for k, v in self.metric.compute(
                        predictions=preds, references=labels
                    ).items()
                }

                self.log(f"{step}_loss_{split}", loss, prog_bar=True)
                self.log_dict(split_metrics, prog_bar=True)

            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        split_metrics = {
            f"{step}_{k}": v
            for k, v in self.metric.compute(
                predictions=preds, references=labels
            ).items()
        }

        self.log(f"{step}_loss", loss, prog_bar=True)
        self.log_dict(split_metrics, prog_bar=True)

        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.metric_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.metric_epoch_end(outputs, "test")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit":
            return

        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        gpu_ids = self.trainer._accelerator_connector.parallel_device_ids
        tb_size = self.hparams.batch_size * max(1, len(gpu_ids))
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def get_version(self):
        return f"{self.hparams.model_name_or_path}_{self.hparams.max_length}"

    @staticmethod
    def _collate_fn(
        batch,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ):
        text = [x["text"] for x in batch]
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs["labels"] = torch.LongTensor([x["labels"] for x in batch])

        return inputs

    @staticmethod
    def _convert_to_features(
        batch: Union[dict[str, list], list[Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ) -> Union[dict, Any]:
        features = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        features["labels"] = batch["labels"]

        return features
