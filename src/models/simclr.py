from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_metric_learning import losses
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_scheduler

from src.utils.collate import TransformerCollator


class Projection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.model(x)


class SimCLR(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        temperature: float = 0.05,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        scheduler_type: str = "linear",
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = 128,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.collate_fn = TransformerCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )
        self.model = AutoModel.from_pretrained(model_name_or_path)

        config = self.model.config
        self.projection = Projection(
            input_dim=config.hidden_size,
            hidden_dim=hidden_dim or config.hidden_size,
            output_dim=output_dim or config.hidden_size,
        )

        self.loss_func = losses.NTXentLoss(temperature=temperature)

    def forward(self, x) -> Any:
        return self.model(**x).pooler_output

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        if isinstance(batch, tuple):
            x1, x2 = batch
        else:
            x1 = x2 = batch
        h1, h2 = self.forward(x1), self.forward(x2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        if self.trainer.strategy.strategy_name.startswith("ddp"):
            z1 = torch.flatten(self.all_gather(z1, sync_grads=True), end_dim=1)
            z2 = torch.flatten(self.all_gather(z2, sync_grads=True), end_dim=1)

        labels = torch.arange(len(z1), device=self.device)
        embeddings = torch.cat([z1, z2])
        labels = torch.cat([labels, labels])

        loss = self.loss_func(embeddings, labels)
        self.log("loss", loss)

        return loss

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
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
