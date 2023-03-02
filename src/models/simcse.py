from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_scheduler

from src.models.modules import NTXentLoss, Pooler
from src.utils.collators import TransformerCollator


class SimCSE(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        hidden_dropout_prob: float = 0.15,
        pooler_type: Pooler.valid_types = "cls_with_mlp",
        temperature: float = 0.01,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        scheduler_type: str = "linear",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.collate_fn = TransformerCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.hidden_dropout_prob = hidden_dropout_prob
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.pooler = Pooler(pooler_type=pooler_type)
        self.with_mlp = "mlp" in pooler_type
        self.loss_func = NTXentLoss(temperature=temperature, direction="single")

    def forward(self, inputs) -> Any:
        outputs = self.model(**inputs)
        pooled_output = self.pooler(outputs, inputs.attention_mask)
        if self.with_mlp:
            pooled_output = self.model.pooler.dense(pooled_output)
            pooled_output = self.model.pooler.activation(pooled_output)

        return pooled_output

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        if isinstance(batch, list):
            x1, x2 = batch
        else:
            x1 = x2 = batch

        z1, z2 = self.forward(x1), self.forward(x2)

        if self.trainer.strategy.strategy_name.startswith("ddp"):
            z1 = torch.flatten(self.all_gather(z1, sync_grads=True), end_dim=1)
            z2 = torch.flatten(self.all_gather(z2, sync_grads=True), end_dim=1)

        loss = self.loss_func(z1, z2)
        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
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
