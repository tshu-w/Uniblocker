from typing import Any, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_scheduler

from src.models.modules import CircleLoss, Pooler
from src.utils.augment import Augmenter
from src.utils.collators import TransformerCollatorWithDistances


class UniBlocker(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        augment_prob: float = 0.1,
        hidden_dropout_prob: float = 0.15,
        pooler_type: Pooler.valid_types = "cls_with_mlp",
        m: float = 0.4,
        gamma: int = 80,
        learning_rate: float = 2e-5,
        distance: float = 0.8,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        scheduler_type: str = "linear",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        augmenter = Augmenter(augment_prob) if augment_prob != 0 else None
        self.collate_fn = TransformerCollatorWithDistances(
            tokenizer=tokenizer,
            max_length=max_length,
            augmenter=augmenter,
        )
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.hidden_dropout_prob = hidden_dropout_prob
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.pooler = Pooler(pooler_type=pooler_type)
        self.with_mlp = "mlp" in pooler_type
        self.loss_func = CircleLoss(m=m, gamma=gamma)
        self.distance = distance

    def forward(self, inputs) -> Any:
        outputs = self.model(**inputs)
        pooled_output = self.pooler(outputs, inputs.attention_mask)
        if self.with_mlp:
            pooled_output = self.model.pooler.dense(pooled_output)
            pooled_output = self.model.pooler.activation(pooled_output)

        return pooled_output

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        batch, distances = batch
        if isinstance(batch, list):
            x1, x2 = batch
        else:
            x1 = x2 = batch

        z1, z2 = self.forward(x1), self.forward(x2)

        loss = self.loss_func(z1, z2, distances > self.distance)
        self.log("loss", loss, prog_bar=True)

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
