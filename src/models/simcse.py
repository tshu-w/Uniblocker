from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, get_scheduler


class SimCSE(LightningModule):
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # HACK: https://github.com/huggingface/transformers/issues/14931
        tokenizer("Lorem Ipsum", truncation=True, max_length=max_length)
        self.convert_to_features = partial(
            self._convert_to_features, tokenizer=tokenizer, max_length=max_length
        )
        self.feature_columns = tokenizer.model_input_names
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, x) -> Any:
        return self.model(**x).pooler_output

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        x = batch
        z1, z2 = self.forward(x), self.forward(x)

        if self.trainer.strategy.strategy_name.startswith("ddp"):
            z1 = torch.flatten(self.all_gather(z1, sync_grads=True), end_dim=1)
            z2 = torch.flatten(self.all_gather(z2, sync_grads=True), end_dim=1)

        sim = (
            F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
            / self.temperature
        )
        labels = torch.arange(len(sim), device=self.device)

        loss = F.cross_entropy(sim, labels)
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

    @staticmethod
    def _convert_to_features(
        batch: Union[dict[str, list], list[Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ) -> Union[dict, Any]:
        texts = [
            " ".join(f"{t[1] or ''}" for t in record) for record in batch["record"]
        ]
        features = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return features
