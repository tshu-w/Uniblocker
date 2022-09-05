from copy import deepcopy
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from pl_bolts.callbacks import BYOLMAWeightUpdate
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, get_scheduler

from .simsiam import SiamArm


class BYOL(LightningModule):
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
        # HACK: https://github.com/huggingface/transformers/issues/14931
        tokenizer("Lorem Ipsum", truncation=True, max_length=max_length)
        self.convert_to_features = partial(
            self._convert_to_features, tokenizer=tokenizer, max_length=max_length
        )
        self.feature_columns = tokenizer.model_input_names
        model = AutoModel.from_pretrained(model_name_or_path)

        config = self.model.config
        self.online_network = SiamArm(
            encoder=model,
            input_dim=config.hidden_size,
            hidden_dim=hidden_dim or config.hidden_size,
            output_dim=output_dim or config.hidden_size,
        )
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def forward(self, x) -> Any:
        y, _, _ = self.siamarm(**x)
        return y

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        x = batch

        y1, z1, h1 = self.online_network(x)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(x)

        loss = -2 * F.cosine_similarity(h1, z2).mean()
        self.log("loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx
        )

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
