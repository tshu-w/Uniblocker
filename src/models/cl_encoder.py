from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer


class CLEncoder(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        temperature: float = 0.05,
        learning_rate: float = 2e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # HACK: https://github.com/huggingface/transformers/issues/14931
        tokenizer("Lorem Ipsum", truncation=True)
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
        sim = (
            F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
            / self.temperature
        )
        labels = torch.arange(len(sim), device=self.device)

        loss = F.cross_entropy(sim, labels)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.learning_rate
        )
        return optimizer

    @staticmethod
    def _convert_to_features(
        batch: Union[dict[str, list], list[Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ) -> Union[dict, Any]:
        texts = [
            " ".join(f"{t[0]} {t[1] or ''}" for t in record)
            for record in batch["record"]
        ]
        features = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return features
