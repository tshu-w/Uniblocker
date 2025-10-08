from pathlib import Path
from typing import Union

import fasttext
import lightning as L
import torch
import torch.nn.functional as F
import torchtext
from jsonargparse import lazy_instance
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

if hasattr(torchtext, "disable_torchtext_deprecation_warning"):
    torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer

from src.models.modules.aggregators import AGGREGATOR_TYPE, get_aggregator

fasttext.FastText.eprint = lambda *args, **kwargs: None


class DeepBlocker(L.LightningModule):
    def __init__(
        self,
        tokenizer: str = "basic_english",
        fasttext_model_path: str = "./models/wiki.en/wiki.en.bin",
        aggregator_type: AGGREGATOR_TYPE = "sif",
        input_dim: int = 300,
        hidden_dims: Union[int, list[int]] = [300, 150],
        activations: Union[nn.Module, list[nn.Module]] = lazy_instance(nn.ReLU),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["activations"])

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        if not isinstance(activations, list):
            activations = [activations] * (len(hidden_dims) - 1)
        assert len(hidden_dims) == len(activations) + 1

        encoder_layers = [nn.Linear(input_dim, hidden_dims[0])]
        decoder_layers = [nn.Linear(hidden_dims[0], input_dim)]
        for i in range(len(hidden_dims) - 1):
            encoder_layers.extend(
                [activations[i], nn.Linear(hidden_dims[i], hidden_dims[i + 1])]
            )
            decoder_layers.extend(
                [activations[i], nn.Linear(hidden_dims[i + 1], hidden_dims[i])]
            )

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers[::-1])

        fasttext_model_path = Path(fasttext_model_path).expanduser()
        assert fasttext_model_path.exists()

        tokenizer = get_tokenizer(tokenizer)
        embedder = fasttext.load_model(str(fasttext_model_path))
        self.collate_fn = get_aggregator(aggregator_type)(
            tokenizer=tokenizer, embedder=embedder
        )

    def on_fit_start(self):
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        x = batch
        z = self.forward(x)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
