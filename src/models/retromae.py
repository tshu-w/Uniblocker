from typing import Any, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_scheduler,
)

from src.utils.collators import RetroMAECollator


def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


class RetroMAE(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        encoder_mlm_probability: float = 0.15,
        decoder_mlm_probability: float = 0.30,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        scheduler_type: str = "linear",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.collate_fn = RetroMAECollator(
            tokenizer=tokenizer,
            max_length=max_length,
            encoder_mlm_probability=encoder_mlm_probability,
            decoder_mlm_probability=decoder_mlm_probability,
        )
        encoder_config = AutoConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        self.encoder = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=encoder_config
        )
        # An approximation to the RetroMAE enhanced decoder with
        # self-attention and cross-attention rather thant just cross-attention
        decoder_config = AutoConfig.from_pretrained(
            model_name_or_path,
            is_decoder=True,
            num_hidden_layers=1,
            add_cross_attention=True,
        )
        self.decoder = AutoModel.from_config(decoder_config).encoder

    def forward(self, x) -> Any:
        return self.encoder(**x).hidden_states[-1][:, 0]

    def mae(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        encoder_labels,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_labels,
    ) -> STEP_OUTPUT:
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=encoder_labels,
        )
        cls_hidden_states = encoder_outputs.hidden_states[-1][:, :1]  # Bx1xH
        encoder_embeddings = getattr(
            self.encoder, self.encoder.base_model_prefix
        ).embeddings
        decoder_embedding_output = encoder_embeddings(decoder_input_ids)  # BxLxH
        embedding_output_with_cls = torch.cat(
            [cls_hidden_states, decoder_embedding_output[:, 1:]], dim=1
        )

        padding_idx = self.decoder.config.pad_token_id
        decoder_position_ids = create_position_ids_from_input_ids(
            decoder_input_ids, padding_idx
        )
        decoder_position_embeddings = encoder_embeddings.position_embeddings(
            decoder_position_ids
        )
        cls_embeddings = cls_hidden_states + decoder_position_embeddings
        # cls_embeddings = encoder_embeddings.LayerNorm(cls_embeddings)
        # cls_embeddings = encoder_embeddings.dropout(cls_embeddings)

        extended_attention_mask = self.encoder.invert_attention_mask(
            decoder_attention_mask.bool()
        )
        decoder_outputs = self.decoder(
            hidden_states=cls_embeddings,
            encoder_hidden_states=embedding_output_with_cls,
            encoder_attention_mask=extended_attention_mask,
        )
        sequence_output = decoder_outputs[0]
        lm_head = getattr(self.encoder, "lm_head", getattr(self.encoder, "cls", None))
        prediction_scores = lm_head(sequence_output)
        decoder_masked_lm_loss = F.cross_entropy(
            prediction_scores.view(-1, self.decoder.config.vocab_size),
            decoder_labels.view(-1),
        )

        self.log("encoder_loss", encoder_outputs.loss, prog_bar=True)
        self.log("decoder_loss", decoder_masked_lm_loss, prog_bar=True)
        loss = encoder_outputs.loss + decoder_masked_lm_loss
        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        loss = self.mae(**batch)
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
