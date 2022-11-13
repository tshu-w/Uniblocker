from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, get_scheduler

from src.utils.collators import LexMAECollator


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


class LexMAE(LightningModule):
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
        self.collate_fn = LexMAECollator(
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
        decoder_config = AutoConfig.from_pretrained(
            model_name_or_path, num_hidden_layers=2
        )
        self.decoder = AutoModelForMaskedLM.from_config(decoder_config)

        # ties embeddings
        encoder_base_model = getattr(self.encoder, self.encoder.base_model_prefix)
        decoder_base_model = getattr(self.decoder, self.decoder.base_model_prefix)
        decoder_base_model.embeddings = encoder_base_model.embeddings
        if hasattr(self.decoder, "cls"):
            self.decoder.cls = self.encoder.cls
        if hasattr(self.decoder, "lm_head"):
            self.decoder.lm_head = self.encoder.lm_head

    def forward(self, x) -> Any:
        logits = self.encoder(**x).logits
        scores = logits.max(dim=1)[0].softmax(dim=-1)
        encoder_base_model = getattr(self.encoder, self.encoder.base_model_prefix)
        bottleneck = scores @ encoder_base_model.get_input_embeddings().weight.detach()
        return bottleneck

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

        logits = encoder_outputs.logits
        scores = logits.max(dim=1)[0].softmax(dim=-1)
        encoder_base_model = getattr(self.encoder, self.encoder.base_model_prefix)
        bottleneck = scores @ encoder_base_model.get_input_embeddings().weight.detach()

        word_embeddings = getattr(
            self.decoder, self.decoder.base_model_prefix
        ).get_input_embeddings()
        decoder_inputs_embeds = word_embeddings(decoder_input_ids)
        decoder_inputs_embeds[:, 0] = bottleneck

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=decoder_labels,
        )

        self.log("encoder_loss", encoder_outputs.loss, prog_bar=True)
        self.log("decoder_loss", decoder_outputs.loss, prog_bar=True)
        loss = encoder_outputs.loss + decoder_outputs.loss
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
