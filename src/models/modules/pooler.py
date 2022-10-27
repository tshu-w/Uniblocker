from typing import Literal

import torch
import torch.nn as nn


class Pooler(nn.Module):
    valid_types = Literal["cls", "average", "cls_with_mlp", "average_with_mlp"]

    def __init__(
        self,
        pooler_type: valid_types,
        original_pooler: nn.Module,
    ):
        super().__init__()
        self.pooler_type = pooler_type.split("_")[0]
        self.with_mlp = "mlp" in pooler_type
        self.dense = original_pooler.dense
        self.activation = original_pooler.activation

    def forward(self, outputs, attention_mask) -> torch.Tensor:
        if self.pooler_type == "cls":
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0]
        elif self.pooler_type == "average":
            last_hidden_state = outputs.last_hidden_state
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(
                dim=1
            ) / attention_mask.sum(dim=-1).unsqueeze(-1)

        if self.with_mlp:
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)

        return pooled_output
