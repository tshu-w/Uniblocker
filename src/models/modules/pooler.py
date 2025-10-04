from typing import Literal

import torch
import torch.nn as nn


class Pooler(nn.Module):
    valid_types = Literal[
        "last_token", "cls", "average", "cls_with_mlp", "average_with_mlp"
    ]

    def __init__(
        self,
        pooler_type: valid_types,
    ):
        super().__init__()
        self.pooler_type = pooler_type

    def forward(self, outputs, attention_mask) -> torch.Tensor:
        if self.pooler_type.startswith("cls"):
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0]
        elif self.pooler_type.startswith("average"):
            last_hidden_state = outputs.last_hidden_state
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(
                dim=1
            ) / attention_mask.sum(dim=-1).unsqueeze(-1)
        elif self.pooler_type == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            last_hidden_state = outputs.last_hidden_state
            if left_padding:
                pooled_output = last_hidden_state[:, 0]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                pooled_output = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]

        return pooled_output
