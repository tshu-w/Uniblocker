from typing import Optional

import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_size: Optional[int] = None,
        batchnorm_bottleneck: bool = True,
    ):
        super().__init__()
        if bottleneck_size is not None:
            layers = [nn.Linear(input_dim, bottleneck_size, bias=False)]
            if batchnorm_bottleneck:
                layers += [nn.BatchNorm1d(bottleneck_size)]
            layers += [nn.Linear(bottleneck_size, output_dim)]
        else:
            layers = [nn.Linear(input_dim, output_dim)]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layer: int = 2,
        activation=nn.ReLU(),
        **kwargs,
    ):
        super().__init__()
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090
        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            activation,
        ]
        for _ in range(hidden_layer - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                activation,
            ]
        layers += [Linear(hidden_dim, output_dim, **kwargs)]
        self.module = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.module(x)
