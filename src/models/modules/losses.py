from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


class NTXentLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        direction: Literal["single", "both"] = "both",
    ):
        super().__init__()
        self.loss_func = losses.NTXentLoss(temperature)
        self.direction = direction

    def forward(self, z1, z2):
        labels = torch.arange(len(z1), device=z1.device)
        if self.direction == "single":
            return self.loss_func(
                embeddings=z1, labels=labels, ref_emb=z2, ref_labels=labels.clone()
            )
        else:
            embeddings = torch.cat([z1, z2])
            labels = torch.cat([labels, labels])
            return self.loss_func(embeddings, labels)


# taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        dim: int,
        lambd: float = 0.005,
    ):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(dim, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class SelfContLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        z1, z2 = F.normalize(z1), F.normalize(z2)
        loss = (z1 * z2).sum(dim=-1).mean()
        return loss


class CircleLoss(nn.Module):
    """
    According to the paper, the suggested default values of m and gamma are:
    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 128
    Fine-grained Image Retrieval: m = 0.4, gamma = 80
    """

    def __init__(
        self,
        m: float = 0.4,
        gamma: float = 80,
    ):
        super().__init__()
        self.loss_func = losses.CircleLoss(m=m, gamma=gamma)

    def forward(self, z1, z2, matches):
        a1_idx, p_idx = torch.where(matches)
        a2_idx, n_idx = torch.where(matches.logical_not())
        indices_tuple = a1_idx, p_idx, a2_idx, n_idx
        labels = torch.arange(len(z1), device=z1.device)
        return self.loss_func(
            embeddings=z1,
            labels=labels,
            indices_tuple=indices_tuple,
            ref_emb=z2,
            ref_labels=labels,
        )
