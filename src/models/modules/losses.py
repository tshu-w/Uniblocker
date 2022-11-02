import torch
import torch.nn as nn
import torch.nn.functional as F


# taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NTXentLoss(nn.Module):
    def __init__(
        self,
        temperature: float,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z = torch.cat([z1, z2])
        z = F.normalize(z)

        pos = ((z1 * z2).sum(dim=-1) / self.temperature).exp()
        pos = torch.cat([pos, pos])

        c = z @ z.T
        neg = (
            (off_diagonal(c).reshape(c.size(0), -1) / self.temperature)
            .exp()
            .sum(dim=-1)
        )

        loss = -torch.log(pos / neg).mean()
        return loss


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
