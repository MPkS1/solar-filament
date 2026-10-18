"""A small, dependency-free U-Net for binary filament segmentation.

Pure `torch` (no torchvision) so it trains/infers on CPU locally and on Kaggle
GPUs alike. The encoder can optionally be swapped for a torchvision/timm
backbone later; the default is a plain conv U-Net that is plenty for this task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.depth = depth
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(prev, chs[-1])
        self.ups = nn.ModuleList()
        prev = chs[-1]
        for c in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.ups.append(DoubleConv(c * 2, c))
            prev = c
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for down in self.downs:
            x = down(x)
            feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.ups[2 * i](x)
            skip = feats[self.depth - 1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[2 * i + 1](x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def tversky_loss(logits, target, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0):
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def combined_loss(logits, target, bce_weight: float = 0.5, tversky_weight: float = 0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + tversky_weight * tversky_loss(logits, target)
