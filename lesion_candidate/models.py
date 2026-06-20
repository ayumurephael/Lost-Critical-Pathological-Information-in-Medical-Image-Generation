from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Residual convolution block used by the MUIS-style clustering teacher."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class MUISStyleEncoder(nn.Module):
    """Encoder for the DCCS/MUIS image-level clustering teacher."""

    def __init__(self, dim_zs: int = 64, dim_zc: int = 8, base: int = 64) -> None:
        super().__init__()
        self.dim_zs = dim_zs
        self.dim_zc = dim_zc
        self.base = base
        self.feature_dim = base * 8
        self.block1 = ConvBlock(1, base, stride=2)
        self.block2 = ConvBlock(base, base * 2, stride=2)
        self.block3 = ConvBlock(base * 2, base * 4, stride=2)
        self.block4 = ConvBlock(base * 4, base * 8, stride=2)
        self.block5 = ConvBlock(base * 8, base * 8, stride=1)
        self.pre = nn.Sequential(nn.GroupNorm(_groups(base * 8), base * 8), nn.SiLU(inplace=True))
        self.head = nn.Conv2d(base * 8, dim_zs + dim_zc, kernel_size=1)

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        skips = []
        x = self.block1(x)
        skips.append(x)
        x = self.block2(x)
        skips.append(x)
        x = self.block3(x)
        skips.append(x)
        x = self.block4(x)
        bottle = self.block5(x)
        feat = self.pre(bottle)
        pooled = F.adaptive_avg_pool2d(feat, 1)
        logits = self.head(pooled).flatten(1)
        zs = logits[:, : self.dim_zs]
        zc_logits = logits[:, self.dim_zs :]
        cam = None
        if return_cam:
            raw = F.conv2d(feat, self.head.weight, self.head.bias)
            cam = F.relu(raw[:, self.dim_zs :])
        return zs, zc_logits, bottle, cam, list(reversed(skips))


class LatentCritic(nn.Module):
    def __init__(self, dim_zs: int = 64, dim_zc: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_zs + dim_zc, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MutualInfoDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, z_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + z_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.net(torch.cat([feat, z], dim=1))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        y = self.se(y)
        return self.act(y + self.skip(x))


class ASPP(nn.Module):
    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        branch_ch = channels // 2
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_ch, 3, padding=rate, dilation=rate, bias=False),
                    nn.GroupNorm(_groups(branch_ch), branch_ch),
                    nn.SiLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.SiLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.gate(gate) + self.skip(skip))
        return skip * alpha


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = AttentionGate(in_ch, skip_ch, out_ch)
        self.block = ResidualSEBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        return self.block(torch.cat([x, skip], dim=1))


class StudentUNet(nn.Module):
    """Research-grade A-only lesion segmenter.

    The model is still called StudentUNet so existing command-line tools keep the
    same public API, but internally it is a residual attention U-Net with ASPP
    and deep supervision.
    """

    def __init__(self, base: int = 64, dropout: float = 0.10, deep_supervision: bool = True) -> None:
        super().__init__()
        self.base = base
        self.deep_supervision = deep_supervision
        self.enc1 = ResidualSEBlock(1, base, dropout=dropout * 0.25)
        self.enc2 = ResidualSEBlock(base, base * 2, stride=2, dropout=dropout * 0.50)
        self.enc3 = ResidualSEBlock(base * 2, base * 4, stride=2, dropout=dropout * 0.75)
        self.enc4 = ResidualSEBlock(base * 4, base * 8, stride=2, dropout=dropout)
        self.enc5 = ResidualSEBlock(base * 8, base * 16, stride=2, dropout=dropout)
        self.context = nn.Sequential(ASPP(base * 16), ResidualSEBlock(base * 16, base * 16, dropout=dropout))
        self.dec4 = DecoderBlock(base * 16, base * 8, base * 8, dropout=dropout)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4, dropout=dropout * 0.75)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2, dropout=dropout * 0.50)
        self.dec1 = DecoderBlock(base * 2, base, base, dropout=dropout * 0.25)
        self.out = nn.Conv2d(base, 1, kernel_size=1)
        self.aux4 = nn.Conv2d(base * 8, 1, kernel_size=1)
        self.aux3 = nn.Conv2d(base * 4, 1, kernel_size=1)
        self.aux2 = nn.Conv2d(base * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        center = self.context(e5)
        d4 = self.dec4(center, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        logits = self.out(d1)
        if self.training and self.deep_supervision:
            return (
                logits,
                F.interpolate(self.aux2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.aux3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.aux4(d4), size=logits.shape[-2:], mode="bilinear", align_corners=False),
            )
        return logits


def main_logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - prob) * target).sum(dim=(1, 2, 3))
    return (1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()
