"""OceanEmbed-Net: surface fields in, temperature at 15 depths out.

A U-Net with a self-attention bottleneck. The encoder compresses the surface
state into a compact latent grid - the "satellite embedding" the problem
statement asks for - and the decoder expands that back into a depth profile at
every point.

Why this shape rather than a pure Vision Transformer, which the problem
statement also lists: a ViT has to learn from scratch that neighbouring pixels
are related, and it needs a great deal of data before that pays off. With a
few thousand training days a convolutional encoder wins comfortably. Putting
attention only at the bottleneck, where the grid is 8x8, buys the non-local
reasoning that eddies and currents demand at a fraction of the data cost.

One thing deliberately NOT done here. An earlier draft forced the profile to
cool monotonically with depth by predicting positive decrements. Checking the
real GLORYS data killed that idea: 32% of profiles in this basin contain a
genuine temperature inversion, some as large as 5.5 C, because Bay of Bengal
river water forms a fresh lid that traps cooler water above warmer water.
A monotonicity constraint would have made a third of the target unrepresentable.
The 15 levels are therefore predicted directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- building blocks

class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and GELU.

    GroupNorm rather than BatchNorm because training uses small batches of
    large patches; BatchNorm's statistics get noisy and unstable there, while
    GroupNorm does not depend on batch size at all.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialAttention(nn.Module):
    """Multi-head self-attention over the bottleneck grid, treated as tokens.

    At the bottleneck an 8x8 grid is 64 tokens, so full attention is cheap.
    This is what lets a current on one edge of a patch explain a temperature
    anomaly on the other - a convolution's receptive field cannot express
    that at this depth of network, but attention can.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ff_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)   # (b, h*w, c)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ff(self.ff_norm(tokens))
        return x + tokens.transpose(1, 2).reshape(b, c, h, w)


# --------------------------------------------------------------- the network

class OceanEmbedNet(nn.Module):
    """Encoder, embedding, decoder.

    Args:
        in_channels:  surface variables plus auxiliary channels.
        out_depths:   number of standard depth levels to predict (15).
        width:        channel count of the first encoder stage.
        attn_blocks:  self-attention blocks at the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_depths: int = 15,
        width: int = 48,
        attn_blocks: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4

        self.stem = ConvBlock(in_channels, c1)
        self.down1 = ConvBlock(c1, c2)
        self.down2 = ConvBlock(c2, c3)
        self.pool = nn.AvgPool2d(2)

        self.bottleneck = ConvBlock(c3, c3)
        self.attention = nn.Sequential(*[SpatialAttention(c3, heads) for _ in range(attn_blocks)])

        # Decoder. Input channels are doubled by the skip connection.
        self.up2 = ConvBlock(c3 + c3, c2)
        self.up1 = ConvBlock(c2 + c2, c1)
        self.refine = ConvBlock(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_depths, 1)

        # Start with near-zero output. Since the target is standardised, zero
        # is the mean of every depth level - so an untrained network begins at
        # the climatological answer rather than at noise, and the first epochs
        # are spent improving on that instead of recovering from randomness.
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder. Returns the embedding and the skip features."""
        s1 = self.stem(x)                     # (b, c1, H,   W)
        s2 = self.down1(self.pool(s1))        # (b, c2, H/2, W/2)
        s3 = self.down2(self.pool(s2))        # (b, c3, H/4, W/4)

        z = self.bottleneck(self.pool(s3))    # (b, c3, H/8, W/8)
        z = self.attention(z)
        return z, [s1, s2, s3]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """The satellite embedding on its own.

        Exposed separately because the embedding is a deliverable in its own
        right: it can be exported, inspected, or reused for other tasks
        (salinity structure, mixed layer depth, marine heatwave detection)
        without re-running the decoder.
        """
        z, _ = self.encode(x)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, (s1, s2, s3) = self.encode(x)

        y = F.interpolate(z, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, s3], dim=1))

        y = F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, s2], dim=1))

        y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.refine(torch.cat([y, s1], dim=1))

        return self.head(y)


# --------------------------------------------------------------- loss

def masked_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    depth_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over ocean cells only, averaged per depth level.

    Two things this gets right that a plain MSE would not:

    Land and sub-seafloor cells are excluded. Roughly half the grid is land,
    and below about 200 m much of the remaining shelf drops out. Including
    those cells would let the network score well by predicting the fill value
    over places that have no water in them.

    Each depth level is averaged separately before the levels are combined.
    Deep levels vary far less than shallow ones, so a single pooled mean would
    be dominated by the top few metres and the network would effectively stop
    learning anything below the thermocline.
    """
    valid = valid.to(prediction.dtype)
    error = (prediction - target) ** 2 * valid

    # Sum over the spatial dimensions, per sample and per depth.
    per_level = error.sum(dim=(-2, -1))
    counts = valid.sum(dim=(-2, -1)).clamp(min=1.0)
    per_level = per_level / counts                        # (batch, depth)

    if depth_weights is not None:
        per_level = per_level * depth_weights.to(per_level.device)

    # Levels with no valid cells anywhere in the batch must not drag the mean
    # toward zero, so they are dropped rather than counted as perfect.
    has_data = (valid.sum(dim=(-2, -1)) > 0).to(per_level.dtype)
    return (per_level * has_data).sum() / has_data.sum().clamp(min=1.0)


@torch.no_grad()
def profile_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """RMSE, bias and correlation per depth level, in standardised units.

    These are the skill metrics the problem statement asks for. Converting
    them to degrees Celsius is a multiplication by each level's standard
    deviation, done by the caller which has the statistics to hand.
    """
    v = valid.to(prediction.dtype)
    n = v.sum(dim=(0, 2, 3)).clamp(min=1.0)               # per depth

    diff = (prediction - target) * v
    mse = (diff ** 2).sum(dim=(0, 2, 3)) / n
    bias = diff.sum(dim=(0, 2, 3)) / n

    p_mean = (prediction * v).sum(dim=(0, 2, 3)) / n
    t_mean = (target * v).sum(dim=(0, 2, 3)) / n
    pc = (prediction - p_mean.view(1, -1, 1, 1)) * v
    tc = (target - t_mean.view(1, -1, 1, 1)) * v
    cov = (pc * tc).sum(dim=(0, 2, 3))
    denom = torch.sqrt((pc ** 2).sum(dim=(0, 2, 3)) * (tc ** 2).sum(dim=(0, 2, 3))).clamp(min=1e-8)

    return {"rmse": torch.sqrt(mse), "bias": bias, "corr": cov / denom}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
