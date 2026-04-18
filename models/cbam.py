"""CBAM – Convolutional Block Attention Module.

Reference:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    https://arxiv.org/abs/1807.06521

CBAM sequentially applies channel-wise and spatial attention to refine
feature maps.  It is inserted after key feature-extraction stages in the
YOLOv8 neck so the detector learns *where* and *what* to focus on when
spotting small blade defects in UAV imagery.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention – recalibrates inter-channel relationships.

    Args:
        in_channels: Number of input (and output) channels.
        reduction:   Bottleneck reduction ratio for the shared MLP.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)
        attn = self.sigmoid(self.shared_mlp(avg_feat) + self.shared_mlp(max_feat))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention – highlights informative spatial locations.

    Args:
        kernel_size: Convolution kernel size (7 as per CBAM paper).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies channel attention followed by spatial attention.
    Designed to be a drop-in residual wrapper – the input passes through
    both attention stages and is returned with the same channel count.

    Args:
        in_channels: Number of input channels (pass ``c1`` from YAML).
        reduction:   Channel reduction ratio for the MLP bottleneck.
        kernel_size: Spatial attention convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
