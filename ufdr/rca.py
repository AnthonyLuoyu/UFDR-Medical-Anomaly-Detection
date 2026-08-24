"""Re-parameterized calibration attention (RCA)."""

import torch
from torch import nn
from torch.nn import functional as F


class RCA(nn.Module):
    """Residual spatial attention adapted from the historical ``SA`` source."""

    def __init__(self, channels):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        attention_channels = max(1, channels // 8)
        self.query_conv = nn.Conv2d(channels, attention_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, attention_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        if not torch.is_tensor(inputs) or inputs.ndim != 4:
            raise ValueError("input must be a rank-4 NCHW tensor")
        if inputs.shape[1] != self.channels:
            raise ValueError(
                f"input channels must match configured channels ({self.channels})"
            )

        batch, channels, height, width = inputs.shape
        locations = height * width
        query = self.query_conv(inputs).reshape(batch, -1, locations).transpose(1, 2)
        key = self.key_conv(inputs).reshape(batch, -1, locations)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value_conv(inputs).reshape(batch, channels, locations)
        attended = torch.bmm(value, attention.transpose(1, 2)).reshape(
            batch, channels, height, width
        )
        return inputs + self.gamma * attended


__all__ = ["RCA"]
