"""Canonical ResNet-50 feature decoder used by UFDR."""

import torch
from torch import nn


_FEATURE_CHANNELS = (256, 512, 1024, 2048)


def _conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def _conv3x3(channels):
    return nn.Conv2d(
        channels, channels, kernel_size=3, padding=1, bias=False
    )


class _DecoderBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, *, stride=1, norm_layer=None):
        super().__init__()
        norm = nn.BatchNorm2d if norm_layer is None else norm_layer
        out_channels = planes * self.expansion
        self.conv1 = _conv1x1(in_channels, planes)
        self.bn1 = norm(planes)
        self.upsample = (
            nn.Upsample(
                scale_factor=stride, mode="bilinear", align_corners=False
            )
            if stride != 1
            else nn.Identity()
        )
        self.conv2 = _conv3x3(planes)
        self.bn2 = norm(planes)
        self.conv3 = _conv1x1(planes, out_channels)
        self.bn3 = norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                _conv1x1(in_channels, out_channels),
                nn.Upsample(
                    scale_factor=stride,
                    mode="bilinear",
                    align_corners=False,
                ),
                norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        identity = self.shortcut(inputs)
        hidden = self.relu(self.bn1(self.conv1(inputs)))
        hidden = self.upsample(hidden)
        hidden = self.relu(self.bn2(self.conv2(hidden)))
        hidden = self.bn3(self.conv3(hidden))
        return self.relu(hidden + identity)


class ResNet50FeatureDecoder(nn.Module):
    """Decode ``(e1, e2, e3, e4)`` through the canonical skip route."""

    def __init__(self, *, norm_layer=None):
        super().__init__()
        norm = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.layer3 = self._make_layer(2048, 256, blocks=6, stride=2, norm=norm)
        self.layer2 = self._make_layer(1024, 128, blocks=4, stride=2, norm=norm)
        self.layer1 = self._make_layer(512, 64, blocks=3, stride=2, norm=norm)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _make_layer(in_channels, planes, *, blocks, stride, norm):
        layers = [
            _DecoderBottleneck(
                in_channels,
                planes,
                stride=stride,
                norm_layer=norm,
            )
        ]
        out_channels = planes * _DecoderBottleneck.expansion
        layers.extend(
            _DecoderBottleneck(
                out_channels, planes, norm_layer=norm
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    @staticmethod
    def _validate_features(features):
        if not isinstance(features, tuple) or len(features) != 4:
            raise ValueError("features must be a tuple of exactly 4 tensors")
        for index, (feature, channels) in enumerate(
            zip(features, _FEATURE_CHANNELS), start=1
        ):
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise ValueError(f"feature {index} must be a rank-4 tensor")
            if feature.shape[1] != channels:
                raise ValueError(
                    f"feature {index} channels must be {channels}; "
                    f"got {feature.shape[1]}"
                )
        reference = features[0]
        for index, feature in enumerate(features[1:], start=2):
            if feature.shape[0] != reference.shape[0]:
                raise ValueError("all feature batch dimensions must match")
            if feature.device != reference.device or feature.dtype != reference.dtype:
                raise ValueError("all features must have matching dtype and device")
            previous = features[index - 2]
            if (
                previous.shape[2] != feature.shape[2] * 2
                or previous.shape[3] != feature.shape[3] * 2
            ):
                raise ValueError(
                    "feature spatial dimensions must halve at each level"
                )

    def forward(self, features):
        self._validate_features(features)
        _, e2, e3, e4 = features
        d3 = self.layer3(e4)
        d2 = self.layer2(d3 + e3)
        d1 = self.layer1(d2 + e2)
        return d1, d2, d3


__all__ = ["ResNet50FeatureDecoder"]
