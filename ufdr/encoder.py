"""Feature encoders for the public UFDR implementation."""

import importlib
from pathlib import Path

import torch
from torch import nn


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_image(inputs, in_channels):
    if not torch.is_tensor(inputs) or inputs.ndim != 4:
        raise ValueError("input must be a rank-4 NCHW tensor")
    if inputs.shape[1] != in_channels:
        raise ValueError(
            f"input channels must match configured channels ({in_channels})"
        )
    if inputs.shape[2] <= 0 or inputs.shape[3] <= 0:
        raise ValueError("input spatial dimensions must be positive")
    if inputs.shape[2] % 32 or inputs.shape[3] % 32:
        raise ValueError("input spatial dimensions must be divisible by 32")
    if not inputs.is_floating_point():
        raise ValueError("input must have a floating-point dtype")


def _validate_in_channels(in_channels):
    if isinstance(in_channels, bool) or not isinstance(in_channels, int):
        raise ValueError("in_channels must be a positive integer")
    if in_channels <= 0:
        raise ValueError("in_channels must be a positive integer")


class _PyramidDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(32, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class TinyPyramidEncoder(nn.Module):
    """A small deterministic-convolution pyramid for tests and smoke runs only.

    This encoder has no stochastic layers or external weights.  It preserves the
    production encoder's four-scale channel contract, but it is not intended as a
    replacement for the pretrained DINOv3 encoder in research experiments.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        _validate_in_channels(in_channels)
        self.in_channels = in_channels
        # Fixed smoke-model weights make comparisons reproducible without
        # consuming or depending on the caller's random-number stream.
        with torch.random.fork_rng(devices=[]):
            fixed_cpu_state = torch.Generator(device="cpu").manual_seed(0).get_state()
            torch.random.set_rng_state(fixed_cpu_state)
            self.stem = _PyramidDownsample(in_channels, 64)
            self.stage1 = _PyramidDownsample(64, 256)
            self.stage2 = _PyramidDownsample(256, 512)
            self.stage3 = _PyramidDownsample(512, 1024)
            self.stage4 = _PyramidDownsample(1024, 2048)

    def forward(self, inputs):
        _validate_image(inputs, self.in_channels)
        hidden = self.stem(inputs)
        e1 = self.stage1(hidden)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return e1, e2, e3, e4


class DINOv3ConvNeXtTinyEncoder(nn.Module):
    """DINOv3 ConvNeXt-Tiny adapted to UFDR's four-scale feature contract.

    The provider is imported lazily so importing :mod:`ufdr.encoder` does not
    require ``lightly_train``.  Published weights are RGB; a configured
    single-channel input is repeated to RGB immediately before the backbone.
    """

    _PROVIDER_MODULE = (
        "lightly_train._models.dinov3.dinov3_src.hub.backbones"
    )

    def __init__(
        self,
        weights,
        in_channels=3,
        input_norm="imagenet_from_minus1_1",
    ):
        super().__init__()
        _validate_in_channels(in_channels)
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3 for pretrained DINOv3")
        if input_norm not in {"none", "imagenet_from_minus1_1"}:
            raise ValueError(
                "input_norm must be 'none' or 'imagenet_from_minus1_1'"
            )
        if weights is None:
            raise FileNotFoundError("DINOv3 weights path is required")
        try:
            weights_path = Path(weights).expanduser()
        except TypeError as exc:
            raise FileNotFoundError("DINOv3 weights path is required") from exc
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"DINOv3 weights file does not exist: {weights_path}"
            )

        try:
            backbones = importlib.import_module(self._PROVIDER_MODULE)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "DINOv3 ConvNeXt-Tiny requires the installed 'lightly_train' "
                "provider; install a compatible lightly-train package"
            ) from exc
        try:
            builder = backbones.dinov3_convnext_tiny
        except AttributeError as exc:
            raise ImportError(
                "the installed lightly_train provider does not expose "
                "dinov3_convnext_tiny"
            ) from exc

        self.in_channels = in_channels
        self.input_norm = input_norm
        self._repeat_to_3ch = in_channels == 1
        self.backbone = builder(
            in_chans=3,
            pretrained=True,
            weights=str(weights_path),
        )

        downsample_layers = getattr(self.backbone, "downsample_layers", None)
        stages = getattr(self.backbone, "stages", None)
        embed_dims = getattr(self.backbone, "embed_dims", None)
        try:
            stage_counts = (
                len(downsample_layers),
                len(stages),
                len(embed_dims),
            )
        except TypeError as exc:
            raise RuntimeError(
                "DINOv3 ConvNeXt-Tiny must expose exactly four stages "
                "through downsample_layers, stages, and embed_dims"
            ) from exc
        if stage_counts != (4, 4, 4):
            raise RuntimeError(
                "DINOv3 ConvNeXt-Tiny must expose exactly four stages "
                "through downsample_layers, stages, and embed_dims; "
                f"got counts {stage_counts}"
            )
        self.projections = nn.ModuleList(
            nn.Conv2d(source, target, kernel_size=1)
            for source, target in zip(embed_dims, (256, 512, 1024, 2048))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(_IMAGENET_STD).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, inputs):
        _validate_image(inputs, self.in_channels)
        if self._repeat_to_3ch:
            inputs = inputs.repeat(1, 3, 1, 1)
        if self.input_norm == "imagenet_from_minus1_1":
            inputs = (inputs + 1.0) * 0.5
            inputs = (inputs - self._imagenet_mean) / self._imagenet_std

        features = []
        hidden = inputs
        for downsample, stage, projection in zip(
            self.backbone.downsample_layers,
            self.backbone.stages,
            self.projections,
        ):
            hidden = downsample(hidden)
            hidden = stage(hidden)
            features.append(projection(hidden))
        if len(features) != 4:
            raise RuntimeError("DINOv3 ConvNeXt-Tiny must produce four stages")
        return tuple(features)


__all__ = ["DINOv3ConvNeXtTinyEncoder", "TinyPyramidEncoder"]
