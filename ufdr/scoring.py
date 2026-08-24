"""Cosine-discrepancy maps and dual-view score fusion."""

import torch
from torch.nn import functional as F

from .losses import _validate_feature_pairs


def _validate_view_input(inputs, mode):
    if not torch.is_tensor(inputs) or inputs.ndim != 4:
        raise ValueError("view input must be a rank-4 NCHW tensor")
    if mode != "rot180":
        raise ValueError("only the 'rot180' auxiliary-view mode is supported")


def apply_aux_view(inputs, mode="rot180"):
    """Rotate an NCHW tensor 180 degrees for UFDR's auxiliary branch."""

    _validate_view_input(inputs, mode)
    return torch.rot90(inputs, 2, dims=(2, 3))


def invert_aux_view(inputs, mode="rot180"):
    """Align a rotated NCHW tensor back to primary-view coordinates."""

    _validate_view_input(inputs, mode)
    return torch.rot90(inputs, 2, dims=(2, 3))


def _validate_output_size(output_size):
    if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
        raise ValueError("output_size must contain explicit (height, width)")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in output_size
    ):
        raise ValueError("output_size height and width must be positive integers")
    return tuple(output_size)


@torch.no_grad()
def feature_discrepancy_maps(decoded, encoded, *, output_size):
    """Average resized, per-level channelwise cosine-distance maps."""

    _validate_feature_pairs(decoded, encoded)
    if any(
        not prediction.is_floating_point() or not target.is_floating_point()
        for prediction, target in zip(decoded, encoded)
    ):
        raise ValueError("decoded and encoded features must be floating-point")
    output_size = _validate_output_size(output_size)
    level_maps = []
    for prediction, target in zip(decoded, encoded):
        distance = 1.0 - F.cosine_similarity(prediction, target, dim=1)
        level_maps.append(
            F.interpolate(
                distance.unsqueeze(1),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        )
    return torch.stack(level_maps, dim=0).mean(dim=0)


@torch.no_grad()
def fused_discrepancy_score(primary_map, auxiliary_map, *, mode="rot180"):
    """Align and average two branch maps, then spatially average image scores."""

    _validate_view_input(primary_map, mode)
    _validate_view_input(auxiliary_map, mode)
    if primary_map.shape != auxiliary_map.shape:
        raise ValueError("primary and auxiliary maps must have matching shapes")
    if primary_map.shape[1] != 1:
        raise ValueError("discrepancy maps must have exactly 1 channel")
    if (
        not primary_map.is_floating_point()
        or not auxiliary_map.is_floating_point()
    ):
        raise ValueError("discrepancy maps must be floating-point")
    if (
        primary_map.device != auxiliary_map.device
        or primary_map.dtype != auxiliary_map.dtype
    ):
        raise ValueError("primary and auxiliary maps must match dtype and device")

    aligned_auxiliary = invert_aux_view(auxiliary_map, mode)
    fused_map = (primary_map + aligned_auxiliary) * 0.5
    image_score = fused_map.mean(dim=(1, 2, 3))
    return fused_map, image_score


__all__ = [
    "apply_aux_view",
    "feature_discrepancy_maps",
    "fused_discrepancy_score",
    "invert_aux_view",
]
