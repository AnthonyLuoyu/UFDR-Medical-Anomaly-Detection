"""Feature reconstruction objectives for UFDR."""

import math

import torch
from torch.nn import functional as F


def _validate_weight(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_feature_pairs(decoded, encoded):
    if not isinstance(decoded, (tuple, list)) or not isinstance(
        encoded, (tuple, list)
    ):
        raise ValueError("decoded and encoded features must be sequences")
    if len(decoded) != 3 or len(encoded) != 3:
        raise ValueError("decoded and encoded features must contain exactly 3 levels")
    for prediction, target in zip(decoded, encoded):
        if not torch.is_tensor(prediction) or not torch.is_tensor(target):
            raise ValueError("decoded and encoded levels must be tensors")
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("decoded and encoded levels must be rank-4 tensors")
        if prediction.shape != target.shape:
            raise ValueError("decoded and encoded levels must have matching shapes")
        if prediction.device != target.device or prediction.dtype != target.dtype:
            raise ValueError("decoded and encoded levels must match dtype and device")
        if prediction.shape[0] == 0:
            raise ValueError("feature batch dimension must be positive")


def feature_reconstruction_loss(
    decoded,
    encoded,
    *,
    cosine_weight=0.5,
    mse_weight=0.05,
):
    """Return the weighted canonical three-level reconstruction objective.

    Encoded features are stop-gradient targets.  The cosine component is the
    sum of per-level flattened cosine losses, and the MSE component is the sum
    of per-level mean-squared errors.  Branch averaging belongs to the caller.
    """

    _validate_weight("cosine_weight", cosine_weight)
    _validate_weight("mse_weight", mse_weight)
    _validate_feature_pairs(decoded, encoded)

    cosine_loss = decoded[0].new_zeros(())
    mse_loss = decoded[0].new_zeros(())
    for prediction, target in zip(decoded, encoded):
        detached_target = target.detach()
        batch_size = prediction.shape[0]
        cosine_loss = cosine_loss + 1.0 - F.cosine_similarity(
            prediction.reshape(batch_size, -1),
            detached_target.reshape(batch_size, -1),
            dim=1,
        ).mean()
        mse_loss = mse_loss + F.mse_loss(prediction, detached_target)
    return cosine_weight * cosine_loss + mse_weight * mse_loss


__all__ = ["feature_reconstruction_loss"]
