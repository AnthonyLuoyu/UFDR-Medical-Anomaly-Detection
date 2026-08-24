"""Trajectory-Guided Decoder Regulation (TGDR)."""

import math
from collections import deque

import torch


class TGDR:
    """Adapt decoder regularization from recent train/validation trajectories.

    The correlation-gap calculation follows the historical
    ``SelfCalibratedRegularizer`` reference implementation.
    """

    def __init__(
        self,
        parameters,
        window_size=20,
        base_l2_lambda=3e-4,
        max_lambda=0.03,
        eps=1e-6,
    ):
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise ValueError("window_size must be an integer of at least 2")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._validate_scalar("base_l2_lambda", base_l2_lambda)
        self._validate_scalar("max_lambda", max_lambda)
        self._validate_scalar("eps", eps)
        if base_l2_lambda < 0:
            raise ValueError("base_l2_lambda must be non-negative")
        if max_lambda < base_l2_lambda:
            raise ValueError("max_lambda must be at least base_l2_lambda")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.parameters = [p for p in parameters if p.requires_grad]
        self.train_losses = deque(maxlen=window_size)
        self.val_losses = deque(maxlen=window_size)
        self.base_l2_lambda = float(base_l2_lambda)
        self.max_lambda = float(max_lambda)
        self.eps = float(eps)
        self._adaptive_lambda = 0.0
        self._reliability = 0.5

    @staticmethod
    def _validate_scalar(field, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be finite")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")

    @property
    def adaptive_lambda(self):
        return self._adaptive_lambda

    @property
    def reliability(self):
        return self._reliability

    def rebind(self, parameters):
        """Replace regularization targets without resetting trajectory state."""
        self.parameters = [p for p in parameters if p.requires_grad]

    @staticmethod
    def _as_finite_loss(value):
        try:
            converted = float(value.detach().item() if torch.is_tensor(value) else value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("train_loss and val_loss must be finite scalars") from exc
        if not math.isfinite(converted):
            raise ValueError("train_loss and val_loss must be finite scalars")
        return converted

    @torch.no_grad()
    def update(self, train_loss, val_loss):
        """Record one loss pair and recompute the correlation-gap statistics."""
        train_value = self._as_finite_loss(train_loss)
        val_value = self._as_finite_loss(val_loss)
        self.train_losses.append(train_value)
        self.val_losses.append(val_value)
        if len(self.train_losses) < 2:
            self._reliability = 0.5
            self._adaptive_lambda = 0.0
            return

        train = torch.tensor(list(self.train_losses), dtype=torch.float64)
        validation = torch.tensor(list(self.val_losses), dtype=torch.float64)

        def zscore(values):
            mean = values.mean()
            std = values.std(unbiased=False)
            return (values - mean) / (std + self.eps)

        covariance = torch.cov(torch.stack([zscore(train), zscore(validation)]))
        covariance = covariance + torch.eye(2, dtype=covariance.dtype) * self.eps
        denominator = torch.sqrt(
            covariance[0, 0] * covariance[1, 1] + self.eps
        )
        correlation = torch.clamp(
            torch.abs(covariance[0, 1] / denominator), 0.0, 1.0
        )

        train_mean = train.abs().mean()
        relative_gap = (validation - train).abs().mean() / (train_mean + self.eps)
        uncertainty = (1.0 - correlation) + torch.clamp(relative_gap, 0.0, 10.0)
        uncertainty = torch.nan_to_num(
            uncertainty, nan=11.0, posinf=11.0, neginf=0.0
        )
        reliability = 1.0 / (1.0 + uncertainty)

        self._reliability = float(torch.clamp(reliability, 0.0, 1.0).item())
        adaptive = self.base_l2_lambda * float(uncertainty.item())
        self._adaptive_lambda = max(0.0, min(adaptive, self.max_lambda))

    def regularize(self, base_loss):
        """Add the current mean-square parameter penalty to ``base_loss``."""
        if not torch.is_tensor(base_loss):
            raise ValueError("base_loss must be a tensor")
        if self._adaptive_lambda <= 0:
            return base_loss
        penalty_dtype = (
            base_loss.dtype
            if base_loss.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
        penalty = torch.zeros((), device=base_loss.device, dtype=penalty_dtype)
        for parameter in self.parameters:
            stable_parameter = (
                parameter.float()
                if parameter.dtype in (torch.float16, torch.bfloat16)
                else parameter
            )
            term = stable_parameter.square().mean()
            penalty = penalty + term.to(
                device=base_loss.device, dtype=penalty_dtype
            )
        return base_loss + penalty.new_tensor(self._adaptive_lambda) * penalty


__all__ = ["TGDR"]
