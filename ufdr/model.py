"""Canonical two-view UFDR model assembly."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import losses, pucl, scoring
from .decoder import ResNet50FeatureDecoder
from .encoder import DINOv3ConvNeXtTinyEncoder
from .rca import RCA
from .tgdr import TGDR


class UFDR(nn.Module):
    """Assemble the canonical shared-encoder, dual-decoder UFDR route.

    TGDR trajectory statistics are runtime controller state rather than tensors,
    so :meth:`get_extra_state` includes them in regular ``state_dict``
    checkpoints.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module | None = None,
        weights: str | None = None,
        in_channels: int = 3,
        input_norm: str = "imagenet_from_minus1_1",
        image_size: int = 256,
        cosine_weight: float = 0.5,
        mse_weight: float = 0.05,
        pucl_weight: float = 0.01,
        pucl_temperature: float = 0.1,
        pucl_eps: float = 1e-12,
        total_epochs: int = 600,
        tgdr_window_size: int = 20,
        tgdr_base_l2_lambda: float = 3e-4,
        tgdr_max_lambda: float = 0.03,
        tgdr_eps: float = 1e-6,
        decoder_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self._validate_positive_integer("image_size", image_size)
        self._validate_positive_integer("total_epochs", total_epochs)
        self._validate_nonnegative_weight("cosine_weight", cosine_weight)
        self._validate_nonnegative_weight("mse_weight", mse_weight)
        self._validate_nonnegative_weight("pucl_weight", pucl_weight)
        self._validate_positive_scalar("pucl_temperature", pucl_temperature)
        self._validate_positive_scalar("pucl_eps", pucl_eps)

        if encoder is None:
            encoder = DINOv3ConvNeXtTinyEncoder(
                weights=weights,
                in_channels=in_channels,
                input_norm=input_norm,
            )
        if not isinstance(encoder, nn.Module):
            raise ValueError("encoder must be a torch module or None")

        self.encoder = encoder
        self.in_channels = self._encoder_channels(encoder, in_channels)
        self.image_size = image_size
        self.cosine_weight = float(cosine_weight)
        self.mse_weight = float(mse_weight)
        self.pucl_weight = float(pucl_weight)
        self.pucl_temperature = float(pucl_temperature)
        self.pucl_eps = float(pucl_eps)
        self.total_epochs = total_epochs
        self.current_epoch = 0

        factory = ResNet50FeatureDecoder if decoder_factory is None else decoder_factory
        if not callable(factory):
            raise ValueError("decoder_factory must be callable")
        self.decoder1 = factory()
        self.decoder2 = factory()
        if not isinstance(self.decoder1, nn.Module) or not isinstance(
            self.decoder2, nn.Module
        ):
            raise ValueError("decoder_factory must return torch modules")
        if self.decoder1 is self.decoder2:
            raise ValueError("decoder_factory must return independent modules")
        decoder1_parameter_ids = {
            id(parameter) for parameter in self.decoder1.parameters()
        }
        decoder2_parameter_ids = {
            id(parameter) for parameter in self.decoder2.parameters()
        }
        if decoder1_parameter_ids & decoder2_parameter_ids:
            raise ValueError("decoders must have independent parameters")

        self.rca_e2_1 = RCA(512)
        self.rca_e3_1 = RCA(1024)
        self.rca_e2_2 = RCA(512)
        self.rca_e3_2 = RCA(1024)
        self.tgdr = TGDR(
            self.decoder1.parameters(),
            window_size=tgdr_window_size,
            base_l2_lambda=tgdr_base_l2_lambda,
            max_lambda=tgdr_max_lambda,
            eps=tgdr_eps,
        )

    @property
    def tgdr_lambda(self) -> float:
        """Return the controller's current adaptive regularization weight."""

        return self.tgdr.adaptive_lambda

    @property
    def tgdr_reliability(self) -> float:
        """Return the controller's current trajectory reliability estimate."""

        return self.tgdr.reliability

    def forward(
        self,
        inputs: Tensor,
        labels: Tensor,
        *,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> dict[str, Tensor]:
        """Compute the canonical training objective and inference-only scores."""

        epoch, total_epochs = self._validate_forward_inputs(
            inputs, labels, epoch, total_epochs
        )
        labels = torch.as_tensor(labels, device=inputs.device).to(dtype=torch.long)
        auxiliary_inputs = scoring.apply_aux_view(inputs)

        primary = self._encode(inputs)
        auxiliary = self._encode(auxiliary_inputs)
        decoded_primary = self.decoder1(
            (
                primary[0],
                self.rca_e2_1(primary[1]),
                self.rca_e3_1(primary[2]),
                primary[3],
            )
        )
        decoded_auxiliary = self.decoder2(
            (
                auxiliary[0],
                self.rca_e2_2(auxiliary[1]),
                self.rca_e3_2(auxiliary[2]),
                auxiliary[3],
            )
        )

        primary_reconstruction = losses.feature_reconstruction_loss(
            decoded_primary,
            primary[:3],
            cosine_weight=self.cosine_weight,
            mse_weight=self.mse_weight,
        )
        auxiliary_reconstruction = losses.feature_reconstruction_loss(
            decoded_auxiliary,
            auxiliary[:3],
            cosine_weight=self.cosine_weight,
            mse_weight=self.mse_weight,
        )
        loss_reconstruction = 0.5 * (
            primary_reconstruction + auxiliary_reconstruction
        )

        pooled_features = torch.stack(
            (
                F.adaptive_avg_pool2d(primary[3], 1).flatten(1),
                F.adaptive_avg_pool2d(auxiliary[3], 1).flatten(1),
            ),
            dim=1,
        )
        loss_pucl = pucl.pucl_loss(
            pooled_features,
            labels,
            epoch=epoch,
            total_epochs=total_epochs,
            temperature=self.pucl_temperature,
            eps=self.pucl_eps,
        )
        loss_base = loss_reconstruction + self.pucl_weight * loss_pucl
        self._rebind_tgdr_parameters()
        loss = self.tgdr.regularize(loss_base)

        primary_map = scoring.feature_discrepancy_maps(
            decoded_primary, primary[:3], output_size=(self.image_size, self.image_size)
        )
        auxiliary_map = scoring.feature_discrepancy_maps(
            decoded_auxiliary,
            auxiliary[:3],
            output_size=(self.image_size, self.image_size),
        )
        anomaly_map, anomaly_score = scoring.fused_discrepancy_score(
            primary_map, auxiliary_map
        )
        return {
            "loss": loss,
            "loss_base": loss_base,
            "loss_reconstruction": loss_reconstruction,
            "loss_pucl": loss_pucl,
            "tgdr_lambda": loss.new_tensor(self.tgdr_lambda),
            "anomaly_map": anomaly_map,
            "anomaly_score": anomaly_score,
        }

    def update_tgdr(self, train_loss: Tensor | float, val_loss: Tensor | float) -> None:
        """Update TGDR from one pair of train and validation observations."""

        self.tgdr.update(train_loss, val_loss)

    def set_epoch(self, epoch: int) -> None:
        """Set the default PUCL curriculum epoch used by :meth:`forward`."""

        self._validate_epoch(epoch, self.total_epochs)
        self.current_epoch = epoch

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load tensors and migrate legacy checkpoints without extra state."""

        migrated_state = state_dict
        if isinstance(state_dict, Mapping) and "_extra_state" not in state_dict:
            migrated_state = OrderedDict(state_dict)
            metadata = getattr(state_dict, "_metadata", None)
            if metadata is not None:
                migrated_state._metadata = metadata
            migrated_state["_extra_state"] = self._default_extra_state()
        result = super().load_state_dict(
            migrated_state, strict=strict, assign=assign
        )
        self._rebind_tgdr_parameters()
        return result

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        if hasattr(self, "tgdr"):
            self._rebind_tgdr_parameters()
        return module

    def get_extra_state(self) -> dict[str, Any]:
        """Return checkpointable TGDR trajectories and curriculum position."""

        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "tgdr_train_losses": list(self.tgdr.train_losses),
            "tgdr_val_losses": list(self.tgdr.val_losses),
            "tgdr_lambda": self.tgdr.adaptive_lambda,
            "tgdr_reliability": self.tgdr.reliability,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        """Restore TGDR trajectories saved by :meth:`get_extra_state`."""

        normalized = self._normalize_extra_state(state)
        self.tgdr.train_losses.clear()
        self.tgdr.val_losses.clear()
        self.tgdr.train_losses.extend(normalized["tgdr_train_losses"])
        self.tgdr.val_losses.extend(normalized["tgdr_val_losses"])
        self.tgdr._adaptive_lambda = normalized["tgdr_lambda"]
        self.tgdr._reliability = normalized["tgdr_reliability"]
        self.current_epoch = normalized["current_epoch"]

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_nonnegative_weight(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")

    @staticmethod
    def _validate_positive_scalar(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    @staticmethod
    def _encoder_channels(encoder: nn.Module, fallback: int) -> int:
        channels = getattr(encoder, "in_channels", fallback)
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("encoder in_channels must be a positive integer")
        return channels

    @staticmethod
    def _validate_epoch(epoch: int, total_epochs: int) -> None:
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or epoch > total_epochs
        ):
            raise ValueError("epoch must be between 0 and total_epochs")

    @staticmethod
    def _finite_checkpoint_scalar(name: str, value: Any) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be finite")
        return float(value)

    def _default_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": 0,
            "tgdr_train_losses": [],
            "tgdr_val_losses": [],
            "tgdr_lambda": 0.0,
            "tgdr_reliability": 0.5,
        }

    def _normalize_extra_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise ValueError("extra state must be a mapping")
        snapshot = dict(state)
        version = snapshot.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError("version must be the supported integer value 1")

        required = (
            "current_epoch",
            "tgdr_train_losses",
            "tgdr_val_losses",
            "tgdr_lambda",
            "tgdr_reliability",
        )
        for field in required:
            if field not in snapshot:
                raise ValueError(f"{field} is required in extra state")

        histories = {}
        for field in ("tgdr_train_losses", "tgdr_val_losses"):
            values = snapshot[field]
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{field} must be a finite scalar sequence")
            histories[field] = [
                self._finite_checkpoint_scalar(field, value) for value in values
            ]
        train_losses = histories["tgdr_train_losses"]
        val_losses = histories["tgdr_val_losses"]
        window_size = self.tgdr.train_losses.maxlen
        if (
            len(train_losses) != len(val_losses)
            or len(train_losses) > window_size
        ):
            raise ValueError(
                "TGDR histories must have matching lengths within the window size"
            )

        adaptive_lambda = self._finite_checkpoint_scalar(
            "tgdr_lambda", snapshot["tgdr_lambda"]
        )
        if not 0.0 <= adaptive_lambda <= self.tgdr.max_lambda:
            raise ValueError("tgdr_lambda must be between 0 and max_lambda")
        reliability = self._finite_checkpoint_scalar(
            "tgdr_reliability", snapshot["tgdr_reliability"]
        )
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("tgdr_reliability must be between 0 and 1")

        current_epoch = snapshot["current_epoch"]
        if (
            isinstance(current_epoch, bool)
            or not isinstance(current_epoch, int)
            or current_epoch < 0
            or current_epoch > self.total_epochs
        ):
            raise ValueError(
                "current_epoch must be between 0 and total_epochs"
            )
        return {
            "version": 1,
            "current_epoch": current_epoch,
            "tgdr_train_losses": train_losses,
            "tgdr_val_losses": val_losses,
            "tgdr_lambda": adaptive_lambda,
            "tgdr_reliability": reliability,
        }

    def _rebind_tgdr_parameters(self) -> None:
        self.tgdr.rebind(self.decoder1.parameters())

    def _validate_forward_inputs(
        self,
        inputs: Tensor,
        labels: Tensor,
        epoch: int | None,
        total_epochs: int | None,
    ) -> tuple[int, int]:
        if not torch.is_tensor(inputs) or inputs.ndim != 4:
            raise ValueError("inputs must be an NCHW rank-4 tensor")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"input channels must match configured channels ({self.in_channels})"
            )
        if inputs.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"input spatial dimensions must match image_size ({self.image_size})"
            )
        labels_tensor = torch.as_tensor(labels)
        if labels_tensor.ndim != 1 or labels_tensor.shape[0] != inputs.shape[0]:
            raise ValueError("labels must have one entry per input sample")
        effective_total = self.total_epochs if total_epochs is None else total_epochs
        self._validate_positive_integer("total_epochs", effective_total)
        effective_epoch = self.current_epoch if epoch is None else epoch
        self._validate_epoch(effective_epoch, effective_total)
        return effective_epoch, effective_total

    def _encode(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.encoder(inputs)
        if not isinstance(features, tuple) or len(features) != 4:
            raise ValueError("encoder must return a tuple of exactly 4 features")
        return features


__all__ = ["UFDR"]
