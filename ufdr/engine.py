"""Configuration, training, and evaluation entry points for UFDR."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import UFDRFolderDataset, build_dataloaders
from .model import UFDR


_REQUIRED_SECTIONS = ("data", "model", "pucl", "tgdr", "train")


def _require_integer(section, field):
    value = section.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_numeric(section, field):
    value = section.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite numeric value")
    return value


def load_config(path):
    """Load a YAML configuration file and validate its contents."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    validate_config(config)
    return config


def validate_config(config):
    """Validate the canonical UFDR configuration schema."""
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    for section in _REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} section is required")

    data = config["data"]
    model = config["model"]
    pucl = config["pucl"]
    tgdr = config["tgdr"]
    train = config["train"]

    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    device = config.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", device):
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")

    for section_name, section, field in (
        ("data", data, "root"),
        ("model", model, "weights"),
        ("train", train, "output_dir"),
    ):
        if not isinstance(section.get(field), str) or not section[field].strip():
            raise ValueError(f"{section_name}.{field} must be a non-empty path string")

    for field in ("image_size", "channels", "batch_size"):
        if _require_integer(data, field) <= 0:
            raise ValueError(f"data.{field} must be positive")
    if _require_integer(data, "workers") < 0:
        raise ValueError("data.workers must be non-negative")
    if data["channels"] not in (1, 3):
        raise ValueError("data.channels must be 1 or 3")

    if model.get("aux_view") != "rot180":
        raise ValueError("model.aux_view must be rot180")
    if model.get("encoder") != "dinov3_convnext_tiny":
        raise ValueError("model.encoder must be dinov3_convnext_tiny")
    if _require_integer(model, "freeze_encoder_epochs") < 0:
        raise ValueError("model.freeze_encoder_epochs must be non-negative")
    for field in ("cosine_weight", "mse_weight"):
        if _require_numeric(model, field) < 0:
            raise ValueError(f"model.{field} must be non-negative")

    if _require_numeric(pucl, "temperature") <= 0:
        raise ValueError("pucl.temperature must be positive")
    if _require_numeric(pucl, "weight") < 0:
        raise ValueError("pucl.weight must be non-negative")
    if _require_numeric(pucl, "eps") <= 0:
        raise ValueError("pucl.eps must be positive")
    if _require_integer(pucl, "group_size") < 1:
        raise ValueError("pucl.group_size must be at least 1")
    if pucl.get("label_mode") not in {"class", "group", "instance"}:
        raise ValueError("pucl.label_mode must be class, group, or instance")

    if _require_integer(tgdr, "window_size") < 2:
        raise ValueError("tgdr.window_size must be at least 2")
    base_l2_lambda = _require_numeric(tgdr, "base_l2_lambda")
    max_lambda = _require_numeric(tgdr, "max_lambda")
    if base_l2_lambda < 0:
        raise ValueError("tgdr.base_l2_lambda must be non-negative")
    if base_l2_lambda > max_lambda:
        raise ValueError("tgdr.max_lambda must be at least base_l2_lambda")
    if tgdr.get("target") != "decoder1":
        raise ValueError("tgdr.target must be decoder1")
    if tgdr.get("reliability") != "corr_gap":
        raise ValueError("tgdr.reliability must be corr_gap")

    if _require_integer(train, "epochs") <= 0:
        raise ValueError("train.epochs must be positive")
    for field in ("lr_encoder", "lr_projection", "lr_decoder"):
        if _require_numeric(train, field) <= 0:
            raise ValueError(f"train.{field} must be positive")


def resolve_paths(config, config_path):
    """Return a copy whose portable path fields are anchored to the config."""

    validate_config(config)
    resolved = deepcopy(config)
    config_path = Path(config_path).expanduser().resolve()
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for section, field in (
        ("data", "root"),
        ("model", "weights"),
        ("train", "output_dir"),
    ):
        path = Path(resolved[section][field]).expanduser()
        if not path.is_absolute():
            path = base / path
        resolved[section][field] = str(path.resolve())
    return resolved


def build_model(config):
    """Construct the canonical UFDR model after cheap local validation."""

    validate_config(config)
    weights = Path(config["model"]["weights"]).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights file does not exist: {weights}")

    data = config["data"]
    model = config["model"]
    pucl = config["pucl"]
    tgdr = config["tgdr"]
    train_config = config["train"]
    return UFDR(
        weights=str(weights),
        in_channels=data["channels"],
        image_size=data["image_size"],
        cosine_weight=model["cosine_weight"],
        mse_weight=model["mse_weight"],
        pucl_weight=pucl["weight"],
        pucl_temperature=pucl["temperature"],
        pucl_eps=pucl["eps"],
        total_epochs=train_config["epochs"],
        tgdr_window_size=tgdr["window_size"],
        tgdr_base_l2_lambda=tgdr["base_l2_lambda"],
        tgdr_max_lambda=tgdr["max_lambda"],
    )


def _device_from_config(config) -> torch.device:
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is not available: {config['device']}"
        )
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {device.index}")
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _optimizer_for(model, config):
    rates = config["train"]
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("model has no trainable parameters")

    if isinstance(model, UFDR):
        groups = {"encoder_backbone": [], "encoder_proj": [], "decoder": []}
        for name, parameter in named_parameters:
            if name.startswith("encoder.backbone."):
                groups["encoder_backbone"].append(parameter)
            elif name.startswith("encoder.projections."):
                groups["encoder_proj"].append(parameter)
            else:
                groups["decoder"].append(parameter)
        parameters = [
            {
                "params": groups["encoder_backbone"],
                "lr": rates["lr_encoder"],
                "name": "encoder_backbone",
            },
            {
                "params": groups["encoder_proj"],
                "lr": rates["lr_projection"],
                "name": "encoder_proj",
            },
            {
                "params": groups["decoder"],
                "lr": rates["lr_decoder"],
                "name": "decoder",
            },
        ]
        parameters = [group for group in parameters if group["params"]]
    else:
        parameters = [
            {
                "params": [parameter for _, parameter in named_parameters],
                "lr": rates["lr_decoder"],
                "name": "decoder",
            }
        ]
    return torch.optim.AdamW(parameters, weight_decay=1e-4)


def _scheduler_for(optimizer, config):
    """Build the source-aligned 10% warmup plus cosine schedule."""

    epochs = config["train"]["epochs"]
    warmup_epochs = max(1, int(epochs * 0.1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=1e-7,
    )
    return scheduler, warmup_epochs


def _step_learning_rate(
    optimizer, scheduler, warmup_epochs: int, *, completed_epoch: int
) -> None:
    """Update LR at epoch end using the source's one-based epoch convention."""

    if completed_epoch <= warmup_epochs:
        warmup_factor = completed_epoch / warmup_epochs
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr * warmup_factor
    else:
        scheduler.step()


def _set_encoder_trainable(model, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", None)
    if backbone is not None:
        for parameter in backbone.parameters():
            parameter.requires_grad_(trainable)


def _pucl_labels(batch, config, device) -> torch.Tensor:
    """Build PUCL labels with the canonical class/group/instance policy."""

    class_labels = batch.get("contrast_label")
    if not torch.is_tensor(class_labels) or class_labels.ndim != 1:
        raise RuntimeError("batch contrast_label must be a rank-1 tensor")
    class_labels = class_labels.to(device=device, dtype=torch.long, non_blocking=True)
    mode = config["pucl"]["label_mode"]
    if mode == "class":
        return class_labels

    batch_size = class_labels.shape[0]
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    if mode == "instance":
        return labels
    group_size = config["pucl"]["group_size"]
    if group_size >= batch_size:
        group_size = max(1, batch_size // 2)
    return labels // group_size


def _run_loss_epoch(model, loader, config, device, epoch, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_base_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            contrast_labels = _pucl_labels(batch, config, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(images, contrast_labels, epoch=epoch)
            loss = output.get("loss") if isinstance(output, dict) else None
            if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError("model must return a finite scalar 'loss'")
            loss_base = output.get("loss_base", loss)
            if (
                not torch.is_tensor(loss_base)
                or loss_base.ndim != 0
                or not torch.isfinite(loss_base)
            ):
                raise RuntimeError(
                    "model must return a finite scalar 'loss_base' when provided"
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_base_loss += float(loss_base.detach()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("data loader produced no samples")
    mean_loss = total_loss / total_samples
    mean_base_loss = total_base_loss / total_samples
    if not math.isfinite(mean_loss) or not math.isfinite(mean_base_loss):
        raise RuntimeError("epoch loss is not finite")
    return mean_loss, mean_base_loss


def _atomic_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train(config, model=None):
    """Train UFDR and save the best validation checkpoint."""

    validate_config(config)
    device = _device_from_config(config)
    _seed_everything(config["seed"], device)
    loaders = build_dataloaders(config)
    model = build_model(config) if model is None else model
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch module or None")
    model = model.to(device)
    optimizer = _optimizer_for(model, config)
    scheduler, warmup_epochs = _scheduler_for(optimizer, config)

    history = []
    best_val_loss = math.inf
    checkpoint_path = Path(config["train"]["output_dir"]) / "best.pt"
    freeze_epochs = config["model"]["freeze_encoder_epochs"]
    for epoch in range(config["train"]["epochs"]):
        _set_encoder_trainable(model, epoch >= freeze_epochs)
        train_loss, train_base_loss = _run_loss_epoch(
            model, loaders["train"], config, device, epoch, optimizer=optimizer
        )
        val_loss, val_base_loss = _run_loss_epoch(
            model, loaders["val"], config, device, epoch
        )
        update_tgdr = getattr(model, "update_tgdr", None)
        if callable(update_tgdr):
            update_tgdr(train_base_loss, val_base_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(record)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _atomic_checkpoint(
                checkpoint_path,
                {
                    "version": 1,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                },
            )
        # ``epoch`` is zero-based here; the source worker steps at the end of
        # one-based epochs, so ``epoch + 1`` preserves that timing.
        _step_learning_rate(
            optimizer,
            scheduler,
            warmup_epochs,
            completed_epoch=epoch + 1,
        )
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoint_path),
    }


def _load_checkpoint(path, device):
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("version")) is not int
        or checkpoint["version"] != 1
        or type(checkpoint.get("epoch")) is not int
        or checkpoint["epoch"] < 0
        or not isinstance(checkpoint.get("model_state"), dict)
    ):
        raise RuntimeError(f"checkpoint has an invalid format: {checkpoint_path}")
    return checkpoint


def evaluate(config, checkpoint, model=None):
    """Evaluate a strict checkpoint on the two-class test folder."""

    validate_config(config)
    checkpoint_data = _load_checkpoint(checkpoint, torch.device("cpu"))
    device = _device_from_config(config)
    _seed_everything(config["seed"], device)
    test_dataset = UFDRFolderDataset(
        config["data"]["root"],
        "test",
        image_size=config["data"]["image_size"],
        channels=config["data"]["channels"],
    )
    generator = torch.Generator().manual_seed(config["seed"])
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    model = build_model(config) if model is None else model
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch module or None")
    model = model.to(device)
    try:
        model.load_state_dict(checkpoint_data["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint is incompatible with the model: {exc}") from exc

    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            contrast_labels = _pucl_labels(batch, config, device)
            output = model(
                images,
                contrast_labels,
                epoch=config["train"]["epochs"],
            )
            anomaly_score = output.get("anomaly_score") if isinstance(output, dict) else None
            if not torch.is_tensor(anomaly_score) or anomaly_score.ndim != 1:
                raise RuntimeError("model must return a rank-1 'anomaly_score'")
            if anomaly_score.shape[0] != images.shape[0]:
                raise RuntimeError(
                    "model anomaly_score length must match the input batch"
                )
            if not torch.isfinite(anomaly_score).all():
                raise RuntimeError("model returned non-finite anomaly scores")
            labels.extend(batch["label"].tolist())
            scores.extend(anomaly_score.detach().cpu().tolist())
    if len(set(labels)) != 2:
        raise ValueError("test data must contain both normal and anomaly classes")
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "num_samples": len(labels),
    }


__all__ = [
    "build_model",
    "evaluate",
    "load_config",
    "resolve_paths",
    "train",
    "validate_config",
]
