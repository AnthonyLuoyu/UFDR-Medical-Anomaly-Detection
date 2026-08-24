"""Integration tests for the portable UFDR data and engine surfaces."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from PIL import Image
from torch import nn
import yaml

import ufdr.engine as engine
from ufdr.data import UFDRFolderDataset, build_dataloaders
from ufdr.engine import (
    build_model,
    evaluate,
    resolve_paths,
    train,
    validate_config,
)
from ufdr.model import UFDR


PACKAGE_ROOT = Path(__file__).parents[1]


def _save_image(path: Path, value: int, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = value if mode == "L" else (value, value, value)
    Image.new(mode, (7, 5), color=color).save(path)


def _make_folder_tree(root: Path) -> None:
    _save_image(root / "train" / "normal" / "z.PNG", 0)
    _save_image(root / "train" / "normal" / "nested" / "a.jpg", 64)
    _save_image(root / "val" / "normal" / "v.bmp", 96)
    _save_image(root / "test" / "normal" / "n.tif", 0)
    _save_image(root / "test" / "anomaly" / "a.tiff", 255)


def _runtime_config(canonical_config, root: Path, output_dir: Path) -> dict:
    config = deepcopy(canonical_config)
    config["seed"] = 7
    config["device"] = "cpu"
    config["data"].update(
        {
            "root": str(root),
            "image_size": 8,
            "channels": 3,
            "batch_size": 1,
            "workers": 0,
        }
    )
    config["model"]["weights"] = str(root / "unused-weights.pth")
    config["model"]["freeze_encoder_epochs"] = 1
    config["train"].update({"epochs": 1, "output_dir": str(output_dir)})
    return config


class ToyUFDR(nn.Module):
    """One-parameter UFDR-shaped model used to keep engine tests fast."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.tgdr_updates: list[tuple[float, float]] = []
        self.seen_contrast_labels: list[torch.Tensor] = []

    def forward(self, image, contrast_labels, *, epoch=None):
        self.seen_contrast_labels.append(contrast_labels.detach().cpu().clone())
        scores = image.mean(dim=(1, 2, 3)) + self.bias
        loss_base = (scores - contrast_labels.float()).square().mean()
        anomaly_map = image.mean(dim=1, keepdim=True)
        zero = loss_base * 0.0
        return {
            "loss": loss_base,
            "loss_base": loss_base,
            "loss_reconstruction": loss_base,
            "loss_pucl": zero,
            "tgdr_lambda": zero.detach(),
            "anomaly_map": anomaly_map.detach(),
            "anomaly_score": scores.detach(),
        }

    def update_tgdr(self, train_loss, val_loss):
        self.tgdr_updates.append((float(train_loss), float(val_loss)))


class FakeCanonicalUFDR(UFDR):
    """Canonical-shaped shell that never constructs DINO or ResNet."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(2, 2)
        self.encoder.projections = nn.ModuleList([nn.Linear(2, 2)])
        self.rca_e2_1 = nn.Linear(2, 2)
        self.decoder1 = nn.Linear(2, 2)


class WrongLengthScoreToy(ToyUFDR):
    def forward(self, image, contrast_labels, *, epoch=None):
        output = super().forward(image, contrast_labels, epoch=epoch)
        output["anomaly_score"] = output["anomaly_score"][:-1]
        return output


class RegularizedToyUFDR(ToyUFDR):
    """Expose distinct total/base losses so TGDR feedback can be tested."""

    def forward(self, image, contrast_labels, *, epoch=None):
        self.seen_contrast_labels.append(contrast_labels.detach().cpu().clone())
        loss_base = self.bias * 0.0 + 2.0
        regularizer = (self.bias - 5.0).square() + 1.0
        loss = loss_base + regularizer
        scores = image.mean(dim=(1, 2, 3)) + self.bias
        zero = loss * 0.0
        return {
            "loss": loss,
            "loss_base": loss_base,
            "loss_reconstruction": loss_base,
            "loss_pucl": zero,
            "tgdr_lambda": zero.detach(),
            "anomaly_map": image.mean(dim=1, keepdim=True).detach(),
            "anomaly_score": scores.detach(),
        }


class InvalidBaseLossToy(ToyUFDR):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def forward(self, image, contrast_labels, *, epoch=None):
        output = super().forward(image, contrast_labels, epoch=epoch)
        output["loss_base"] = output["loss"].new_tensor(self.value)
        return output


def test_folder_dataset_recurses_sorts_and_normalizes_rgb(tmp_path):
    root = tmp_path / "data"
    _make_folder_tree(root)

    train_set = UFDRFolderDataset(root, "train", image_size=8, channels=3)
    test_set = UFDRFolderDataset(root, "test", image_size=8, channels=3)

    assert len(train_set) == 2
    assert [Path(train_set[index]["path"]).name for index in range(2)] == [
        "a.jpg",
        "z.PNG",
    ]
    sample = train_set[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert -1.0 <= sample["image"].min() <= sample["image"].max() <= 1.0
    assert sample["label"] == 0
    assert sample["contrast_label"] == 0
    assert Path(sample["path"]).is_file()
    assert [test_set[index]["label"] for index in range(2)] == [0, 1]


def test_folder_dataset_supports_grayscale_and_jpeg_extension(tmp_path):
    root = tmp_path / "data"
    _save_image(root / "train" / "normal" / "gray.jpeg", 128, mode="L")

    sample = UFDRFolderDataset(root, "train", image_size=8, channels=1)[0]

    assert sample["image"].shape == (1, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["image"].abs().max() < 0.01


@pytest.mark.parametrize(
    "split,image_size,channels,match",
    [
        ("unknown", 8, 3, "split"),
        ("train", 0, 3, "image_size"),
        ("train", True, 3, "image_size"),
        ("train", 8, 2, "channels"),
        ("train", 8, 1.0, "channels"),
        ("train", 8, 3.0, "channels"),
        ("train", 8, True, "channels"),
    ],
)
def test_folder_dataset_rejects_invalid_arguments(
    tmp_path, split, image_size, channels, match
):
    with pytest.raises(ValueError, match=match):
        UFDRFolderDataset(
            tmp_path / "data",
            split,
            image_size=image_size,
            channels=channels,
        )


def test_folder_dataset_reports_missing_and_empty_split(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="train/normal"):
        UFDRFolderDataset(root, "train", image_size=8)

    (root / "train" / "normal").mkdir(parents=True)
    with pytest.raises(ValueError, match="no supported images"):
        UFDRFolderDataset(root, "train", image_size=8)


@pytest.mark.parametrize("present_class", ["normal", "anomaly"])
def test_test_split_requires_both_classes(tmp_path, present_class):
    root = tmp_path / "data"
    _save_image(root / "test" / present_class / "one.png", 0)

    expected_missing = "anomaly" if present_class == "normal" else "normal"
    with pytest.raises((FileNotFoundError, ValueError), match=expected_missing):
        UFDRFolderDataset(root, "test", image_size=8)


def test_build_dataloaders_is_deterministic_for_same_seed(
    tmp_path, canonical_config
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "out")

    first = build_dataloaders(config)
    second = build_dataloaders(config)
    first_order = [Path(batch["path"][0]).name for batch in first["train"]]
    second_order = [Path(batch["path"][0]).name for batch in second["train"]]

    assert first_order == second_order
    assert set(first) == {"train", "val", "test"}
    assert first["train"].drop_last is False
    assert first["train"].pin_memory is False


def test_build_dataloaders_uses_independent_split_generators(
    tmp_path, canonical_config
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "out")
    source = torch.Generator().manual_seed(123)

    loaders = build_dataloaders(config, generator=source)

    split_generators = [loaders[split].generator for split in ("train", "val", "test")]
    assert len({id(generator) for generator in split_generators}) == 3
    train_state = split_generators[0].get_state().clone()
    list(loaders["val"])
    assert torch.equal(split_generators[0].get_state(), train_state)


def test_canonical_optimizer_uses_three_precise_parameter_groups(canonical_config):
    model = FakeCanonicalUFDR()

    optimizer = engine._optimizer_for(model, canonical_config)
    rates_by_parameter = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    for name, parameter in model.named_parameters():
        if name.startswith("encoder.backbone."):
            expected = canonical_config["train"]["lr_encoder"]
        elif name.startswith("encoder.projections."):
            expected = canonical_config["train"]["lr_projection"]
        else:
            expected = canonical_config["train"]["lr_decoder"]
        assert rates_by_parameter[id(parameter)] == expected, name
    assert isinstance(optimizer, torch.optim.AdamW)
    assert {group["name"] for group in optimizer.param_groups} == {
        "encoder_backbone",
        "encoder_proj",
        "decoder",
    }
    assert all(group["weight_decay"] == pytest.approx(1e-4) for group in optimizer.param_groups)
    rca_parameter = next(model.rca_e2_1.parameters())
    assert rates_by_parameter[id(rca_parameter)] == canonical_config["train"]["lr_decoder"]


def test_injected_model_uses_named_adamw_decoder_group(canonical_config):
    optimizer = engine._optimizer_for(ToyUFDR(), canonical_config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "decoder"
    assert optimizer.param_groups[0]["lr"] == canonical_config["train"]["lr_decoder"]
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_freeze_only_toggles_canonical_encoder_backbone():
    model = FakeCanonicalUFDR()

    engine._set_encoder_trainable(model, False)

    assert all(not parameter.requires_grad for parameter in model.encoder.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.projections.parameters())
    assert all(parameter.requires_grad for parameter in model.rca_e2_1.parameters())
    engine._set_encoder_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.encoder.backbone.parameters())


def test_warmup_and_cosine_scheduler_follow_one_based_epoch_end_mapping(
    canonical_config,
):
    canonical_config["train"]["epochs"] = 20
    optimizer = engine._optimizer_for(FakeCanonicalUFDR(), canonical_config)
    scheduler, warmup_epochs = engine._scheduler_for(optimizer, canonical_config)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    assert warmup_epochs == 2
    assert scheduler.T_max == 18
    optimizer.step()
    engine._step_learning_rate(optimizer, scheduler, warmup_epochs, completed_epoch=1)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [value * 0.5 for value in base_lrs]
    )
    optimizer.step()
    engine._step_learning_rate(optimizer, scheduler, warmup_epochs, completed_epoch=2)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(base_lrs)

    optimizer.step()
    engine._step_learning_rate(optimizer, scheduler, warmup_epochs, completed_epoch=3)
    expected_first_cosine = [
        1e-7 + (value - 1e-7) * (1.0 + math.cos(math.pi / 18.0)) / 2.0
        for value in base_lrs
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        expected_first_cosine
    )
    for completed_epoch in range(4, 21):
        optimizer.step()
        engine._step_learning_rate(
            optimizer, scheduler, warmup_epochs, completed_epoch=completed_epoch
        )
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-7] * len(base_lrs), abs=1e-12
    )


def test_single_epoch_scheduler_has_safe_tmax(canonical_config):
    canonical_config["train"]["epochs"] = 1
    optimizer = engine._optimizer_for(ToyUFDR(), canonical_config)

    scheduler, warmup_epochs = engine._scheduler_for(optimizer, canonical_config)

    assert warmup_epochs == 1
    assert scheduler.T_max == 1
    engine._step_learning_rate(optimizer, scheduler, warmup_epochs, completed_epoch=1)


@pytest.mark.parametrize(
    "mode,group_size,expected",
    [
        ("class", 2, [4, 9, 4, 1, 2]),
        ("group", 2, [0, 0, 1, 1, 2]),
        ("instance", 2, [0, 1, 2, 3, 4]),
    ],
)
def test_pucl_label_modes(mode, group_size, expected, canonical_config):
    canonical_config["pucl"].update({"label_mode": mode, "group_size": group_size})
    batch = {"contrast_label": torch.tensor([4, 9, 4, 1, 2])}

    labels = engine._pucl_labels(batch, canonical_config, torch.device("cpu"))

    assert labels.tolist() == expected
    assert labels.dtype == torch.long


@pytest.mark.parametrize(
    "batch_size,group_size,expected",
    [(4, 4, [0, 0, 1, 1]), (4, 99, [0, 0, 1, 1]), (1, 8, [0])],
)
def test_pucl_group_mode_adapts_group_size_at_batch_boundary(
    batch_size, group_size, expected, canonical_config
):
    canonical_config["pucl"].update(
        {"label_mode": "group", "group_size": group_size}
    )
    batch = {"contrast_label": torch.arange(batch_size) + 10}

    labels = engine._pucl_labels(batch, canonical_config, torch.device("cpu"))

    assert labels.tolist() == expected


def test_resolve_paths_uses_package_root_for_configs_directory(
    tmp_path, canonical_config
):
    package = tmp_path / "portable"
    config_path = package / "configs" / "ufdr.yaml"
    config_path.parent.mkdir(parents=True)
    original = deepcopy(canonical_config)

    resolved = resolve_paths(canonical_config, config_path)

    assert canonical_config == original
    assert resolved["data"]["root"] == str((package / "data/example").resolve())
    assert resolved["model"]["weights"] == str(
        (package / "weights/dinov3_convnext_tiny_lvd1689m.pth").resolve()
    )
    assert resolved["train"]["output_dir"] == str(
        (package / "outputs/ufdr").resolve()
    )


def test_resolve_paths_uses_config_parent_outside_configs_directory(
    tmp_path, canonical_config
):
    config_path = tmp_path / "settings" / "experiment.yaml"

    resolved = resolve_paths(canonical_config, config_path)

    assert resolved["data"]["root"] == str(
        (config_path.parent / "data/example").resolve()
    )


@pytest.mark.parametrize(
    "section,field,value,match",
    [
        (None, "seed", True, "seed"),
        (None, "seed", 1.5, "seed"),
        (None, "device", 1, "device"),
        (None, "device", "mps", "device"),
        ("data", "root", 1, "data.root"),
        ("model", "weights", None, "model.weights"),
        ("train", "output_dir", [], "train.output_dir"),
        ("data", "channels", 2, "channels"),
    ],
)
def test_validate_config_rejects_runtime_fields(
    canonical_config, section, field, value, match
):
    target = canonical_config if section is None else canonical_config[section]
    target[field] = value

    with pytest.raises(ValueError, match=match):
        validate_config(canonical_config)


@pytest.mark.parametrize(
    "section,field,value,match",
    [
        ("model", "cosine_weight", -0.1, "cosine_weight"),
        ("model", "mse_weight", math.nan, "mse_weight"),
        ("pucl", "temperature", math.inf, "temperature"),
        ("pucl", "weight", -0.1, "weight"),
        ("pucl", "eps", 0.0, "eps"),
        ("tgdr", "base_l2_lambda", math.nan, "base_l2_lambda"),
        ("tgdr", "max_lambda", math.inf, "max_lambda"),
        ("train", "lr_encoder", math.nan, "lr_encoder"),
        ("train", "lr_projection", math.inf, "lr_projection"),
    ],
)
def test_validate_config_rejects_nonfinite_or_out_of_range_numeric_values(
    canonical_config, section, field, value, match
):
    canonical_config[section][field] = value

    with pytest.raises(ValueError, match=match):
        validate_config(canonical_config)


@pytest.mark.parametrize(
    "section,field",
    [
        ("model", "cosine_weight"),
        ("model", "mse_weight"),
        ("pucl", "weight"),
        ("pucl", "eps"),
        ("train", "lr_decoder"),
    ],
)
def test_validate_config_missing_numeric_field_is_value_error(
    canonical_config, section, field
):
    del canonical_config[section][field]

    with pytest.raises(ValueError, match=field):
        validate_config(canonical_config)


def test_build_model_checks_weights_before_importing_provider(
    tmp_path, canonical_config, monkeypatch
):
    config = deepcopy(canonical_config)
    missing = tmp_path / "missing.pth"
    config["model"]["weights"] = str(missing)

    def forbidden_import(*args, **kwargs):
        raise AssertionError("provider import must not run for missing weights")

    monkeypatch.setattr("ufdr.encoder.importlib.import_module", forbidden_import)

    with pytest.raises(FileNotFoundError, match=str(missing)):
        build_model(config)


def test_train_one_cpu_epoch_writes_minimal_checkpoint_and_updates_tgdr(
    tmp_path, canonical_config
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    output_dir = tmp_path / "outputs"
    config = _runtime_config(canonical_config, root, output_dir)
    model = ToyUFDR()

    result = train(config, model=model)

    checkpoint_path = output_dir / "best.pt"
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"version", "epoch", "model_state"}
    assert checkpoint["version"] == 1
    assert model.tgdr_updates and len(model.tgdr_updates) == 1
    assert set(result) == {"history", "best_val_loss", "best_checkpoint"}
    assert result["best_checkpoint"] == str(checkpoint_path)
    assert len(result["history"]) == 1
    assert all(
        math.isfinite(result["history"][0][key])
        for key in ("train_loss", "val_loss")
    )


def test_train_updates_tgdr_from_base_losses_but_logs_total_losses(
    tmp_path, canonical_config
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    model = RegularizedToyUFDR()

    result = train(config, model=model)

    assert model.bias.item() > 0.0
    assert model.tgdr_updates == pytest.approx([(2.0, 2.0)])
    record = result["history"][0]
    assert record["train_loss"] > model.tgdr_updates[0][0]
    assert record["val_loss"] > model.tgdr_updates[0][1]
    assert result["best_val_loss"] == pytest.approx(record["val_loss"])


@pytest.mark.parametrize("invalid_base", [[1.0], float("nan")])
def test_train_rejects_non_scalar_or_nonfinite_loss_base(
    tmp_path, canonical_config, invalid_base
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")

    with pytest.raises(RuntimeError, match="loss_base"):
        train(config, model=InvalidBaseLossToy(invalid_base))


def test_train_clips_all_gradients_to_half_norm(
    tmp_path, canonical_config, monkeypatch
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    model = ToyUFDR()
    calls = []

    def capture_clip(parameters, max_norm):
        calls.append((list(parameters), max_norm))
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)

    train(config, model=model)

    assert calls
    assert all(max_norm == pytest.approx(0.5) for _, max_norm in calls)
    assert all(any(parameter is model.bias for parameter in parameters) for parameters, _ in calls)


def test_train_uses_configured_group_pucl_labels(tmp_path, canonical_config):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    config["data"]["batch_size"] = 2
    config["pucl"].update({"label_mode": "group", "group_size": 1})
    model = ToyUFDR()

    train(config, model=model)

    assert model.seen_contrast_labels[0].tolist() == [0, 1]


def test_evaluate_returns_exact_metrics_for_injected_model(
    tmp_path, canonical_config
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    model = ToyUFDR()
    checkpoint_path = tmp_path / "toy.pt"
    torch.save(
        {"version": 1, "epoch": 0, "model_state": model.state_dict()},
        checkpoint_path,
    )

    metrics = evaluate(config, checkpoint_path, model=ToyUFDR())

    assert set(metrics) == {"auc", "average_precision", "num_samples"}
    assert metrics["num_samples"] == 2
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_evaluate_rejects_incompatible_checkpoint(tmp_path, canonical_config):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(
        {"version": 1, "epoch": 0, "model_state": {"wrong": torch.tensor(1)}},
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="checkpoint.*incompatible"):
        evaluate(config, checkpoint_path, model=ToyUFDR())


def test_load_checkpoint_uses_safe_weights_only_mode(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    observed = {}

    def fake_load(path, **kwargs):
        observed.update(kwargs)
        return {"version": 1, "epoch": 0, "model_state": {}}

    monkeypatch.setattr(torch, "load", fake_load)

    assert engine._load_checkpoint(checkpoint_path, torch.device("cpu"))["version"] == 1
    assert observed["weights_only"] is True


@pytest.mark.parametrize(
    "checkpoint",
    [
        {},
        {"version": 2, "epoch": 0, "model_state": {}},
        {"version": True, "epoch": 0, "model_state": {}},
        {"version": 1, "epoch": True, "model_state": {}},
        {"version": 1, "epoch": -1, "model_state": {}},
        {"version": 1, "epoch": 0, "model_state": []},
    ],
)
def test_load_checkpoint_rejects_invalid_format(tmp_path, checkpoint):
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="invalid format"):
        engine._load_checkpoint(checkpoint_path, torch.device("cpu"))


def test_evaluate_rejects_one_class_test_data(tmp_path, canonical_config):
    root = tmp_path / "data"
    _save_image(root / "test" / "normal" / "n.png", 0)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    checkpoint_path = tmp_path / "toy.pt"
    model = ToyUFDR()
    torch.save(
        {"version": 1, "epoch": 0, "model_state": model.state_dict()},
        checkpoint_path,
    )

    with pytest.raises((FileNotFoundError, ValueError), match="anomaly"):
        evaluate(config, checkpoint_path, model=model)


def test_evaluate_rejects_anomaly_score_length_mismatch(
    tmp_path, canonical_config
):
    root = tmp_path / "data"
    _make_folder_tree(root)
    config = _runtime_config(canonical_config, root, tmp_path / "outputs")
    config["data"]["batch_size"] = 2
    model = WrongLengthScoreToy()
    checkpoint_path = tmp_path / "toy.pt"
    torch.save(
        {"version": 1, "epoch": 0, "model_state": model.state_dict()},
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="anomaly_score.*batch"):
        evaluate(config, checkpoint_path, model=model)


@pytest.mark.parametrize("script_name", ["train.py", "test.py"])
def test_cli_help_is_portable_outside_package_cwd(tmp_path, script_name):
    completed = subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    if script_name == "test.py":
        assert "--checkpoint" in completed.stdout


@pytest.mark.parametrize("script_name", ["train.py", "test.py"])
def test_cli_missing_config_has_actionable_error(tmp_path, script_name):
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / script_name),
        "--config",
        str(tmp_path / "missing.yaml"),
    ]
    if script_name == "test.py":
        command.extend(["--checkpoint", str(tmp_path / "missing.pt")])

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "missing.yaml" in completed.stderr
    assert "error" in completed.stderr.lower()


def test_test_cli_reports_missing_checkpoint_before_model_build(
    tmp_path, canonical_config
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(canonical_config), encoding="utf-8")
    missing = tmp_path / "missing.pt"

    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts" / "test.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(missing),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert str(missing) in completed.stderr
