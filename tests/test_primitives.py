import importlib
import sys
import types
import warnings

import pytest
import torch
from torch.nn import functional as F

from ufdr.decoder import ResNet50FeatureDecoder
from ufdr.encoder import DINOv3ConvNeXtTinyEncoder, TinyPyramidEncoder
from ufdr.losses import feature_reconstruction_loss
from ufdr.scoring import (
    apply_aux_view,
    feature_discrepancy_maps,
    fused_discrepancy_score,
    invert_aux_view,
)


_DINO_PROVIDER = "lightly_train._models.dinov3.dinov3_src.hub.backbones"


class _FakeDownsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, recorder=None):
        super().__init__()
        self.convolution = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=stride, stride=stride
        )
        self.recorder = recorder

    def forward(self, inputs):
        if self.recorder is not None:
            self.recorder["backbone_input"] = inputs.detach().clone()
        return self.convolution(inputs)


class _FakeConvNeXt(torch.nn.Module):
    def __init__(
        self,
        recorder,
        *,
        downsample_count=4,
        stage_count=4,
        embed_dims=(8, 16, 32, 64),
    ):
        super().__init__()
        channels = (3, 8, 16, 32, 64)
        all_downsamples = [
            _FakeDownsample(
                channels[index],
                channels[index + 1],
                4 if index == 0 else 2,
                recorder if index == 0 else None,
            )
            for index in range(4)
        ]
        self.downsample_layers = torch.nn.ModuleList(
            all_downsamples[:downsample_count]
        )
        self.stages = torch.nn.ModuleList(
            torch.nn.Identity() for _ in range(stage_count)
        )
        self.embed_dims = embed_dims


def _install_fake_dino_provider(
    monkeypatch,
    *,
    downsample_count=4,
    stage_count=4,
    embed_dims=(8, 16, 32, 64),
):
    recorder = {}
    provider = types.ModuleType(_DINO_PROVIDER)

    def build_convnext_tiny(**kwargs):
        recorder["builder_kwargs"] = kwargs
        return _FakeConvNeXt(
            recorder,
            downsample_count=downsample_count,
            stage_count=stage_count,
            embed_dims=embed_dims,
        )

    provider.dinov3_convnext_tiny = build_convnext_tiny
    monkeypatch.setitem(sys.modules, _DINO_PROVIDER, provider)
    return recorder


def _canonical_features(*, batch=2, requires_grad=False):
    return tuple(
        torch.randn(batch, channels, size, size, requires_grad=requires_grad)
        for channels, size in zip((256, 512, 1024, 2048), (16, 8, 4, 2))
    )


def test_tiny_pyramid_encoder_returns_canonical_features_and_backpropagates():
    encoder = TinyPyramidEncoder(in_channels=3)
    inputs = torch.randn(2, 3, 64, 64, requires_grad=True)

    features = encoder(inputs)
    sum(feature.mean() for feature in features).backward()

    assert isinstance(features, tuple)
    assert tuple(feature.shape for feature in features) == (
        (2, 256, 16, 16),
        (2, 512, 8, 8),
        (2, 1024, 4, 4),
        (2, 2048, 2, 2),
    )
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0


def test_tiny_pyramid_encoder_preserves_rng_states_and_is_deterministic():
    torch.random.set_rng_state(torch.Generator().manual_seed(11).get_state())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(29)
    cpu_state_before = torch.random.get_rng_state().clone()
    cuda_states_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if cuda_available
        else []
    )

    first = TinyPyramidEncoder()
    second = TinyPyramidEncoder()

    assert torch.equal(torch.random.get_rng_state(), cpu_state_before)
    if cuda_available:
        cuda_states_after = torch.cuda.get_rng_state_all()
        assert len(cuda_states_after) == len(cuda_states_before)
        for state, expected in zip(cuda_states_after, cuda_states_before):
            assert torch.equal(state, expected)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


@pytest.mark.parametrize(
    "inputs,field",
    [
        (torch.randn(3, 64, 64), "rank-4"),
        (torch.randn(2, 1, 64, 64), "channels"),
        (torch.randn(2, 3, 63, 64), "divisible by 32"),
    ],
)
def test_tiny_pyramid_encoder_rejects_invalid_inputs(inputs, field):
    encoder = TinyPyramidEncoder(in_channels=3)

    with pytest.raises(ValueError, match=field):
        encoder(inputs)


def test_resnet50_feature_decoder_matches_contract_and_backpropagates():
    decoder = ResNet50FeatureDecoder()
    features = _canonical_features(requires_grad=True)

    decoded = decoder(features)
    sum(feature.mean() for feature in decoded).backward()

    assert tuple(feature.shape for feature in decoded) == tuple(
        feature.shape for feature in features[:3]
    )
    for feature in features[1:]:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    used_gradients = [
        parameter.grad
        for parameter in decoder.parameters()
        if parameter.grad is not None
    ]
    assert used_gradients
    assert all(torch.isfinite(gradient).all() for gradient in used_gradients)


@pytest.mark.parametrize(
    "features,field",
    [
        (_canonical_features()[:3], "exactly 4"),
        (
            (
                torch.randn(2, 255, 16, 16),
                *_canonical_features()[1:],
            ),
            "feature 1 channels",
        ),
        (
            (
                _canonical_features()[0],
                torch.randn(2, 512, 7, 8),
                *_canonical_features()[2:],
            ),
            "spatial",
        ),
    ],
)
def test_resnet50_feature_decoder_rejects_invalid_feature_contract(features, field):
    decoder = ResNet50FeatureDecoder()

    with pytest.raises(ValueError, match=field):
        decoder(features)


def test_feature_reconstruction_loss_matches_formula_and_detaches_targets():
    torch.manual_seed(3)
    decoded = tuple(
        torch.randn(2, channels, size, size, dtype=torch.float64, requires_grad=True)
        for channels, size in ((3, 4), (4, 2), (5, 1))
    )
    encoded = tuple(
        torch.randn(2, channels, size, size, dtype=torch.float64, requires_grad=True)
        for channels, size in ((3, 4), (4, 2), (5, 1))
    )

    cosine = sum(
        1.0
        - F.cosine_similarity(
            prediction.reshape(2, -1), target.detach().reshape(2, -1), dim=1
        ).mean()
        for prediction, target in zip(decoded, encoded)
    )
    mse = sum(
        F.mse_loss(prediction, target.detach())
        for prediction, target in zip(decoded, encoded)
    )
    expected = 0.5 * cosine + 0.05 * mse

    actual = feature_reconstruction_loss(decoded, encoded)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    assert all(feature.grad is not None for feature in decoded)
    assert all(torch.isfinite(feature.grad).all() for feature in decoded)
    assert all(feature.grad is None for feature in encoded)


@pytest.mark.parametrize(
    "decoded,encoded,field",
    [
        ((torch.randn(1, 2, 2, 2),), (torch.randn(1, 2, 2, 2),), "3 levels"),
        (
            (
                torch.randn(1, 2, 2, 2),
                torch.randn(1, 2, 1, 1),
                torch.randn(1, 2, 1, 1),
            ),
            (
                torch.randn(1, 3, 2, 2),
                torch.randn(1, 2, 1, 1),
                torch.randn(1, 2, 1, 1),
            ),
            "matching shapes",
        ),
    ],
)
def test_feature_reconstruction_loss_rejects_mismatched_features(
    decoded, encoded, field
):
    with pytest.raises(ValueError, match=field):
        feature_reconstruction_loss(decoded, encoded)


def test_feature_discrepancy_maps_resizes_and_averages_level_cosine_distance():
    torch.manual_seed(7)
    decoded = (
        torch.randn(2, 3, 4, 6),
        torch.randn(2, 4, 2, 3),
        torch.randn(2, 5, 1, 2),
    )
    encoded = tuple(torch.randn_like(feature) for feature in decoded)
    output_size = (7, 9)

    actual = feature_discrepancy_maps(decoded, encoded, output_size=output_size)
    expected = torch.stack(
        [
            F.interpolate(
                (1.0 - F.cosine_similarity(prediction, target, dim=1)).unsqueeze(1),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            for prediction, target in zip(decoded, encoded)
        ],
        dim=0,
    ).mean(dim=0)

    assert actual.shape == (2, 1, 7, 9)
    torch.testing.assert_close(actual, expected)


def test_discrepancy_scoring_has_inference_only_no_grad_semantics():
    decoded = tuple(
        torch.randn(2, channels, size, size, requires_grad=True)
        for channels, size in ((3, 4), (4, 2), (5, 1))
    )
    encoded = tuple(
        torch.randn(2, channels, size, size, requires_grad=True)
        for channels, size in ((3, 4), (4, 2), (5, 1))
    )
    branch_map = feature_discrepancy_maps(
        decoded, encoded, output_size=(4, 4)
    )
    primary = torch.randn(2, 1, 4, 4, requires_grad=True)
    auxiliary = torch.randn(2, 1, 4, 4, requires_grad=True)
    fused_map, score = fused_discrepancy_score(primary, auxiliary)

    for output in (branch_map, fused_map, score):
        assert not output.requires_grad
        assert output.grad_fn is None


def test_feature_discrepancy_maps_rejects_integer_features():
    decoded = tuple(torch.ones(1, 2, 2, 2, dtype=torch.int64) for _ in range(3))
    encoded = tuple(torch.ones_like(feature) for feature in decoded)

    with pytest.raises(ValueError, match="floating-point"):
        feature_discrepancy_maps(decoded, encoded, output_size=(2, 2))


def test_fused_discrepancy_score_rejects_integer_maps():
    primary = torch.ones(1, 1, 2, 3, dtype=torch.int64)
    auxiliary = torch.ones_like(primary)

    with pytest.raises(ValueError, match="floating-point"):
        fused_discrepancy_score(primary, auxiliary)


def test_fused_discrepancy_score_aligns_rotated_branch_before_averaging():
    primary = torch.zeros(1, 1, 2, 3)
    aligned_auxiliary = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]])
    rotated_auxiliary = apply_aux_view(aligned_auxiliary, "rot180")

    fused, score = fused_discrepancy_score(
        primary, rotated_auxiliary, mode="rot180"
    )

    torch.testing.assert_close(fused, aligned_auxiliary / 2.0)
    torch.testing.assert_close(score, fused.mean(dim=(1, 2, 3)))
    assert fused.shape == (1, 1, 2, 3)
    assert score.shape == (1,)


def test_rot180_view_helpers_are_inverse():
    inputs = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    viewed = apply_aux_view(inputs, "rot180")

    torch.testing.assert_close(viewed, torch.rot90(inputs, 2, dims=(2, 3)))
    torch.testing.assert_close(invert_aux_view(viewed, "rot180"), inputs)


@pytest.mark.parametrize("helper", [apply_aux_view, invert_aux_view])
def test_view_helpers_reject_invalid_rank_and_mode(helper):
    with pytest.raises(ValueError, match="rank-4"):
        helper(torch.randn(3, 4, 5), "rot180")
    with pytest.raises(ValueError, match="rot180"):
        helper(torch.randn(1, 1, 4, 5), "rot90")


def test_dinov3_encoder_rejects_invalid_configuration_before_import(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.touch()

    with pytest.raises(ValueError, match="in_channels"):
        DINOv3ConvNeXtTinyEncoder(weights, in_channels=2)
    with pytest.raises(ValueError, match="input_norm"):
        DINOv3ConvNeXtTinyEncoder(weights, input_norm="auto")


def test_dinov3_encoder_reports_missing_weights_before_import(tmp_path):
    missing = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError, match="weights"):
        DINOv3ConvNeXtTinyEncoder(missing)


def test_dinov3_encoder_reports_missing_provider_without_constructing(
    tmp_path, monkeypatch
):
    weights = tmp_path / "weights.pth"
    weights.touch()
    real_import_module = importlib.import_module

    def unavailable_provider(name, package=None):
        if name.startswith("lightly_train"):
            raise ModuleNotFoundError("No module named 'lightly_train'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable_provider)

    with pytest.raises(ImportError, match="lightly_train"):
        DINOv3ConvNeXtTinyEncoder(weights)


@pytest.mark.parametrize("in_channels", [1, 3])
def test_dinov3_encoder_fake_provider_preserves_four_stage_contract(
    tmp_path, monkeypatch, in_channels
):
    recorder = _install_fake_dino_provider(monkeypatch)
    weights = tmp_path / "weights.pth"
    weights.touch()
    encoder = DINOv3ConvNeXtTinyEncoder(
        weights, in_channels=in_channels, input_norm="none"
    )
    inputs = torch.randn(2, in_channels, 64, 64)

    features = encoder(inputs)

    assert tuple(feature.shape for feature in features) == (
        (2, 256, 16, 16),
        (2, 512, 8, 8),
        (2, 1024, 4, 4),
        (2, 2048, 2, 2),
    )
    assert recorder["builder_kwargs"] == {
        "in_chans": 3,
        "pretrained": True,
        "weights": str(weights),
    }
    expected_backbone_input = (
        inputs.repeat(1, 3, 1, 1) if in_channels == 1 else inputs
    )
    torch.testing.assert_close(
        recorder["backbone_input"], expected_backbone_input
    )


@pytest.mark.parametrize(
    "provider_kwargs",
    [
        {"downsample_count": 3},
        {"stage_count": 3},
        {"embed_dims": (8, 16, 32)},
    ],
)
def test_dinov3_encoder_rejects_provider_without_exactly_four_stages(
    tmp_path, monkeypatch, provider_kwargs
):
    _install_fake_dino_provider(monkeypatch, **provider_kwargs)
    weights = tmp_path / "weights.pth"
    weights.touch()

    with pytest.raises(RuntimeError, match="four stages"):
        DINOv3ConvNeXtTinyEncoder(weights)
