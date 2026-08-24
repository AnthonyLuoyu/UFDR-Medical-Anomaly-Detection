import copy

import pytest
import torch
from torch import nn

from ufdr.decoder import ResNet50FeatureDecoder
from ufdr.encoder import TinyPyramidEncoder
from ufdr.losses import feature_reconstruction_loss
from ufdr.model import UFDR
from ufdr.pucl import pucl_loss


class _FastDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.75)) for _ in range(3)]
        )

    def forward(self, features):
        return tuple(
            feature * scale for feature, scale in zip(features[:3], self.scales)
        )


class _SharedParameterDecoder(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.scale = shared

    def forward(self, features):
        return tuple(feature * self.scale for feature in features[:3])


class _SpyEncoder(TinyPyramidEncoder):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().clone())
        return super().forward(inputs)


def _build_fast_model(**kwargs):
    return UFDR(
        encoder=kwargs.pop("encoder", TinyPyramidEncoder()),
        image_size=64,
        decoder_factory=_FastDecoder,
        **kwargs,
    )


def _assert_used_gradients(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


class TestUFDR:
    def test_forward_returns_canonical_outputs_and_backpropagates(self):
        torch.manual_seed(4)
        model = UFDR(encoder=TinyPyramidEncoder(), image_size=64)
        inputs = torch.randn(4, 3, 64, 64)
        labels = torch.tensor([0, 0, 1, 1])

        outputs = model(inputs, labels, epoch=1, total_epochs=2)
        outputs["loss"].backward()

        assert set(outputs) == {
            "loss",
            "loss_base",
            "loss_reconstruction",
            "loss_pucl",
            "tgdr_lambda",
            "anomaly_map",
            "anomaly_score",
        }
        assert outputs["anomaly_map"].shape == (4, 1, 64, 64)
        assert outputs["anomaly_score"].shape == (4,)
        for field in (
            "loss",
            "loss_base",
            "loss_reconstruction",
            "loss_pucl",
            "tgdr_lambda",
        ):
            assert outputs[field].ndim == 0
            assert torch.isfinite(outputs[field])
        assert torch.isfinite(outputs["anomaly_map"]).all()
        assert torch.isfinite(outputs["anomaly_score"]).all()
        assert not outputs["anomaly_map"].requires_grad
        assert not outputs["anomaly_score"].requires_grad
        _assert_used_gradients(model.encoder)
        _assert_used_gradients(model.decoder1)
        _assert_used_gradients(model.decoder2)
        for rca in (model.rca_e2_1, model.rca_e3_1, model.rca_e2_2, model.rca_e3_2):
            assert rca.gamma.grad is not None
            assert torch.isfinite(rca.gamma.grad).all()
            assert rca.gamma.grad.abs().sum() > 0

    def test_forward_matches_canonical_loss_assembly(self):
        torch.manual_seed(8)
        model = _build_fast_model()
        inputs = torch.randn(4, 3, 64, 64)
        labels = torch.tensor([0, 0, 1, 1])

        outputs = model(inputs, labels, epoch=1, total_epochs=2)
        primary = model.encoder(inputs)
        auxiliary = model.encoder(torch.rot90(inputs, 2, dims=(2, 3)))
        decoded_primary = model.decoder1(
            (
                primary[0],
                model.rca_e2_1(primary[1]),
                model.rca_e3_1(primary[2]),
                primary[3],
            )
        )
        decoded_auxiliary = model.decoder2(
            (
                auxiliary[0],
                model.rca_e2_2(auxiliary[1]),
                model.rca_e3_2(auxiliary[2]),
                auxiliary[3],
            )
        )
        reconstruction = 0.5 * (
            feature_reconstruction_loss(decoded_primary, primary[:3])
            + feature_reconstruction_loss(decoded_auxiliary, auxiliary[:3])
        )
        pooled = torch.stack(
            [
                primary[3].mean(dim=(2, 3)),
                auxiliary[3].mean(dim=(2, 3)),
            ],
            dim=1,
        )
        contrastive = pucl_loss(
            pooled,
            labels,
            epoch=1,
            total_epochs=2,
            temperature=0.1,
            eps=1e-12,
        )
        base = reconstruction + 0.01 * contrastive

        torch.testing.assert_close(outputs["loss_reconstruction"], reconstruction)
        torch.testing.assert_close(outputs["loss_pucl"], contrastive)
        torch.testing.assert_close(outputs["loss_base"], base)
        torch.testing.assert_close(outputs["loss"], model.tgdr.regularize(base))
        assert model.cosine_weight == 0.5
        assert model.mse_weight == 0.05
        assert model.pucl_weight == 0.01
        assert model.pucl_temperature == 0.1

    def test_views_share_encoder_but_have_independent_decoders_and_rca(self):
        encoder = _SpyEncoder()
        model = _build_fast_model(encoder=encoder)
        inputs = torch.arange(4 * 3 * 64 * 64, dtype=torch.float32).reshape(
            4, 3, 64, 64
        )

        model(inputs, torch.tensor([0, 0, 1, 1]), epoch=1, total_epochs=2)

        assert model.encoder is encoder
        assert len(encoder.inputs) == 2
        torch.testing.assert_close(encoder.inputs[0], inputs)
        torch.testing.assert_close(
            encoder.inputs[1], torch.rot90(inputs, 2, dims=(2, 3))
        )
        assert model.decoder1 is not model.decoder2
        first_parameters = list(model.decoder1.parameters())
        second_parameters = list(model.decoder2.parameters())
        assert all(
            first is not second
            for first, second in zip(first_parameters, second_parameters)
        )
        rcas = (model.rca_e2_1, model.rca_e3_1, model.rca_e2_2, model.rca_e3_2)
        assert len({id(module) for module in rcas}) == 4

    def test_default_construction_uses_independent_resnet50_decoders(self):
        model = UFDR(encoder=TinyPyramidEncoder(), image_size=64)

        assert isinstance(model.decoder1, ResNet50FeatureDecoder)
        assert isinstance(model.decoder2, ResNet50FeatureDecoder)
        assert model.decoder1 is not model.decoder2
        assert all(
            first is not second
            for first, second in zip(
                model.decoder1.parameters(), model.decoder2.parameters()
            )
        )

    def test_update_tgdr_changes_lambda_and_targets_only_decoder1(self):
        model = _build_fast_model(
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )
        for train_loss, val_loss in zip([1.0, 0.8, 0.6], [1.0, 1.3, 1.7]):
            model.update_tgdr(train_loss, val_loss)
        assert model.tgdr_lambda > 0.0
        assert {id(parameter) for parameter in model.tgdr.parameters} == {
            id(parameter)
            for parameter in model.decoder1.parameters()
            if parameter.requires_grad
        }

        outputs = model(
            torch.randn(4, 3, 64, 64),
            torch.tensor([0, 0, 1, 1]),
            epoch=1,
            total_epochs=2,
        )

        torch.testing.assert_close(
            outputs["loss"], model.tgdr.regularize(outputs["loss_base"])
        )
        torch.testing.assert_close(
            outputs["tgdr_lambda"], outputs["loss"].new_tensor(model.tgdr_lambda)
        )

    @pytest.mark.parametrize(
        "inputs,labels,kwargs,field",
        [
            (torch.randn(3, 64, 64), torch.arange(3), {}, "NCHW"),
            (torch.randn(4, 1, 64, 64), torch.arange(4), {}, "channels"),
            (torch.randn(4, 3, 32, 64), torch.arange(4), {}, "image_size"),
            (torch.randn(4, 3, 64, 64), torch.arange(3), {}, "labels"),
            (
                torch.randn(4, 3, 64, 64),
                torch.arange(4),
                {"epoch": -1},
                "epoch",
            ),
            (
                torch.randn(4, 3, 64, 64),
                torch.arange(4),
                {"total_epochs": 0},
                "total_epochs",
            ),
        ],
    )
    def test_forward_rejects_invalid_inputs(self, inputs, labels, kwargs, field):
        model = _build_fast_model()

        with pytest.raises(ValueError, match=field):
            model(inputs, labels, **kwargs)

    def test_set_epoch_validates_and_supplies_forward_epoch(self):
        model = _build_fast_model(total_epochs=2)
        model.set_epoch(1)

        outputs = model(
            torch.randn(4, 3, 64, 64), torch.tensor([0, 0, 1, 1])
        )

        assert torch.isfinite(outputs["loss"])
        with pytest.raises(ValueError, match="epoch"):
            model.set_epoch(3)

    def test_state_dict_roundtrip_preserves_parameters_and_tgdr_trajectory(self):
        torch.manual_seed(12)
        model = _build_fast_model(
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )
        for train_loss, val_loss in zip([1.0, 0.8, 0.6], [1.0, 1.3, 1.7]):
            model.update_tgdr(train_loss, val_loss)
        state = copy.deepcopy(model.state_dict())
        restored = _build_fast_model(
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )

        restored.load_state_dict(state)

        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)
        assert restored.tgdr_lambda == model.tgdr_lambda
        assert restored.tgdr_reliability == model.tgdr_reliability
        assert list(restored.tgdr.train_losses) == list(model.tgdr.train_losses)
        assert list(restored.tgdr.val_losses) == list(model.tgdr.val_losses)

    def test_load_state_dict_assign_rebinds_tgdr_to_current_decoder1(self):
        model = _build_fast_model(
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )
        old_parameters = list(model.decoder1.parameters())
        state = copy.deepcopy(model.state_dict())

        model.load_state_dict(state, assign=True)

        current_parameters = list(model.decoder1.parameters())
        assert all(
            current is not old
            for current, old in zip(current_parameters, old_parameters)
        )
        assert {id(parameter) for parameter in model.tgdr.parameters} == {
            id(parameter) for parameter in current_parameters
        }
        for train_loss, val_loss in ((1.0, 1.0), (0.5, 1.5)):
            model.update_tgdr(train_loss, val_loss)
        loss = model.tgdr.regularize(torch.tensor(1.0, requires_grad=True))
        loss.backward()
        assert all(parameter.grad is not None for parameter in current_parameters)
        assert all(parameter.grad is None for parameter in old_parameters)

    def test_apply_rebinds_tgdr_after_parameter_replacing_conversion(self):
        model = _build_fast_model()
        old_parameters = list(model.decoder1.parameters())

        model.to("meta")

        current_parameters = list(model.decoder1.parameters())
        assert all(
            current is not old
            for current, old in zip(current_parameters, old_parameters)
        )
        assert {id(parameter) for parameter in model.tgdr.parameters} == {
            id(parameter) for parameter in current_parameters
        }

    def test_forward_rebinds_tgdr_before_regularization(self):
        model = _build_fast_model(
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )
        old_parameter = model.decoder1.scales[0]
        model.decoder1.scales[0] = nn.Parameter(old_parameter.detach().clone())
        for train_loss, val_loss in ((1.0, 1.0), (0.5, 1.5)):
            model.update_tgdr(train_loss, val_loss)

        outputs = model(
            torch.randn(4, 3, 64, 64),
            torch.tensor([0, 0, 1, 1]),
            epoch=1,
            total_epochs=2,
        )
        outputs["loss"].backward()

        assert {id(parameter) for parameter in model.tgdr.parameters} == {
            id(parameter) for parameter in model.decoder1.parameters()
        }
        assert model.decoder1.scales[0].grad is not None
        assert old_parameter.grad is None

    @pytest.mark.parametrize(
        "malformed,field",
        [
            (None, "extra state"),
            ({"version": 2}, "version"),
            ({"version": True}, "version"),
            (
                {
                    "tgdr_train_losses": [1.0, float("nan")],
                },
                "tgdr_train_losses",
            ),
            (
                {
                    "tgdr_val_losses": [1.0, float("inf")],
                },
                "tgdr_val_losses",
            ),
            (
                {
                    "tgdr_train_losses": [1.0],
                    "tgdr_val_losses": [1.0, 2.0],
                },
                "histories",
            ),
            (
                {
                    "tgdr_train_losses": [1.0] * 21,
                    "tgdr_val_losses": [1.0] * 21,
                },
                "histories",
            ),
            ({"tgdr_lambda": float("nan")}, "tgdr_lambda"),
            ({"tgdr_lambda": -0.1}, "tgdr_lambda"),
            ({"tgdr_lambda": 0.031}, "tgdr_lambda"),
            ({"tgdr_reliability": float("inf")}, "tgdr_reliability"),
            ({"tgdr_reliability": -0.1}, "tgdr_reliability"),
            ({"tgdr_reliability": 1.1}, "tgdr_reliability"),
            ({"current_epoch": True}, "current_epoch"),
            ({"current_epoch": -1}, "current_epoch"),
            ({"current_epoch": 1.5}, "current_epoch"),
        ],
    )
    def test_load_state_dict_rejects_malformed_extra_state_atomically(
        self, malformed, field
    ):
        model = _build_fast_model(
            total_epochs=10,
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )
        model.set_epoch(3)
        for train_loss, val_loss in ((1.0, 1.0), (0.5, 1.5)):
            model.update_tgdr(train_loss, val_loss)
        before = copy.deepcopy(model.get_extra_state())
        state = copy.deepcopy(model.state_dict())
        if malformed is None or "version" in malformed:
            state["_extra_state"] = malformed
        else:
            state["_extra_state"].update(malformed)

        with pytest.raises(ValueError, match=field):
            model.load_state_dict(state)

        assert model.get_extra_state() == before

    def test_load_state_dict_accepts_legacy_tensor_only_state_strictly(self):
        source = _build_fast_model()
        legacy_state = copy.deepcopy(source.state_dict())
        legacy_state.pop("_extra_state")
        restored = _build_fast_model(
            total_epochs=10,
            tgdr_base_l2_lambda=0.01,
            tgdr_max_lambda=0.03,
        )
        restored.set_epoch(3)
        for train_loss, val_loss in ((1.0, 1.0), (0.5, 1.5)):
            restored.update_tgdr(train_loss, val_loss)

        restored.load_state_dict(legacy_state, strict=True)

        assert restored.current_epoch == 0
        assert restored.tgdr_lambda == 0.0
        assert restored.tgdr_reliability == 0.5
        assert list(restored.tgdr.train_losses) == []
        assert list(restored.tgdr.val_losses) == []
        for expected, actual in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_load_state_dict_legacy_migration_keeps_tensor_keys_strict(self):
        model = _build_fast_model()
        legacy_state = copy.deepcopy(model.state_dict())
        legacy_state.pop("_extra_state")
        legacy_state["unexpected_tensor"] = torch.tensor(1.0)

        with pytest.raises(RuntimeError, match="Unexpected key"):
            model.load_state_dict(legacy_state, strict=True)

    def test_constructor_rejects_decoders_with_shared_parameters(self):
        shared = nn.Parameter(torch.tensor(1.0))

        with pytest.raises(ValueError, match="independent parameters"):
            UFDR(
                encoder=TinyPyramidEncoder(),
                image_size=64,
                decoder_factory=lambda: _SharedParameterDecoder(shared),
            )
