import math
from pathlib import Path

import pytest
import torch

from ufdr.engine import load_config, validate_config
from ufdr.pucl import pucl_loss
from ufdr.rca import RCA
from ufdr.tgdr import TGDR


def test_validate_config_accepts_canonical_config(canonical_config):
    assert validate_config(canonical_config) is None


def test_load_config_matches_canonical_config(canonical_config):
    config_path = Path(__file__).parents[1] / "configs" / "ufdr.yaml"

    assert load_config(config_path) == canonical_config


def test_validate_config_rejects_nonpositive_pucl_group_size(canonical_config):
    canonical_config["pucl"]["group_size"] = 0

    with pytest.raises(ValueError, match="group_size"):
        validate_config(canonical_config)


def test_validate_config_rejects_nonpositive_batch_size(canonical_config):
    canonical_config["data"]["batch_size"] = 0

    with pytest.raises(ValueError, match="batch_size"):
        validate_config(canonical_config)


@pytest.mark.parametrize("field,value", [("base_l2_lambda", -1e-4), ("max_lambda", 1e-4)])
def test_validate_config_rejects_invalid_tgdr_lambda_range(
    canonical_config, field, value
):
    canonical_config["tgdr"][field] = value

    with pytest.raises(ValueError, match=field):
        validate_config(canonical_config)


def test_validate_config_rejects_unsupported_aux_view(canonical_config):
    canonical_config["model"]["aux_view"] = "flip"

    with pytest.raises(ValueError, match="aux_view"):
        validate_config(canonical_config)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("data", "image_size", "bad"),
        ("pucl", "temperature", None),
        ("train", "epochs", "600"),
        ("data", "batch_size", True),
        ("data", "workers", 1.5),
    ],
)
def test_validate_config_rejects_malformed_scalar_values(
    canonical_config, section, field, value
):
    canonical_config[section][field] = value

    with pytest.raises(ValueError, match=field):
        validate_config(canonical_config)


def test_pucl_returns_finite_scalar_and_backpropagates():
    torch.manual_seed(0)
    embeddings = torch.randn(4, 2, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])

    loss = pucl_loss(embeddings, labels, epoch=3, total_epochs=10)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_pucl_matches_upstream_frozen_oracle():
    embeddings = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.8, 0.3, 0.4]],
            [[0.1, 1.0, 0.2], [0.3, 0.7, 0.6]],
            [[-0.8, 0.2, 0.5], [-0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = pucl_loss(
        embeddings,
        torch.tensor([0, 0, 1]),
        epoch=3,
        total_epochs=10,
        temperature=0.2,
        eps=1e-12,
    )
    loss.backward()

    torch.testing.assert_close(
        loss,
        torch.tensor(1.550558378822716, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    selected = embeddings.grad.flatten()[torch.tensor([0, 1, 6, 11, 17])]
    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                0.10814442445224448,
                -0.8544479500012917,
                -0.9264557203424786,
                0.2913077588784893,
                0.18723881524222324,
            ],
            dtype=torch.float64,
        ),
        rtol=1e-9,
        atol=1e-10,
    )


def test_pucl_matches_upstream_historical_float_label_coercion():
    embeddings = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.8, 0.3, 0.4]],
            [[0.1, 1.0, 0.2], [0.3, 0.7, 0.6]],
            [[-0.8, 0.2, 0.5], [-0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    floating_labels = torch.tensor([0.1, 0.2, 1.1], dtype=torch.float64)
    arguments = {
        "epoch": 3,
        "total_epochs": 10,
        "temperature": 0.2,
        "eps": 1e-12,
    }

    loss = pucl_loss(embeddings, floating_labels, **arguments)
    coerced_loss = pucl_loss(embeddings, floating_labels.long(), **arguments)

    torch.testing.assert_close(loss, coerced_loss, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        loss,
        torch.tensor(1.550558378822716, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_pucl_normalization_uses_upstream_fixed_epsilon():
    embeddings = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.8, 0.3, 0.4]],
            [[0.1, 1.0, 0.2], [0.3, 0.7, 0.6]],
            [[-0.8, 0.2, 0.5], [-0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    embeddings = embeddings * 1e-9
    embeddings.retain_grad()

    loss = pucl_loss(
        embeddings,
        torch.tensor([0, 0, 1]),
        epoch=3,
        total_epochs=10,
        temperature=0.2,
        eps=1e-2,
    )
    loss.backward()

    torch.testing.assert_close(
        loss,
        torch.tensor(0.5537142089565878, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    selected = embeddings.grad.flatten()[torch.tensor([0, 1, 6, 11, 17])]
    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                -2721975.975535833,
                -3298531.2727192394,
                -6501917.012790155,
                2887419.591509508,
                -310829.6700792535,
            ],
            dtype=torch.float64,
        ),
        rtol=1e-9,
        atol=1e-5,
    )


def test_pucl_zero_float16_embeddings_are_finite_and_differentiable():
    embeddings = torch.zeros(4, 2, 8, dtype=torch.float16, requires_grad=True)

    loss = pucl_loss(
        embeddings,
        torch.tensor([0, 0, 1, 1]),
        epoch=5,
        total_epochs=10,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


@pytest.mark.parametrize(
    "embeddings,labels,kwargs,field",
    [
        (torch.randn(4, 8), torch.tensor([0, 0, 1, 1]), {}, "embeddings"),
        (torch.randn(4, 3, 8), torch.tensor([0, 0, 1, 1]), {}, "views"),
        (torch.randn(4, 2, 8), torch.tensor([0, 1, 1]), {}, "labels"),
        (
            torch.randn(4, 2, 8),
            torch.tensor([0, 0, 1, 1]),
            {"temperature": 0.0},
            "temperature",
        ),
        (
            torch.randn(4, 2, 8),
            torch.tensor([0, 0, 1, 1]),
            {"total_epochs": 0},
            "total_epochs",
        ),
    ],
)
def test_pucl_rejects_invalid_arguments(embeddings, labels, kwargs, field):
    arguments = {"epoch": 0, "total_epochs": 10, **kwargs}

    with pytest.raises(ValueError, match=field):
        pucl_loss(embeddings, labels, **arguments)


@pytest.mark.parametrize(
    "labels", [torch.zeros(4, dtype=torch.long), torch.arange(4)]
)
def test_pucl_is_finite_for_degenerate_label_distributions(labels):
    embeddings = torch.randn(4, 2, 8, requires_grad=True)

    loss = pucl_loss(embeddings, labels, epoch=5, total_epochs=10)
    loss.backward()

    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_pucl_no_positive_degeneracy_returns_differentiable_zero():
    embeddings = torch.empty(0, 2, 8, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    loss = pucl_loss(embeddings, labels, epoch=0, total_epochs=10)
    loss.backward()

    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0
    assert loss.device == embeddings.device
    assert embeddings.grad is not None


def test_pucl_curriculum_changes_weak_pair_contribution():
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1]],
            [[0.4, 0.9], [0.2, 1.0]],
            [[-1.0, 0.0], [-0.9, -0.1]],
        ]
    )
    labels = torch.tensor([0, 0, 1])

    initial = pucl_loss(embeddings, labels, epoch=0, total_epochs=10)
    final = pucl_loss(embeddings, labels, epoch=10, total_epochs=10)

    assert not torch.isclose(initial, final)


def test_tgdr_keeps_zero_lambda_with_one_observation():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    regulator = TGDR([parameter])

    regulator.update(1.0, 1.2)

    assert regulator.reliability == 0.5
    assert regulator.adaptive_lambda == 0.0


def test_tgdr_divergent_trajectories_increase_bounded_lambda():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    regulator = TGDR([parameter], base_l2_lambda=0.01, max_lambda=0.03)
    for train_loss, val_loss in zip([1.0, 0.8, 0.6], [1.0, 1.3, 1.7]):
        regulator.update(train_loss, val_loss)

    assert 0.0 < regulator.adaptive_lambda <= 0.03
    assert 0.0 <= regulator.reliability <= 1.0


def test_tgdr_regularization_adds_mean_square_penalty_and_backpropagates():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
    ignored = torch.nn.Parameter(torch.tensor([100.0]), requires_grad=False)
    regulator = TGDR(
        [parameter, ignored], base_l2_lambda=0.01, max_lambda=0.03
    )
    regulator.update(1.0, 1.0)
    regulator.update(0.5, 1.5)
    base_loss = torch.tensor(2.0, requires_grad=True)

    loss = regulator.regularize(base_loss)
    expected = base_loss + regulator.adaptive_lambda * parameter.square().mean()
    loss.backward()

    assert torch.allclose(loss.detach(), expected.detach())
    assert base_loss.grad is not None
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0
    assert ignored.grad is None


def test_tgdr_regularization_promotes_float16_before_squaring():
    parameter = torch.nn.Parameter(torch.full((4,), 256.0, dtype=torch.float16))
    regulator = TGDR(
        [parameter], base_l2_lambda=0.01, max_lambda=0.03
    )
    regulator.update(1.0, 1.0)
    regulator.update(0.5, 1.5)
    base_loss = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)

    loss = regulator.regularize(base_loss)
    expected = base_loss + regulator.adaptive_lambda * parameter.float().square().mean()
    loss.backward()

    assert regulator.adaptive_lambda > 0.0
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss.detach(), expected.detach())
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"window_size": 1}, "window_size"),
        ({"base_l2_lambda": -1e-4}, "base_l2_lambda"),
        ({"base_l2_lambda": 0.02, "max_lambda": 0.01}, "max_lambda"),
    ],
)
def test_tgdr_rejects_invalid_configuration(kwargs, field):
    with pytest.raises(ValueError, match=field):
        TGDR([], **kwargs)


@pytest.mark.parametrize("train_loss,val_loss", [(float("nan"), 1.0), (1.0, float("inf"))])
def test_tgdr_rejects_nonfinite_observations(train_loss, val_loss):
    regulator = TGDR([])

    with pytest.raises(ValueError, match="loss"):
        regulator.update(train_loss, val_loss)


def test_tgdr_rejected_update_is_atomic_and_next_update_works():
    regulator = TGDR([])

    with pytest.raises(ValueError, match="loss"):
        regulator.update(1.0, float("inf"))

    assert list(regulator.train_losses) == []
    assert list(regulator.val_losses) == []
    regulator.update(2.0, 3.0)
    assert list(regulator.train_losses) == [2.0]
    assert list(regulator.val_losses) == [3.0]
    assert regulator.adaptive_lambda == 0.0


def test_tgdr_extreme_finite_observations_keep_statistics_usable():
    regulator = TGDR(
        [], window_size=3, base_l2_lambda=0.01, max_lambda=0.03
    )

    regulator.update(1e308, 8e307)
    regulator.update(9e307, 7e307)

    assert math.isfinite(regulator.reliability)
    assert math.isfinite(regulator.adaptive_lambda)
    assert all(math.isfinite(value) for value in regulator.train_losses)
    assert all(math.isfinite(value) for value in regulator.val_losses)

    regulator.update(1.0, 2.0)

    assert math.isfinite(regulator.reliability)
    assert math.isfinite(regulator.adaptive_lambda)
    assert list(regulator.train_losses) == [1e308, 9e307, 1.0]
    assert list(regulator.val_losses) == [8e307, 7e307, 2.0]


@pytest.mark.parametrize(
    "parameter",
    [
        torch.nn.Parameter(torch.tensor([float("inf")])),
        torch.nn.Parameter(torch.tensor([65504.0], dtype=torch.float16)),
    ],
)
def test_tgdr_zero_lambda_skips_unsafe_penalty(parameter):
    regulator = TGDR([parameter])
    base_loss = torch.tensor(2.0, requires_grad=True)

    result = regulator.regularize(base_loss)

    assert result is base_loss
    assert torch.isfinite(result)
    assert result.item() == 2.0


def test_rca_is_shape_preserving_identity_at_initialization():
    module = RCA(16)
    inputs = torch.randn(2, 16, 5, 7)

    outputs = module(inputs)

    assert outputs.shape == inputs.shape
    assert torch.equal(outputs, inputs)


def test_rca_backpropagates_to_input_and_gamma():
    torch.manual_seed(0)
    module = RCA(8)
    inputs = torch.randn(2, 8, 4, 4, requires_grad=True)

    module(inputs).square().sum().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    assert module.gamma.grad is not None
    assert torch.isfinite(module.gamma.grad).all()
    assert module.gamma.grad.abs().sum() > 0


def test_rca_handles_fewer_than_eight_channels():
    module = RCA(3)
    inputs = torch.randn(2, 3, 4, 4)

    outputs = module(inputs)

    assert module.query_conv.out_channels == 1
    assert module.key_conv.out_channels == 1
    assert outputs.shape == inputs.shape


@pytest.mark.parametrize("inputs", [torch.randn(2, 8, 4), torch.randn(2, 7, 4, 4)])
def test_rca_rejects_invalid_input_shape(inputs):
    module = RCA(8)

    with pytest.raises(ValueError, match="input"):
        module(inputs)
